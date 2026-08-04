# SPDX-License-Identifier: Apache-2.0
"""Persist the engine's KV cache event stream to ``blocks.jsonl``.

Runs a SUB socket in a background thread of the driver process. Sequence gaps
are recovered through the publisher's replay socket, because a missing
``BlockStored`` would silently corrupt the offline radix-tree reconstruction.
"""

import json
import threading
import time
from pathlib import Path

import zmq
from msgspec.msgpack import Decoder

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)


def _hexify(block_hash):
    return block_hash.hex() if isinstance(block_hash, bytes) else block_hash


def _as_record(event) -> dict:
    if isinstance(event, BlockStored):
        return {
            "type": "BlockStored",
            "block_hashes": [_hexify(h) for h in event.block_hashes],
            "parent_block_hash": _hexify(event.parent_block_hash)
            if event.parent_block_hash is not None
            else None,
            "token_ids": event.token_ids,
            "block_size": event.block_size,
            "group_idx": event.group_idx,
        }
    if isinstance(event, BlockRemoved):
        return {
            "type": "BlockRemoved",
            "block_hashes": [_hexify(h) for h in event.block_hashes],
            "group_idx": event.group_idx,
        }
    if isinstance(event, AllBlocksCleared):
        return {"type": "AllBlocksCleared"}
    return {"type": type(event).__name__}


class KVEventCollector:
    """Subscribes to the engine's KV events and appends them as JSONL.

    Args:
        out_path: Destination ``blocks.jsonl``.
        endpoint: Publisher PUB endpoint, e.g. ``tcp://localhost:5557``.
        replay_endpoint: Publisher ROUTER endpoint used to fill sequence gaps.
        topic: Publisher topic to subscribe to.
    """

    def __init__(
        self,
        out_path: str | Path,
        endpoint: str = "tcp://localhost:5557",
        replay_endpoint: str | None = "tcp://localhost:5558",
        topic: str = "kv-events",
    ):
        self.out_path = Path(out_path)
        self.endpoint = endpoint
        self.replay_endpoint = replay_endpoint
        self.topic = topic

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self.num_events = 0
        self.num_missed = 0
        self.last_seq = -1

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.out_path.open("a", buffering=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def stop(self, drain_s: float = 3.0) -> None:
        """Waits ``drain_s`` for in-flight events, then joins the thread."""
        time.sleep(drain_s)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write(self, ts: float, seq: int, event) -> None:
        record = _as_record(event)
        record["ts"] = ts
        record["seq"] = seq
        record["recv_ts"] = time.time()
        self._file.write(json.dumps(record) + "\n")
        self.num_events += 1

    def _write_batch(self, payload: bytes, seq: int, decoder: Decoder) -> None:
        batch = decoder.decode(payload)
        for event in batch.events:
            self._write(batch.ts, seq, event)

    def _run(self) -> None:
        decoder = Decoder(type=KVEventBatch)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)

        replay = None
        if self.replay_endpoint:
            replay = ctx.socket(zmq.REQ)
            replay.connect(self.replay_endpoint)

        self._ready.set()
        try:
            while not self._stop.is_set():
                if not sub.poll(50):
                    continue
                _, seq_bytes, payload = sub.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                if self.last_seq >= 0 and seq > self.last_seq + 1 and replay:
                    self._replay_gap(replay, decoder, seq)
                self._write_batch(payload, seq, decoder)
                self.last_seq = seq
        finally:
            sub.close(linger=0)
            if replay is not None:
                replay.close(linger=0)

    def _replay_gap(self, replay, decoder: Decoder, current_seq: int) -> None:
        missed = current_seq - self.last_seq - 1
        self.num_missed += missed
        replay.send((self.last_seq + 1).to_bytes(8, "big"))
        poller = zmq.Poller()
        poller.register(replay, zmq.POLLIN)
        while poller.poll(timeout=500):
            _, seq_bytes, payload = replay.recv_multipart()
            if not payload:
                break
            replay_seq = int.from_bytes(seq_bytes, "big")
            if replay_seq > self.last_seq:
                self._write_batch(payload, replay_seq, decoder)
                self.last_seq = replay_seq
                self.num_missed -= 1
            if replay_seq >= current_seq - 1:
                break
