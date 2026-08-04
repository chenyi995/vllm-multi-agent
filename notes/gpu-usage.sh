#!/usr/bin/env bash
#
# gpu-usage.sh — 显示 Slurm 集群 GPU 占用表，以及某个账号(默认自己)的占用情况
#
# 用法:
#   ./gpu-usage.sh              # 集群 GPU 占用 + 自己(cw636)的占用 (跑一次)
#   ./gpu-usage.sh <user>       # 换成别的账号
#   ./gpu-usage.sh -a           # 全局占用: 每节点谁在占 + 用户占用排行
#   ./gpu-usage.sh -w [秒]      # 监视模式, 默认每 10s 刷新, 按 Ctrl-C 退出 (可配合 -a)
#   ./gpu-usage.sh -p [user]    # 额外做物理 nvidia-smi 探测(会 srun 起小作业, 慢; 监视模式下忽略)
#   ./gpu-usage.sh -h           # 帮助
#
set -uo pipefail

# 宽度计算依赖 UTF-8 locale
case "${LC_ALL:-${LANG:-}}" in
  *[Uu][Tt][Ff]*8*) : ;;
  *) export LC_ALL=C.utf8 ;;
esac

# ---- 参数解析 ----------------------------------------------------------------
PHYS=0; WATCH=0; ALL=0; INTERVAL=10
ME="$USER"
prev=""
for a in "$@"; do
  # -w 后面若紧跟数字则作为刷新间隔
  if [[ "$prev" == "-w" || "$prev" == "--watch" ]] && [[ "$a" =~ ^[0-9]+$ ]]; then
    INTERVAL="$a"; prev=""; continue
  fi
  case "$a" in
    -p|--phys)   PHYS=1 ;;
    -w|--watch)  WATCH=1 ;;
    -a|--all)    ALL=1 ;;
    -h|--help)   sed -n '3,12p' "$0"; exit 0 ;;
    -*)          echo "未知选项: $a"; exit 1 ;;
    *)           ME="$a" ;;
  esac
  prev="$a"
done
# 监视模式下不做物理探测(太慢)
[[ $WATCH -eq 1 ]] && PHYS=0

# ---- 颜色(尊重 NO_COLOR / 非终端) --------------------------------------------
if { [[ -t 1 ]] || [[ $WATCH -eq 1 ]]; } && [[ -z "${NO_COLOR:-}" ]]; then
  B=$'\e[1m'; DIM=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; C=$'\e[36m'; N=$'\e[0m'
else
  B=''; DIM=''; G=''; Y=''; R=''; C=''; N=''
fi

# ---- 显示宽度对齐工具(中文/全角=2 格) ----------------------------------------
dwidth() {
  local s=$1 n=${#s} i cp w=0
  for ((i=0; i<n; i++)); do
    printf -v cp '%d' "'${s:i:1}"
    if (( cp>=0x1100 && ( cp<=0x115F \
        || (cp>=0x2E80 && cp<=0xA4CF) || (cp>=0xAC00 && cp<=0xD7A3) \
        || (cp>=0xF900 && cp<=0xFAFF) || (cp>=0xFE30 && cp<=0xFE4F) \
        || (cp>=0xFF00 && cp<=0xFF60) || (cp>=0xFFE0 && cp<=0xFFE6) ) )); then
      w=$((w+2)); else w=$((w+1)); fi
  done
  printf '%d' "$w"
}
pad() {  # pad <text> <width> <l|r> [color]
  local s=$1 w=$2 al=${3:-l} col=${4:-} dw p sp body
  dw=$(dwidth "$s"); p=$(( w - dw )); (( p<0 )) && p=0
  printf -v sp '%*s' "$p" ''
  if [[ $al == r ]]; then body="$sp$s"; else body="$s$sp"; fi
  [[ -n $col ]] && body="$col$body$N"
  printf '%s' "$body"
}
rep() { local ch=$1 n=$2 o=''; while (( n-- > 0 )); do o+=$ch; done; printf '%s' "$o"; }
hr()  { local w o=' '; for w in "$@"; do o+="$(rep '─' "$w") "; done; printf '%s%s%s\n' "$DIM" "${o% }" "$N"; }

# ---- 节点 -> GPU 型号(本集群固定, 如扩容请改这里) ----------------------------
gpu_model() {
  case "$1" in
    node1|node2|node3|node4) echo "RTX A5000 24G" ;;
    node5|node6)             echo "L40S 46G" ;;
    *)                       echo "unknown" ;;
  esac
}
is_big() { [[ "$(gpu_model "$1")" == L40S* ]]; }   # 单卡显存 >32G
# 已知型号(用于账号占用明细的固定顺序)
MODELS=("RTX A5000 24G" "L40S 46G")
# 短名(用于全局占用视图)
gpu_short() { case "$(gpu_model "$1")" in RTX*A5000*) echo A5000 ;; L40S*) echo L40S ;; *) echo "?" ;; esac; }
MODELS_SHORT=("A5000" "L40S")

# =============================================================================
# 渲染函数(集群表 + 账号表)
# =============================================================================
render() {
  # ---------- 1) 集群 GPU 占用表 ----------
  local W1=8 W2=15 W3=5 W4=5 W5=6 W6=13
  printf "%s╔══ 集群 GPU 占用 (Slurm 分配) ══╗%s\n" "$B" "$N"
  printf " %s %s %s %s %s %s\n" \
    "$(pad 节点 $W1 l "$DIM")" "$(pad 卡型 $W2 l "$DIM")" \
    "$(pad 总数 $W3 r "$DIM")" "$(pad 已用 $W4 r "$DIM")" \
    "$(pad 空闲 $W5 r "$DIM")" "$(pad 状态 $W6 l "$DIM")"
  hr $W1 $W2 $W3 $W4 $W5 $W6

  local tot_total=0 tot_alloc=0 tot_free=0 big_total=0 big_free_avail=0 avail_free_total=0
  local line node state total alloc free model usable fcol fstr scol
  while IFS= read -r line; do
    node=$(grep -oP 'NodeName=\K[^ ]+' <<<"$line"); [[ -z "$node" ]] && continue
    state=$(grep -oP 'State=\K[^ ]+'    <<<"$line")
    total=$(grep -oP 'CfgTRES=[^ ]*gres/gpu=\K[0-9]+'   <<<"$line"); total=${total:-0}
    alloc=$(grep -oP 'AllocTRES=[^ ]*gres/gpu=\K[0-9]+' <<<"$line"); alloc=${alloc:-0}
    [[ "$total" -eq 0 ]] && continue
    free=$(( total - alloc )); model=$(gpu_model "$node")
    usable=1
    [[ "$state" == *DRAIN* || "$state" == *DOWN* || "$state" == *DRNG* || "$state" == *NO_RESPOND* ]] && usable=0
    tot_total=$(( tot_total+total )); tot_alloc=$(( tot_alloc+alloc )); tot_free=$(( tot_free+free ))
    if is_big "$node"; then big_total=$(( big_total+total )); [[ $usable -eq 1 ]] && big_free_avail=$(( big_free_avail+free )); fi
    [[ $usable -eq 1 ]] && avail_free_total=$(( avail_free_total+free ))
    if   [[ $usable -eq 0 ]]; then fcol="$DIM"; fstr="—"
    elif [[ $free -gt 0 ]];   then fcol="$G";   fstr="$free"
    else                           fcol="$R";   fstr="$free"; fi
    scol="$G"; [[ "$state" == *MIX* ]] && scol="$Y"; [[ $usable -eq 0 ]] && scol="$R"
    printf " %s %s %s %s %s %s\n" \
      "$(pad "$node" $W1)" "$(pad "$model" $W2)" \
      "$(pad "$total" $W3 r)" "$(pad "$alloc" $W4 r)" \
      "$(pad "$fstr" $W5 r "$fcol")" "$(pad "$state" $W6 l "$scol")"
  done < <(scontrol show node -o)

  hr $W1 $W2 $W3 $W4 $W5 $W6
  printf " %s %s %s %s %s\n" \
    "$(pad 合计 $W1 l "$B")" "$(pad "" $W2)" \
    "$(pad "$tot_total" $W3 r "$B")" "$(pad "$tot_alloc" $W4 r "$B")" "$(pad "$tot_free" $W5 r "$B")"
  echo
  printf " 全集群: %s%d%s 张卡, 已分配 %s%d%s, 空闲 %s%d%s (可立即排队 %s%d%s)\n" \
    "$B" "$tot_total" "$N" "$Y" "$tot_alloc" "$N" "$G" "$tot_free" "$N" "$C" "$avail_free_total" "$N"
  printf " 大显存(>32G, L40S): 共 %s%d%s 张, 现在可排队 %s%d%s 张\n" \
    "$B" "$big_total" "$N" "$C" "$big_free_avail" "$N"
  echo

  # ---------- 2) 账号占用 ----------
  local J1=15 J2=14 J3=4 J4=13 J5=4 J6=12
  printf "%s╔══ 账号 [%s] 的占用 ══╗%s\n" "$B" "$ME" "$N"
  printf " %s %s %s %s %s %s\n" \
    "$(pad JobID $J1 l "$DIM")" "$(pad 分区 $J2 l "$DIM")" "$(pad 状态 $J3 l "$DIM")" \
    "$(pad 节点/原因 $J4 l "$DIM")" "$(pad GPU $J5 r "$DIM")" "$(pad 已运行 $J6 l "$DIM")"
  hr $J1 $J2 $J3 $J4 $J5 $J6

  local run_gpu=0 pend_gpu=0 run_n=0 pend_n=0 rows=0
  local jid part st node b nodes tim gpn g scol h m
  declare -A RTYPE   # 运行中 按卡型统计张数
  while IFS='|' read -r jid part st node b nodes tim; do
    [[ -z "$jid" ]] && continue
    rows=$(( rows+1 ))
    gpn=$(grep -oP 'gpu:\K[0-9]+' <<<"$b"); gpn=${gpn:-0}
    g=$(( gpn * ${nodes:-1} ))
    if [[ "$st" == "R" ]]; then
      run_gpu=$(( run_gpu+g )); run_n=$(( run_n+1 )); scol="$G"
      # 按作业实际所在节点归类卡型(展开 node[2-3] 之类)
      if (( gpn > 0 )); then
        while read -r h; do
          [[ -z "$h" ]] && continue
          m=$(gpu_model "$h"); RTYPE["$m"]=$(( ${RTYPE["$m"]:-0} + gpn ))
        done < <(scontrol show hostnames "$node" 2>/dev/null)
      fi
    else
      pend_gpu=$(( pend_gpu+g )); pend_n=$(( pend_n+1 )); scol="$Y"
    fi
    printf " %s %s %s %s %s %s\n" \
      "$(pad "$jid" $J1)" "$(pad "$part" $J2)" "$(pad "$st" $J3 l "$scol")" \
      "$(pad "$node" $J4)" "$(pad "$g" $J5 r)" "$(pad "$tim" $J6)"
  done < <(squeue -h -u "$ME" -o "%i|%P|%t|%R|%b|%D|%M" 2>/dev/null)

  [[ $rows -eq 0 ]] && printf " %s(当前没有作业)%s\n" "$DIM" "$N"
  hr $J1 $J2 $J3 $J4 $J5 $J6

  # 卡型明细(固定顺序, 只显示 >0)
  local bd="" c
  for m in "${MODELS[@]}"; do
    c=${RTYPE["$m"]:-0}
    (( c > 0 )) && bd+="${bd:+, }${C}${c}${N}×${m}"
  done
  [[ -z $bd ]] && bd="${DIM}无${N}"
  printf " 运行中: %s%d%s 作业 / %s%d%s 张 GPU  →  %s\n" \
    "$B" "$run_n" "$N" "$G" "$run_gpu" "$N" "$bd"
  printf " 排队中: %s%d%s 作业 / %s%d%s 张 GPU\n" \
    "$B" "$pend_n" "$N" "$Y" "$pend_gpu" "$N"

  if command -v sshare >/dev/null 2>&1; then
    local fs; fs=$(sshare -h -U -u "$ME" -o "FairShare" 2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$fs" ]] && printf " Fairshare: %s%s%s\n" "$C" "$fs" "$N"
  fi
  echo
}

# =============================================================================
# 物理 nvidia-smi 探测(仅一次性模式)
# =============================================================================
render_phys() {
  printf "%s╔══ 物理 GPU 状态 (nvidia-smi 探测, 每节点起一个小作业) ══╗%s\n" "$B" "$N"
  printf "%s 提示: 需在每个可用节点排到 1 张卡才能读, drain/满载会跳过或超时%s\n" "$DIM" "$N"
  local node part out
  for node in node1 node2 node3 node4 node5 node6; do
    part=$(sinfo -h -n "$node" -o "%P" 2>/dev/null | head -1 | tr -d '*'); [[ -z "$part" ]] && part="athena"
    printf "%s ── %s (%s) ──%s\n" "$C" "$node" "$(gpu_model "$node")" "$N"
    out=$(timeout 45 srun --partition="$part" --nodelist="$node" --gres=gpu:1 \
            --cpus-per-task=1 --mem=1G -t 00:01:00 -J gpuprobe \
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null | sort -u)
    if [[ -z "$out" ]]; then printf "   %s(未能探测: 排队/超时/下线)%s\n" "$DIM" "$N"
    else printf "   %-4s %-12s %-12s %-6s\n" "卡#" "已用显存" "总显存" "利用率"
      awk -F', ' '{printf "   %-4s %-12s %-12s %-6s\n",$1,$2,$3,$4}' <<<"$out"; fi
  done
  echo
}

# =============================================================================
# 全局占用视图: 每节点谁在占 + 用户排行 (-a)
# =============================================================================
render_all() {
  declare -A NUG NTOT UTOT UM CAP NSTATE NMODEL
  local order=""   # 节点顺序
  # 1) 各节点容量/状态/型号(顺序取自 scontrol)
  local line node total state
  while IFS= read -r line; do
    node=$(grep -oP 'NodeName=\K[^ ]+' <<<"$line"); [[ -z "$node" ]] && continue
    total=$(grep -oP 'CfgTRES=[^ ]*gres/gpu=\K[0-9]+' <<<"$line"); total=${total:-0}
    [[ "$total" -eq 0 ]] && continue
    state=$(grep -oP 'State=\K[^ ]+' <<<"$line")
    CAP["$node"]=$total; NSTATE["$node"]=$state; NMODEL["$node"]=$(gpu_short "$node")
    order+="$node "
  done < <(scontrol show node -o)

  # 2) 运行中作业 -> 逐节点归类(展开 node[2-3])
  local user nl b nd gpn h m
  while IFS='|' read -r user nl b nd; do
    [[ -z "$user" ]] && continue
    gpn=$(grep -oP 'gpu:\K[0-9]+' <<<"$b"); gpn=${gpn:-0}
    (( gpn == 0 )) && continue
    while read -r h; do
      [[ -z "$h" ]] && continue
      m=$(gpu_short "$h")
      NUG["$h|$user"]=$(( ${NUG["$h|$user"]:-0} + gpn ))
      NTOT["$h"]=$(( ${NTOT["$h"]:-0} + gpn ))
      UTOT["$user"]=$(( ${UTOT["$user"]:-0} + gpn ))
      UM["$user|$m"]=$(( ${UM["$user|$m"]:-0} + gpn ))
    done < <(scontrol show hostnames "$nl" 2>/dev/null)
  done < <(squeue -h -t RUNNING -o "%u|%R|%b|%D" 2>/dev/null)

  # ---------- 每节点谁在占 ----------
  printf "%s╔══ 每个节点谁在占卡 (运行中) ══╗%s\n" "$B" "$N"
  local used cap st mdl pairs uinfo you k n u g tag
  for node in $order; do
    used=${NTOT["$node"]:-0}; cap=${CAP["$node"]}; st=${NSTATE["$node"]}; mdl=${NMODEL["$node"]}
    # 该节点用户按张数降序
    uinfo=""
    while read -r g u; do
      [[ -z "$u" ]] && continue
      tag=""; [[ "$u" == "$ME" ]] && tag="${B}(你)${N}"
      uinfo+="${uinfo:+, }${C}${u}${N}${tag} ×${g}"
    done < <(for k in "${!NUG[@]}"; do n=${k%%|*}; u=${k#*|}; [[ "$n" == "$node" ]] && echo "${NUG[$k]} $u"; done | sort -rn)
    # 满/空/下线着色
    local col="$G"
    [[ "$st" == *DRAIN* || "$st" == *DOWN* ]] && { col="$R"; uinfo="${DIM}下线 (drain)${N}"; }
    [[ $used -ge $cap && "$col" == "$G" ]] && col="$Y"
    [[ -z "$uinfo" ]] && uinfo="${DIM}空闲${N}"
    printf " %s %s %s%s%s  %s\n" \
      "$(pad "$node" 7)" "$(pad "$mdl" 6)" \
      "$col" "$(pad "$used/$cap" 5 r)" "$N" "$uinfo"
  done
  echo

  # ---------- 用户占用排行 ----------
  local T1=4 T2=12 T3=6 T4=8 T5=8
  printf "%s╔══ 用户占用排行 (运行中, 按 GPU 张数) ══╗%s\n" "$B" "$N"
  printf " %s %s %s %s %s\n" \
    "$(pad "#" $T1 r "$DIM")" "$(pad 用户 $T2 l "$DIM")" \
    "$(pad 总数 $T3 r "$DIM")" "$(pad A5000 $T4 r "$DIM")" "$(pad L40S $T5 r "$DIM")"
  hr $T1 $T2 $T3 $T4 $T5
  local rank=0 a5 l4 ucol
  while read -r g u; do
    [[ -z "$u" ]] && continue
    rank=$(( rank+1 )); a5=${UM["$u|A5000"]:-0}; l4=${UM["$u|L40S"]:-0}
    ucol="$C"; [[ "$u" == "$ME" ]] && { ucol="$B"; u="$u ←你"; }
    printf " %s %s %s %s %s\n" \
      "$(pad "$rank" $T1 r)" "$(pad "$u" $T2 l "$ucol")" \
      "$(pad "$g" $T3 r "$B")" "$(pad "$a5" $T4 r)" "$(pad "$l4" $T5 r)"
  done < <(for u in "${!UTOT[@]}"; do echo "${UTOT[$u]} $u"; done | sort -rn)
  [[ $rank -eq 0 ]] && printf " %s(当前没有运行中的 GPU 作业)%s\n" "$DIM" "$N"
  hr $T1 $T2 $T3 $T4 $T5
  echo
}

# =============================================================================
# 主流程
# =============================================================================
main_render() {
  if [[ $ALL -eq 1 ]]; then
    render_all
  else
    render
    [[ $PHYS -eq 1 ]] && render_phys
  fi
}

if [[ $WATCH -eq 1 ]]; then
  trap 'printf "\n%s已退出监视%s\n" "$DIM" "$N"; exit 0' INT
  while true; do
    printf '\033[H\033[J'   # 光标回原点 + 清屏(低闪烁)
    printf "%s⟳ 每 %ss 刷新  |  %s  |  按 Ctrl-C 退出%s\n\n" \
      "$DIM" "$INTERVAL" "$(date '+%Y-%m-%d %H:%M:%S')" "$N"
    main_render
    sleep "$INTERVAL" || break
  done
else
  main_render
fi
