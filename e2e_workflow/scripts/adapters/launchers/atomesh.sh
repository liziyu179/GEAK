#!/usr/bin/env bash
# Atomesh topology LAUNCHER adapter for bench_e2e.sh.  Sourced (not executed).
# Use with BACKEND=atom BENCH_LAUNCHER=atomesh.
#
# Launches a whole single-node Atomesh deployment as ONE measured "server":
#   regular : N ATOM workers behind `atomesh launch --worker-urls ...`
#   pd      : P prefill + D decode ATOM workers (Mooncake KV transfer over TCP)
#             behind `atomesh launch --pd-disaggregation`
# The router listens on $PORT, so $BASE_URL, the native ATOM bench client, the
# summary and the acceptance contract are unchanged. Only adapter_launch and
# adapter_profile_window are redefined; adapter_health / adapter_bench /
# adapter_prepare_measurement are inherited from adapters/atom.sh.
#
# TOPOLOGY comes from GEAK's own knobs plus ATOMESH_* variables:
#   GPU, TP                 the comma-separated GPU set is cut into TP-sized
#                           chunks; each chunk is one worker. GPU=0,1 TP=1 => 2
#                           workers; GPU=0,1,2,3 TP=2 => 2 workers of TP2.
#   ATOMESH_MODE            regular (default) | pd
#   ATOMESH_POLICY          regular router policy (default round_robin)
#   ATOMESH_PREFILL_COUNT   pd: number of leading chunks used as prefill (default 1)
#   ATOMESH_PREFILL_POLICY / ATOMESH_DECODE_POLICY   pd policies (default round_robin)
#   ATOMESH_PROFILE_WORKER  worker whose torch profiler serves PROFILE=1
#                           (default: worker-0 in regular, decode-0 in pd)
#   ATOMESH_ROUTER_ARGS     extra raw `atomesh launch` flags
#   ATOMESH_WORKER_PORT_BASE  first worker HTTP port (default PORT+2000, outside
#                           bench_e2e's PORT window like atom.sh's PORT+1000 engine
#                           port); worker i uses BASE+10*i (HTTP), +1 (engine),
#                           +2 (Mooncake handshake)
#
# These ATOMESH_* assignments may also arrive inside EXTRA_ENV. That is what
# lets the Config Tuner sweep the router policy through the EXTRA_ENV channel it
# already uses for backend env toggles; those tokens are consumed here and not
# forwarded to the workers.
#
# OPTIMIZATIONS reach EVERY worker: EXTRA_SERVER_ARGS on each worker's command
# line (split by atom.sh's _atom_server_argv, so it must be sourced first, as
# bench_e2e.sh does), EXTRA_ENV in each worker's environment, OVERLAY_PYTHONPATH prepended to
# each worker's PYTHONPATH. The router gets none of them.
#
# LIFECYCLE: one setsid'd supervisor owns every component, so the dispatcher's
# single SERVER_PID / process-group teardown reaps the whole mesh. It starts
# worker 0 alone (cold workers of one deployment race on the shared
# ~/.cache/atom torch.compile cache), then the rest, waits for each /health (and,
# in pd, checks the /kv_transfer_info roles), then starts the router. Afterwards
# it watches every component and kills the whole group on the first death or
# engine-core failure: an ATOM api_server keeps answering /health after its
# EngineCore died, so without that watch a crashed worker would hang requests
# until the router's request timeout instead of failing the leg.
#
# LOGS: all component output is merged into $LOG with a "[name] " prefix, so the
# dispatcher's OOM / fatal / stall detection and ATOM's readiness probe keep
# working; per-component copies go to $OUT_DIR/atomesh/<name>.log.

_ATOMESH_SUPERVISOR='
set -u
log="$ATOMESH_LOG"; dir="$ATOMESH_DIR"
say() { echo "[atomesh-supervisor] $*" >> "$log"; }
pids=(); names=(); offs=()
stop_all() {
  # Ignore (not reset): the group-wide TERM below also reaches this shell, and it
  # must survive to send the KILL that reaps workers ignoring TERM.
  trap "" TERM INT
  say "stopping all components"
  # kill 0 signals our process group, which is only ours to kill when we lead it
  # (adapter_launch starts us under setsid). Otherwise signal what we started.
  if [ "$(ps -o pgid= -p $$ | tr -d " ")" = "$$" ]; then
    kill -TERM 0 2>/dev/null; sleep "$ATOMESH_STOP_GRACE_S"; kill -KILL 0 2>/dev/null
  else
    kill -TERM "${pids[@]}" 2>/dev/null; sleep "$ATOMESH_STOP_GRACE_S"; kill -KILL "${pids[@]}" 2>/dev/null
  fi
  exit "${1:-0}"
}
trap "stop_all 0" TERM INT

start_component() {   # start_component NAME CMD...
  local name="$1"; shift
  : > "$dir/$name.log"
  # Process substitution keeps $! on the component itself (not on a pipeline
  # subshell that TP>1 rank children would keep alive). sed -u, not awk: mawk,
  # the default awk in Debian/Ubuntu images, block-buffers pipe input.
  ( exec "$@" ) > >(tee -a "$dir/$name.log" | sed -u "s/^/[$name] /" >> "$log") 2>&1 &
  pids+=("$!"); names+=("$name"); offs+=(0)
}
alive() { kill -0 "$1" 2>/dev/null; }
engine_failed() {   # engine_failed IDX: scan only what the component logged since the last call
  local f="$dir/${names[$1]}.log" size
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$size" -gt "${offs[$1]}" ] || return 1
  tail -c +"$((offs[$1] + 1))" "$f" | head -c "$((size - offs[$1]))" \
    | grep -Eq "run_engine failed|Engine Core: load model runner failed" && return 0
  offs[$1]=$size
  return 1
}

wait_http() {   # wait_http IDX URL
  while ! curl -sf --max-time 3 "$2" >/dev/null 2>&1; do
    alive "${pids[$1]}" || { say "FATAL: ${names[$1]} exited before becoming healthy"; stop_all 1; }
    engine_failed "$1" && { say "FATAL: ${names[$1]} engine core failed during startup"; stop_all 1; }
    sleep 3
  done
  say "${names[$1]} healthy ($2)"
}

# Workers: one line per worker in $ATOMESH_PLAN: "name<TAB>port<TAB>command".
# The command is a printf %q line, so it is parsed again by eval.
wnames=(); wports=(); wcmds=()
while IFS=$'"'"'\t'"'"' read -r name port cmd; do
  [ -n "$name" ] && { wnames+=("$name"); wports+=("$port"); wcmds+=("$cmd"); }
done < "$ATOMESH_PLAN"
nworkers=${#wnames[@]}

eval "start_component \"\${wnames[0]}\" ${wcmds[0]}"
wait_http 0 "http://127.0.0.1:${wports[0]}/health"
for ((i = 1; i < nworkers; i++)); do eval "start_component \"\${wnames[$i]}\" ${wcmds[$i]}"; done
for ((i = 1; i < nworkers; i++)); do wait_http "$i" "http://127.0.0.1:${wports[$i]}/health"; done
if [ "$ATOMESH_MODE" = "pd" ]; then
  for ((i = 0; i < nworkers; i++)); do
    info="$(curl -sf --max-time 5 "http://127.0.0.1:${wports[$i]}/kv_transfer_info" 2>/dev/null)"
    case "${wnames[$i]}:$info" in
      prefill-*:*kv_producer*|decode-*:*kv_consumer*) say "${wnames[$i]} kv_transfer_info: $info" ;;
      *) say "FATAL: ${wnames[$i]} kv_transfer_info has the wrong role: ${info:-<none>}"; stop_all 1 ;;
    esac
  done
fi

eval "start_component router $ATOMESH_ROUTER_CMD"
while ! curl -sf --max-time 3 "$ATOMESH_ROUTER_URL/health" >/dev/null 2>&1; do
  alive "${pids[$nworkers]}" || { say "FATAL: router exited before becoming healthy"; stop_all 1; }
  sleep 1
done
say "topology ready: mode=$ATOMESH_MODE workers=$nworkers router=$ATOMESH_ROUTER_URL"

while :; do
  for ((i = 0; i < ${#pids[@]}; i++)); do
    if ! alive "${pids[$i]}"; then say "FATAL: ${names[$i]} exited"; stop_all 1; fi
    if [ "$i" -lt "$nworkers" ] && engine_failed "$i"; then
      say "FATAL: ${names[$i]} engine core failed (api server may still answer /health)"; stop_all 1
    fi
  done
  sleep 2
done
'

# Read ATOMESH_* overrides out of EXTRA_ENV (later tokens win) and return the
# remaining tokens for the workers in _atomesh_worker_env.
_atomesh_split_extra_env() {
  _atomesh_worker_env=()
  local line tok
  local -a toks
  [ -n "${EXTRA_ENV:-}" ] || return 0
  while IFS= read -r line; do
    read -ra toks <<< "$line"
    for tok in "${toks[@]}"; do
      case "$tok" in
        ATOMESH_*=*)
          if [[ "$tok" =~ ^ATOMESH_[A-Z0-9_]+= ]]; then
            printf -v "${tok%%=*}" '%s' "${tok#*=}"
            echo ">>> atomesh launcher: ${tok%%=*} taken from EXTRA_ENV"
          fi ;;
        ROCR_VISIBLE_DEVICES=*|HIP_VISIBLE_DEVICES=*|CUDA_VISIBLE_DEVICES=*)
          echo ">>> atomesh launcher: dropping GPU mask from EXTRA_ENV: $tok (topology owns pinning)" >&2 ;;
        *) _atomesh_worker_env+=("$tok") ;;
      esac
    done
  done <<< "$EXTRA_ENV"
}

# Quote one argv for the plan file (eval'd by the supervisor).
_atomesh_q() { printf '%q ' "$@"; }

adapter_launch() {
  if [ "${BACKEND:-}" != "atom" ]; then
    echo "!!! atomesh launcher: requires BACKEND=atom (got '${BACKEND:-}')" >&2; return 2
  fi
  if [ -z "${SERVER_LAUNCH_PREFIX:-}" ]; then
    echo "!!! atomesh launcher: needs setsid (SERVER_LAUNCH_PREFIX) so teardown owns the whole mesh" >&2; return 2
  fi
  _atomesh_split_extra_env
  local mode="${ATOMESH_MODE:-regular}"
  case "$mode" in regular|pd) ;; *) echo "!!! atomesh launcher: ATOMESH_MODE must be regular|pd (got '$mode')" >&2; return 2 ;; esac

  # Cut the GPU set into TP-sized worker chunks.
  local -a gpus chunks=()
  IFS=',' read -ra gpus <<< "${GPU:-0}"
  local tp="${TP:-1}" i
  if [ $(( ${#gpus[@]} % tp )) -ne 0 ] || [ "${#gpus[@]}" -lt "$tp" ]; then
    echo "!!! atomesh launcher: GPU='$GPU' (${#gpus[@]} GPUs) is not a multiple of TP=$tp" >&2; return 2
  fi
  for ((i = 0; i < ${#gpus[@]}; i += tp)); do
    chunks+=("$(IFS=,; echo "${gpus[*]:i:tp}")")
  done
  local n="${#chunks[@]}" nprefill=0
  if [ "$mode" = "pd" ]; then
    nprefill="${ATOMESH_PREFILL_COUNT:-1}"
    if [ "$nprefill" -lt 1 ] || [ "$nprefill" -ge "$n" ]; then
      echo "!!! atomesh launcher: pd needs 1 <= ATOMESH_PREFILL_COUNT < workers ($n)" >&2; return 2
    fi
  fi

  # Per-worker engine flags: the same defaults and gating as adapters/atom.sh's
  # adapter_launch, so a mesh worker is launched exactly like a native server.
  local ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  local mml="${MAX_MODEL_LEN:-}"
  if [ -z "$mml" ] && ! { [ "${ISL:-}" = "1024" ] && [ "${OSL:-}" = "1024" ]; }; then mml=10240; fi
  local -a common=()
  [ -n "$mml" ] && common+=(--max-model-len "$mml")
  [ -n "${KV_CACHE_DTYPE:-fp8}" ] && common+=(--kv_cache_dtype "${KV_CACHE_DTYPE:-fp8}")
  [ -n "${BLOCK_SIZE:-16}" ] && common+=(--block-size "${BLOCK_SIZE:-16}")
  [ -n "${MEM_FRACTION:-}" ] && common+=(--gpu-memory-utilization "$MEM_FRACTION")
  [ "${EP_SIZE:-1}" -gt 1 ] 2>/dev/null && common+=(--enable-expert-parallel)
  [ "${DP_ATTENTION:-false}" = "true" ] && common+=(--enable-dp-attention)
  [ "${BENCH_TRUST_REMOTE_CODE:-0}" = "1" ] && common+=(--trust-remote-code)
  # Split EXTRA_SERVER_ARGS the way adapters/atom.sh does (shell quoting, no globbing, fail
  # closed), so a quoted JSON value reaches every worker exactly as it reaches a native server.
  _atom_server_argv "${EXTRA_SERVER_ARGS:-}" || return 2
  local -a extra_args=(${_ATOM_EXTRA_ARGV[@]+"${_ATOM_EXTRA_ARGV[@]}"})

  local dir="$OUT_DIR/atomesh" plan base="${ATOMESH_WORKER_PORT_BASE:-$((PORT + 2000))}"
  mkdir -p "$dir"; plan="$dir/plan.tsv"; : > "$plan"
  local profile_worker="${ATOMESH_PROFILE_WORKER:-$([ "$mode" = pd ] && echo decode-0 || echo worker-0)}"
  ATOMESH_PROFILE_URL=""; ATOMESH_PROFILE_NAME=""
  local -a router_targets=() topo=()
  local name role port cmd pidx=0 didx=0
  for ((i = 0; i < n; i++)); do
    port=$((base + 10 * i))
    if [ "$mode" = "regular" ]; then
      name="worker-$i"; role=""
      router_targets+=("http://127.0.0.1:$port")
    elif [ "$i" -lt "$nprefill" ]; then
      name="prefill-$pidx"; role=kv_producer; pidx=$((pidx + 1))
      router_targets+=(--prefill "http://127.0.0.1:$port")
    else
      name="decode-$didx"; role=kv_consumer; didx=$((didx + 1))
      router_targets+=(--decode "http://127.0.0.1:$port")
    fi
    local -a wargs=(--model "$MODEL" --host 127.0.0.1 --server-port "$port" --port "$((port + 1))"
      -tp "$tp" "${common[@]}")
    local -a wenv=(PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
      HIP_VISIBLE_DEVICES="${chunks[$i]}" CUDA_VISIBLE_DEVICES="${chunks[$i]}")
    [ -n "$ga" ] && wenv+=(GPU_ARCHS="$ga")
    if [ -n "$role" ]; then
      # Single node, no RDMA HCA: Mooncake over TCP on loopback.
      wargs+=(--kv-transfer-config "{\"kv_role\":\"$role\",\"kv_connector\":\"mooncake\",\"protocol\":\"tcp\",\"http_port\":$port,\"handshake_port\":$((port + 2))}")
      case " ${EXTRA_SERVER_ARGS:-} " in *" --enable_prefix_caching "*|*" --no-enable_prefix_caching "*) ;; *) wargs+=(--no-enable_prefix_caching) ;; esac
      wenv+=(ATOM_HOST_IP=127.0.0.1 MC_FORCE_TCP=true)
    fi
    if [ "$name" = "$profile_worker" ] && [ -n "${PROFILE_DIR:-}" ]; then
      mkdir -p "$PROFILE_DIR"
      wargs+=(--torch-profiler-dir "$PROFILE_DIR")
      wenv+=(ATOM_PROFILER_MORE="${ATOM_PROFILER_MORE:-1}")
      ATOMESH_PROFILE_URL="http://127.0.0.1:$port"
      ATOMESH_PROFILE_NAME="$name"
    fi
    cmd="$(_atomesh_q env ${_atomesh_worker_env[@]+"${_atomesh_worker_env[@]}"} "${wenv[@]}" \
      PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
      python3 -m atom.entrypoints.openai_server "${wargs[@]}" ${extra_args[@]+"${extra_args[@]}"})"
    printf '%s\t%s\t%s\n' "$name" "$port" "$cmd" >> "$plan"
    topo+=("$name=gpu[${chunks[$i]}]:$port")
  done

  # atomesh's own Prometheus default (29000) would collide between legs.
  local -a rargs=(--host "$HOST" --port "$PORT" --backend atom --log-dir "$dir"
    --prometheus-port "$((base - 1))")
  if [ "$mode" = "regular" ]; then
    rargs+=(--worker-urls "${router_targets[@]}" --policy "${ATOMESH_POLICY:-round_robin}")
  else
    rargs+=(--pd-disaggregation "${router_targets[@]}"
      --prefill-policy "${ATOMESH_PREFILL_POLICY:-round_robin}"
      --decode-policy "${ATOMESH_DECODE_POLICY:-round_robin}"
      --disable-health-check --disable-circuit-breaker)
  fi
  local -a rextra=()
  # shellcheck disable=SC2206
  [ -n "${ATOMESH_ROUTER_ARGS:-}" ] && rextra=(${ATOMESH_ROUTER_ARGS})

  echo ">>> atomesh launcher: mode=$mode tp=$tp ${topo[*]} router=:$PORT" \
       "policy=$([ "$mode" = pd ] && echo "${ATOMESH_PREFILL_POLICY:-round_robin}/${ATOMESH_DECODE_POLICY:-round_robin}" || echo "${ATOMESH_POLICY:-round_robin}")"
  {
    echo "mode=$mode"; echo "tp=$tp"; echo "workers=${topo[*]}"
    echo "router_args=${rargs[*]} ${rextra[*]}"; echo "profile_worker=$profile_worker"
  } > "$dir/topology.txt"
  : > "$LOG"

  # shellcheck disable=SC2086
  ATOMESH_LOG="$LOG" ATOMESH_DIR="$dir" ATOMESH_PLAN="$plan" ATOMESH_MODE="$mode" \
  ATOMESH_STOP_GRACE_S="${ATOMESH_STOP_GRACE_S:-5}" ATOMESH_ROUTER_URL="http://$HOST:$PORT" \
  ATOMESH_ROUTER_CMD="$(_atomesh_q atomesh launch "${rargs[@]}" ${rextra[@]+"${rextra[@]}"})" \
    $SERVER_LAUNCH_PREFIX bash -c "$_ATOMESH_SUPERVISOR" atomesh-supervisor >> "$LOG" 2>&1 &
  SERVER_PID=$!
}

# Profile one worker of the mesh: the router does not proxy /start_profile, so
# point the native ATOM window at the chosen worker's own HTTP port.
copy_function adapter_profile_window adapter_profile_window_atom
adapter_profile_window() {
  if [ -z "${ATOMESH_PROFILE_URL:-}" ]; then
    echo "!!! atomesh launcher: no worker was launched with a profiler (ATOMESH_PROFILE_WORKER)" >&2
    return 1
  fi
  echo ">>> atomesh launcher: profiling $ATOMESH_PROFILE_NAME at $ATOMESH_PROFILE_URL"
  BASE_URL="$ATOMESH_PROFILE_URL" adapter_profile_window_atom
}
