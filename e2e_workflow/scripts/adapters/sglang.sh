# sglang serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads env exported by the dispatcher: MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS
#   EXTRA_ENV OVERLAY_PYTHONPATH PROFILE PROFILE_DIR PROFILE_NUM_STEPS BASE_URL RESULT_JSONL LOG
#   ISL OSL CONC SEED
# Sets SERVER_PID (global) in adapter_launch. Append canonical result lines to $RESULT_JSONL.

adapter_default_port() { echo 30000; }

# sglang is editable-installed in the ROCm images; its module finder can miss the `sglang.benchmark`
# subpackage (or CWD=/sgl-workspace shadows the package), which breaks `python -m sglang.bench_serving`
# (-> ModuleNotFoundError -> no load -> empty profile). Prepend the real source tree to PYTHONPATH to
# bypass it. Auto-detected; override/disable with SGLANG_SRC_PYTHONPATH=... (empty to disable). Harmless
# when the dir is absent (non-sglang images).
_SGL_PP="${SGLANG_SRC_PYTHONPATH-/sgl-workspace/sglang/python}"; [ -d "$_SGL_PP" ] || _SGL_PP=""

adapter_launch() {
  # Raise the scheduler watchdog by default: an authored/JIT kernel (FlyDSL/triton-author) overlaid on
  # the path JIT-compiles on first prefill, which can exceed sglang's default watchdog and kill the
  # server before CUDA-graph capture. Harmless for stock runs. Only add it if the caller didn't already
  # set one in EXTRA_SERVER_ARGS (override via WATCHDOG_TIMEOUT=...; set empty to disable).
  local _wd=""
  case " $EXTRA_SERVER_ARGS " in
    *" --watchdog-timeout "*) _wd="" ;;
    *) [ -n "${WATCHDOG_TIMEOUT:-600}" ] && _wd="--watchdog-timeout ${WATCHDOG_TIMEOUT:-600}" ;;
  esac
  # Pin GPU_ARCHS so aiter's JIT (chip_info.get_gfx_list) takes the env branch instead of
  # _detect_native() — the latter shells to rocm_agent_enumerator -> rocminfo PER cold-build worker
  # (~77 per import), which hang under GPU/KFD contention and pile up into a box-degrading storm
  # (observed: 561 procs, e2e throughput halved). Detect once here; honor a caller-set value.
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  # Launch through $SERVER_LAUNCH_PREFIX (adapter contract): it puts the server in its
  # own session so teardown can prove the process group is ours. Empty when unset.
  # shellcheck disable=SC2086
  ${SERVER_LAUNCH_PREFIX:-} env $EXTRA_ENV \
    ${_ga:+GPU_ARCHS=$_ga} \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR" \
    PYTHONPATH="${_SGL_PP:+$_SGL_PP:}${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    python -m sglang.launch_server \
      --model-path "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --tp-size "$TP" \
      --mem-fraction-static "$MEM_FRACTION" \
      $_wd \
      $EXTRA_SERVER_ARGS \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }

# ---- multi-node (read by adapters/launchers/multinode.sh) ----
# Only the launcher knows the values; only this file knows the spelling. Keeping
# the two apart is what lets one launcher serve every framework.
#
# per_rank: sglang wants one server process per node, each told which rank it is.
adapter_dist_mode() { echo per_rank; }
# adapter_dist_args RANK NNODES MASTER DIST_PORT
adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }

# ---- prefill/decode disaggregation (also read by launchers/multinode.sh) ----
# A DIFFERENT topology, not a wider version of the one above: each role is a
# COMPLETE server, TP-closed inside its own node, and the two never join a
# collective. They meet only where prefill hands finished KV pages to decode over
# a transfer backend, and a load balancer in front makes the pair look like one
# endpoint to the client. Hence no rank/world triple anywhere below -- passing one
# would make each engine wait for a peer that is not coming.
adapter_pd_mode() { echo pd_roles; }

# adapter_pd_args ROLE BOOTSTRAP_PORT TRANSFER_BACKEND [IB_DEVICE]
adapter_pd_args() {
  local role="$1" boot="$2" backend="${3:-mooncake_tcp}" ib="${4:-}"
  local out="--disaggregation-mode $role --disaggregation-transfer-backend $backend"
  # The bootstrap registry is a PREFILL-side service. On decode the identically
  # named flag would mean "the port the prefill side listens on", which decode
  # already receives per request from the load balancer -- so setting it there
  # states a second, unused source of truth that can only disagree.
  [ "$role" = "prefill" ] && out="$out --disaggregation-bootstrap-port $boot"
  # Left empty on purpose for the tcp transports: mooncake_tcp resolves the IB
  # device to None itself, and naming a device that has no verbs provider is how
  # you get an engine that starts and then cannot transfer.
  [ -n "$ib" ] && out="$out --disaggregation-ib-device $ib"
  echo "$out"
}

# adapter_pd_lb_launch HOST PORT PREFILL_URL BOOTSTRAP_PORT DECODE_URL
#
# sglang ships two load balancers behind one CLI: the Rust router (default) and
# the FastAPI mini-LB (--mini-lb, which its own banner calls debug-only and which
# forces the random policy). Both speak the same PD protocol, so the choice is
# LB_EXTRA_ARGS' business, not this function's.
#
# --model-path is passed so the LB can answer /get_model_info and /v1/models
# without having to interrogate a worker that may still be loading weights.
adapter_pd_lb_launch() {
  local host="$1" port="$2" purl="$3" boot="$4" durl="$5"
  local bootarg=()
  [ -n "$boot" ] && [ "$boot" != "none" ] && bootarg=("$boot")
  # shellcheck disable=SC2086
  ${SERVER_LAUNCH_PREFIX:-} env $EXTRA_ENV \
    PYTHONPATH="${_SGL_PP:+$_SGL_PP:}${PYTHONPATH:-}" \
    python -m sglang_router.launch_router \
      --pd-disaggregation \
      --prefill "$purl" "${bootarg[@]}" \
      --decode "$durl" \
      --host "$host" --port "$port" \
      --model-path "$MODEL" \
      --policy "${PD_LB_POLICY:-round_robin}" \
      --worker-startup-timeout-secs "${PD_LB_STARTUP_TIMEOUT_S:-2400}" \
      --request-timeout-secs "${PD_LB_REQUEST_TIMEOUT_S:-3600}" \
      ${LB_EXTRA_ARGS:-} \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  local extra=()
  if [ "$PROF" = "1" ]; then
    extra=(--profile --profile-num-steps "$PROFILE_NUM_STEPS"
           --profile-output-dir "$PROFILE_DIR" --profile-prefix e2e)
  fi
  # Optional request-rate (req/s) to STAGGER arrivals so sequences sit at different prefill/decode
  # phases — used by the steady-state profiling path. Empty => inf (max_concurrency still caps).
  [ -n "${REQUEST_RATE:-}" ] && extra+=(--request-rate "$REQUEST_RATE")
  PYTHONPATH="${_SGL_PP:+$_SGL_PP:}${PYTHONPATH:-}" \
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --seed "$SEED" \
    --output-file "$RESULT_JSONL" "${extra[@]}"
  # sglang.bench_serving appends a result json line (output_throughput, median_ttft_ms, median_tpot_ms)
  # to --output-file, which is exactly the dispatcher's canonical schema. Nothing else to do.
}

# adapter_profile_window — capture a profiler window on the ALREADY-RUNNING, warm, mid-load server via
# sglang's HTTP profiler, so the trace reflects the real continuous-batching steady-state mix (prefill
# chunks + decode interleaved) instead of a cold prefill ramp. record_shapes=true so the parser gets
# Input Dims for shape attribution. Called by bench_e2e.sh AFTER a sustained background load is warm.
#
# with_stack=FALSE + activities=["CPU","GPU"]: sglang's profiler otherwise records Python call stacks
# (with_stack) which produce hundreds of MB of python_function events -> a multi-hundred-MB trace whose
# ROCm flush takes >10min AND BLOCKS the scheduler (health drops, requests time out). Dropping stacks
# (and MEM/RPD activities) shrinks the trace ~an order of magnitude while KEEPING everything parse_profile
# needs: the GPU kernel timeline, cpu_op Input Dims (record_shapes), and execute_* phase annotations.
# Override via SGLANG_PROFILE_ACTIVITIES / SGLANG_PROFILE_WITH_STACK.
adapter_profile_window() {
  local before after
  before=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  local _acts="${SGLANG_PROFILE_ACTIVITIES:-[\"CPU\",\"GPU\"]}" _stack="${SGLANG_PROFILE_WITH_STACK:-false}"
  # num_steps set => the server records that many forward steps then auto-saves (async; returns at once).
  if ! curl -sf -X POST "${BASE_URL}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\":\"${PROFILE_DIR}\",\"num_steps\":${PROFILE_NUM_STEPS},\"record_shapes\":true,\"with_stack\":${_stack},\"activities\":${_acts}}" \
        >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (sglang HTTP profiler unavailable?)" >&2
    return 1
  fi
  # wait for a NEW trace to land (server saves after num_steps forward passes)
  local deadline=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && { sleep 2; return 0; }   # +2s for the write to flush
    sleep 3
  done
  # num_steps may not be honored on some builds — force a stop and re-check
  curl -sf -X POST "${BASE_URL}/stop_profile" >/dev/null 2>&1 || true
  sleep 3
  after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ]
}
