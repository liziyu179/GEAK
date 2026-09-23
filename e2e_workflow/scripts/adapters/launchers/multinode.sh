#!/usr/bin/env bash
# Multi-node server LAUNCHER adapter for bench_e2e.sh.  Sourced, never executed.
#
# One server instance whose TP group spans N pinned machines. Only WHO starts the
# server changes: the backend, the client, the metric and the acceptance contract
# are the single-host ones, which is what makes a multi-node number comparable to
# the rest of GEAK's results rather than a separate dialect.
#
# Framework-agnostic: nothing under multi_node/ knows how to start a server. Each
# rank sources adapters/$BACKEND.sh on its own node and asks it for the launch
# line and for adapter_dist_args -- how that framework spells "I am rank i of n".
# Supporting a new backend is therefore a function in its adapter, not a fork of
# this launcher.
#
# Requires a pin from scripts/multi_node/mn.py preflight (mn_state.json). Refusing
# to select machines here is deliberate: a launcher that picked its own pair would
# let the baseline leg and the candidate leg land on different hardware, and the
# A/B delta would then be a measure of which boxes were free.
#
# Division of labour:
#   mn.py preflight   picks + PINS the machines, proves the shared mount (once, in Setup)
#   mn.py launch      renders serve.sh.tmpl per rank and ships it via Ray (per launch)
#   mn.py supervise  the local pid bench_e2e.sh watches; reaps the ranks on SIGTERM
#   serve.sh.tmpl     per-rank NODE facts only; sources the backend adapter for
#                     the launch line (recipe base -> GEAK's swept flags -> the
#                     rank/world triple last, which no sweep may displace)
#
# Three things are set at SOURCE time, not inside adapter_launch, because
# bench_e2e.sh reads them between sourcing this file and calling the launch:
# HOST/PORT (they feed BASE_URL), and the serving mutex path.

# A ONE-CARD launch is not this launcher's job, and the check belongs HERE rather
# than in every caller. BENCH_LAUNCHER is inherited from the run env, so a
# multi-node run hands this launcher to its isolated single-GPU legs as well
# (op_bench bake-offs, shape capture, the recursive kernel layer -- one pinned
# card each). Such a leg can never satisfy what this launcher requires, a machine
# pin and a TP group filling every pinned node, so without this it aborts before
# the leg's server ever starts -- and the failure reads as a multi-node infra
# fault rather than a leg that was never distributed to begin with.
#
# Returning leaves the backend's own adapter_launch in place: bench_e2e.sh CLONED
# it to adapter_launch_native before sourcing this file and never replaced it,
# and for one card it is already the right launcher.
#
# TP unset is the same case -- bench_e2e.sh defaults it to 1 further down. A leg
# that really is distributed always states its TP, because one card cannot span
# machines.
if [ "${TP:-1}" = "1" ]; then
  echo ">>> multinode launcher: TP=1 is a single-card launch; leaving it to the" \
       "backend's native launcher (this launcher serves a pinned node set)." >&2
  return 0
fi

# bench_e2e.sh is COPIED into $EVAL_DIR, and the copy does not bring this
# directory with it, so resolve the same way the script resolves its own
# siblings: staged copy first, then the skill/workflow source tree.
_MN_DIR="${GEAK_MN_DIR:-}"
if [ -z "$_MN_DIR" ]; then
  for _c in "$HERE/multi_node" "${SKILL_DIR:-}/scripts/multi_node" \
            "${WORKFLOW_DIR:-}/scripts/multi_node"; do
    case "$_c" in /scripts/multi_node) continue ;; esac
    [ -d "$_c" ] && { _MN_DIR="$_c"; break; }
  done
fi
_MN_STATE="${GEAK_MN_STATE:-${GEAK_MN_EXT_SHARED_FS:-}/mn_state.json}"

# Last resort: ask the PIN where the launcher lives. Every path above is an
# environment variable, and the agent layer sets SKILL_DIR for the baseline leg
# but not for the sweep legs its config tuner spawns -- so the run records a
# baseline, then loses every A/B leg to "cannot find scripts/multi_node" in a
# per-leg log while the workflow above it carries on none the wiser. The pin
# does not have that problem: `mn.py preflight` stamps its own directory into
# the state file, so a pinned run can always find the code that pinned it --
# and that is the RIGHT copy by construction, since it is the one whose
# preflight this pin came from.
if [ ! -d "$_MN_DIR" ] && [ -f "$_MN_STATE" ]; then
  _MN_DIR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("mn_dir") or "")' \
              "$_MN_STATE" 2>/dev/null || echo "")"
fi
if [ ! -d "$_MN_DIR" ]; then
  echo "!!! multinode launcher: cannot find scripts/multi_node (looked next to" \
       "$HERE, under SKILL_DIR and WORKFLOW_DIR, and in the pin's mn_dir)." \
       "Set GEAK_MN_DIR." >&2
  exit 3
fi

if [ ! -f "$_MN_STATE" ]; then
  echo "!!! multinode launcher: no pinned node set at $_MN_STATE." >&2
  echo "    Run the preflight first:  python3 $_MN_DIR/mn.py preflight --nnodes 2" >&2
  echo "    It pins the machines and proves the shared mount. Without the pin each" >&2
  echo "    leg would choose its own nodes and the A/B would compare hardware." >&2
  exit 3
fi

# HOST: the master's cluster IP, so the client dials the rank that owns the HTTP
# endpoint. The default 127.0.0.1 is only correct when the bench host happens to
# be rank0 -- true in the current deployment, and silently wrong the first time
# the preflight pins a different pair. PORT comes from the same file so that
# every rank, the client and the health probe agree on one number.
_MN_HOST="$(python3 -c 'import json,sys;s=json.load(open(sys.argv[1]));print(s["master_ip"])' "$_MN_STATE")"
_MN_PORT="$(python3 -c 'import json,sys;s=json.load(open(sys.argv[1]));print(s["serve_port"])' "$_MN_STATE")"
if [ -z "$_MN_HOST" ] || [ -z "$_MN_PORT" ]; then
  echo "!!! multinode launcher: $_MN_STATE has no master_ip/serve_port; re-run mn.py preflight." >&2
  exit 3
fi
HOST="$_MN_HOST"
PORT="$_MN_PORT"
export HOST PORT

# Which topology this pin serves. Read from the PIN, not from this shell's
# environment: BENCH_LAUNCHER is inherited by every leg of a run but the rest of
# the env is not, so a leg that guessed would serve the other shape and report a
# throughput for a configuration nobody asked for.
_MN_TOPOLOGY="$(python3 -c 'import json,os,sys;s=json.load(open(sys.argv[1]));print(os.environ.get("GEAK_MN_TOPOLOGY") or s.get("topology") or "tp_span")' "$_MN_STATE")"
# Engine endpoints of a disaggregated pair, "role=url" per line. Empty otherwise.
# The client-facing HOST:PORT above is the LOAD BALANCER in that topology, which
# is exactly why these are needed separately: the LB proxies inference, not the
# profiler or per-engine health.
_MN_PD_ENGINES=""
if [ "$_MN_TOPOLOGY" = "pd_disagg" ]; then
  _MN_PD_ENGINES="$(python3 - "$_MN_STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
pd = s.get("pd") or {}
roles = pd.get("roles") or ["prefill", "decode"]
ports = pd.get("role_ports") or [s["serve_port"] + 1 + i for i in range(len(roles))]
for node in s["nodes"]:
    r = node["rank"]
    print(f"{roles[r]}=http://{node['ip']}:{ports[r]}")
PY
)"
  echo ">>> multinode launcher: pd_disagg topology; client endpoint is the load" \
       "balancer at $HOST:$PORT, engines:" $(echo "$_MN_PD_ENGINES" | tr '\n' ' ')
fi
# The pinned port must survive bench_e2e.sh's window policy: a clamp would
# re-allocate a port that no rank ever binds, and the health loop would then
# time out against an endpoint that does not exist. Prefer a port inside the
# window (`mn.py preflight` defaults to 30890); only disable the policy if the pin
# says otherwise.
if [ "$PORT" -lt "${PORT_BASE:-30000}" ] || [ "$PORT" -ge "$(( ${PORT_BASE:-30000} + ${PORT_SPAN:-1000} ))" ] 2>/dev/null; then
  echo ">>> multinode launcher: pinned port $PORT is outside ${PORT_BASE:-30000}..$(( ${PORT_BASE:-30000} + ${PORT_SPAN:-1000} - 1 ));" \
       "disabling the window policy so the pinned port is not clamped away." >&2
  PORT_ENFORCE_RANGE=0
  export PORT_ENFORCE_RANGE
fi

# The serving mutex must be CLUSTER-wide. Its default lives in /tmp, which is
# per-container: two multi-node runs on different hosts would each take their own
# local lock, both launch, and collide on the dist-init port with 32 ranks
# fighting over 16 cards. On the shared mount the lock is one object for the
# whole cluster.
if [ -z "${SERVING_GPU_LOCK:-}" ] && [ -n "${GEAK_MN_EXT_SHARED_FS:-}" ]; then
  SERVING_GPU_LOCK="${GEAK_MN_EXT_SHARED_FS}/geak_serving_mn.lock"
  export SERVING_GPU_LOCK
  echo ">>> multinode launcher: cluster-wide serving lock at $SERVING_GPU_LOCK"
fi

# TERM->KILL grace. The default 10s is sized for a local process; here the TERM
# handler has to reach every rank over Ray and confirm the processes are gone, so
# a 10s KILL would orphan exactly what the handler exists to reap.
SERVER_STOP_GRACE_S="${SERVER_STOP_GRACE_S:-90}"
export SERVER_STOP_GRACE_S

# adapter_profile_window — the backend's version waits for ONE new trace file
# and returns. With sixteen ranks exporting into one shared directory that is
# wrong twice over: it returns while fifteen ranks are still flushing (and the
# teardown then truncates them), and the trace the parser later picks with
# `ls -t | head -1` is decided by whichever rank's NFS write landed first. So
# wait for the whole set, then gate it.
#
# Delegating to adapter_profile_window_native for the capture itself keeps the
# window semantics (num_steps, record_shapes, with_stack=false) in ONE place --
# the backend adapter -- so a change there is not silently missed here.
adapter_profile_window() {
  local _expect="${GEAK_MN_EXPECT_TRACES:-$TP}"
  local _before _after _deadline _depth=1 _windowed=""
  # pd_disagg exports one subdirectory per engine (below), so every count here
  # has to look one level down.
  [ "$_MN_TOPOLOGY" = "pd_disagg" ] && _depth=2
  _before=$(find "$PROFILE_DIR" -maxdepth "$_depth" -name '*.trace.json*' \
            2>/dev/null | wc -l)

  # pd_disagg: the request cannot be delegated, and not only because the load
  # balancer proxies no profiler route. The native window POSTs to ONE endpoint
  # and then waits for a trace to land, so calling it per engine would open the
  # decode window only after prefill's had already closed -- two captures of
  # different moments, presented as one steady state. Both engines are therefore
  # asked first and waited for afterwards.
  #
  # The payload restates the backend's window because there is no single-endpoint
  # function to borrow it from; the knobs are read from the SAME env names the
  # backend adapter uses, so the two cannot drift into capturing different things.
  if [ "$_MN_TOPOLOGY" = "pd_disagg" ]; then
    local _acts="${SGLANG_PROFILE_ACTIVITIES:-[\"CPU\",\"GPU\"]}"
    local _stack="${SGLANG_PROFILE_WITH_STACK:-false}" _role _url _posted=0
    # Each engine exports into its OWN subdirectory, and that is not tidiness.
    # The two engines are separate TP worlds: both number their ranks TP-0..TP-7,
    # so one shared directory holds sixteen files under eight distinct rank
    # numbers, and every consumer downstream -- the gate's per-rank map, the
    # parser's `ls -t | head -1` -- silently keeps one engine's and discards the
    # other's. WHICH one it keeps is decided by NFS flush order, so the baseline
    # leg and the candidate leg can profile different engines. A directory per
    # role makes the attribution structural instead of inferred, and it is what
    # lets prefill and decode be analysed as the two different machines they are.
    while IFS='=' read -r _role _url; do
      [ -n "$_url" ] || continue
      mkdir -p "$PROFILE_DIR/$_role"
      if curl -sf -X POST "${_url}/start_profile" -H 'Content-Type: application/json' \
          -d "{\"output_dir\":\"${PROFILE_DIR}/${_role}\",\"num_steps\":${PROFILE_NUM_STEPS},\"record_shapes\":true,\"with_stack\":${_stack},\"activities\":${_acts}}" \
          >/dev/null 2>&1; then
        _posted=$(( _posted + 1 ))
        _windowed="${_windowed}${_role}=${_url}"$'\n'
      else
        echo "!!! multinode: /start_profile failed on the $_role engine ($_url)" >&2
      fi
    done <<< "$_MN_PD_ENGINES"
    if [ "$_posted" -eq 0 ]; then
      echo "!!! multinode: no engine accepted a profiler window" >&2
      return 1
    fi
    # How many trace files to expect: ranks PER ENGINE times the engines that
    # accepted -- not $TP, which in this topology is the caller's name for the
    # whole pin and would be right only by coincidence. An engine that refused
    # the window contributes nothing, so counting it would hold the wait until
    # the timeout and then report a set that was never coming.
    if [ -z "${GEAK_MN_EXPECT_TRACES:-}" ]; then
      local _etp
      _etp="$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("pd") or {}).get("engine_tp") or 0)' "${OUT_DIR:-.}/mn_launch.json" 2>/dev/null || echo 0)"
      [ "${_etp:-0}" -gt 0 ] && _expect=$(( _etp * _posted ))
    fi
    echo ">>> multinode: profiler window open on $_posted engine(s);" \
         "expecting $_expect rank trace(s)"

    # num_steps cannot close BOTH windows, and the engine it fails to close is
    # the one whose absence is hardest to notice. It counts FORWARD STEPS, and
    # for a single workload the two engines run wildly different numbers of
    # them: at ISL512/OSL32 decode takes roughly ten steps per prefill pass. So
    # decode reaches num_steps, saves, and looks healthy, while prefill is still
    # recording when the teardown kills it -- and torch writes nothing for a
    # window that never closed. That is exactly how round_0 came back as "8/16
    # traces landed": not a lost rank, an engine whose window was still open.
    #
    # So the window is bounded by TIME here and closed explicitly on every
    # engine -- the same fallback the backend adapter keeps for its single
    # endpoint, promoted to the normal path because here it is the normal case.
    # The side benefit is the one the per-role Top-N weighting depends on: both
    # engines then describe the same interval, so their GPU-time totals are
    # comparable rather than a measure of who recorded longer.
    sleep "${GEAK_MN_PD_PROFILE_SEC:-${PROFILE_WINDOW_SEC:-20}}"
    while IFS='=' read -r _role _url; do
      [ -n "$_url" ] || continue
      curl -sf -X POST "${_url}/stop_profile" >/dev/null 2>&1 \
        || echo ">>> multinode: the $_role engine had already closed its window" >&2
    done <<< "$_windowed"
  elif declare -F adapter_profile_window_native >/dev/null; then
    adapter_profile_window_native || {
      echo "!!! multinode: the profiler window failed on the master rank" >&2
      return 1
    }
  else
    # An older bench_e2e.sh that does not preserve the native window. Issue the
    # request directly rather than skipping the capture: fewer knobs than the
    # backend's version, but a real window beats a silently absent one.
    echo ">>> multinode: no adapter_profile_window_native; posting /start_profile directly" >&2
    curl -sf -X POST "${BASE_URL}/start_profile" -H 'Content-Type: application/json' \
      -d "{\"output_dir\":\"${PROFILE_DIR}\",\"num_steps\":${PROFILE_NUM_STEPS},\"record_shapes\":true,\"with_stack\":false,\"activities\":[\"CPU\",\"GPU\"]}" \
      >/dev/null 2>&1 || {
        echo "!!! multinode: /start_profile request failed" >&2; return 1; }
  fi

  # Waiting for the FILE COUNT is not enough, and getting that wrong looks like
  # success: torch creates each rank's file when the export starts and then
  # streams into it (~25 MB/rank here), so the count reaches N while all
  # sixteen are still writing. Measured consequence of counting: the set was
  # read with three traces truncated mid-write and six still at zero bytes,
  # and the teardown then killed the ranks that had not started flushing.
  #
  # So the completion signal is TOTAL BYTES going quiet: every expected file
  # non-empty, and the byte total unchanged for GEAK_MN_TRACE_SETTLE_S. Empty
  # files are not counted, so a rank that has not begun writing still holds
  # the wait.
  local _settle="${GEAK_MN_TRACE_SETTLE_S:-30}" _quiet="${GEAK_MN_TRACE_QUIET_S:-180}"
  local _interval=5 _bytes _last_bytes=-1 _stable=0 _nochange=0
  _deadline=$(( $(date +%s) + ${GEAK_MN_TRACE_TIMEOUT:-1800} ))
  while [ "$(date +%s)" -lt "$_deadline" ]; do
    _after=$(find "$PROFILE_DIR" -maxdepth "$_depth" -name '*.trace.json*' -size +1c \
             2>/dev/null | wc -l)
    _bytes=$(find "$PROFILE_DIR" -maxdepth "$_depth" -name '*.trace.json*' \
             -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "$_bytes" != "$_last_bytes" ]; then
      _last_bytes="$_bytes"; _stable=0; _nochange=0
    else
      _stable=$(( _stable + _interval )); _nochange=$(( _nochange + _interval ))
    fi
    if [ "$(( _after - _before ))" -ge "$_expect" ] && [ "$_stable" -ge "$_settle" ]; then
      break
    fi
    # Quiet but incomplete: a rank died, or its export never started. Waiting
    # out the full timeout would hold the server and the cluster lock for
    # nothing, so stop and let the gate describe what is actually there.
    if [ "$_nochange" -ge "$_quiet" ]; then
      echo "!!! multinode: $(( _after - _before ))/$_expect trace(s) and no new" \
           "bytes for ${_quiet}s ($_bytes bytes total); giving up on the rest" >&2
      break
    fi
    sleep "$_interval"
  done
  _after=$(find "$PROFILE_DIR" -maxdepth "$_depth" -name '*.trace.json*' -size +1c 2>/dev/null | wc -l)
  echo ">>> multinode: $(( _after - _before ))/$_expect rank trace(s) landed in" \
       "$PROFILE_DIR ($(( ${_last_bytes:-0} / 1048576 )) MiB total," \
       "quiet for ${_stable}s)"

  # The gate decides whether this set can be believed, and leaves exactly one
  # trace at the top level so the parser's selection stops being a race. Under
  # pd_disagg it does that per ENGINE, reading which engines to expect from the
  # launch record -- the same file the window above addressed them from, so the
  # gate cannot end up checking for an engine the capture never went to.
  local _launch_json="${OUT_DIR:-.}/mn_launch.json"
  [ "$_MN_TOPOLOGY" = "pd_disagg" ] && [ -f "$_launch_json" ] || _launch_json=""
  PROFILE_DIR="$PROFILE_DIR" TP="$TP" \
    python3 "$_MN_DIR/mn.py" trace-gate \
      --profile-dir "$PROFILE_DIR" --expect-ranks "$_expect" \
      --gpus-per-node "${GEAK_MN_GPUS_PER_NODE:-8}" \
      ${_launch_json:+--launch-json "$_launch_json"}
  local _rc=$?
  if [ "$_rc" -ne 0 ]; then
    # Non-fatal by design: an incomplete trace set still supports a ranking,
    # and failing the round here would throw away the bench that came with it.
    # The verdict is on disk (mn_trace_report.json) so a Top-N built on a
    # partial set is never mistaken for one built on the whole server.
    echo "!!! multinode: the trace set did NOT pass the gate; see" \
         "$PROFILE_DIR/ranks/mn_trace_report.json before trusting the Top-N." >&2
  fi
  [ "$(( _after - _before ))" -gt 0 ]
}

# adapter_health -- pd_disagg only. bench_e2e.sh preserved the backend's version
# as adapter_health_native (it clones launch/health/profile_window before sourcing
# this file), and that version asks the load balancer only.
#
# Which is not enough here, and the gap is not cosmetic: the Rust router binds its
# port and answers /health as soon as it is up, whether or not a worker is
# reachable. So the native probe goes green while an engine is still loading
# weights -- or has died -- and the bench then starts against a pair that cannot
# answer. The symptom is a run of timeouts attributed to the model or the network,
# not a server that was never ready.
#
# So all three must answer: the LB (the client's address) and BOTH engines. A
# per-engine failure names the role, because "prefill is not up" and "decode is
# not up" have different causes and only one of them is about KV transfer.
if [ "$_MN_TOPOLOGY" = "pd_disagg" ]; then
  adapter_health() {
    local _role _url
    curl -sf "${BASE_URL}/health" >/dev/null 2>&1 || return 1
    printf '%s\n' "$_MN_PD_ENGINES" | while IFS='=' read -r _role _url; do
      [ -n "$_url" ] || continue
      curl -sf --max-time "${GEAK_MN_PD_HEALTH_TIMEOUT_S:-10}" "${_url}/health" \
        >/dev/null 2>&1 || exit 1
    done
  }
fi

adapter_launch() {
  local _out_dir="${OUT_DIR:-$(pwd)}"
  local _launchlog="$_out_dir/mn_launch.log"
  local _suplog="$_out_dir/mn_supervisor.log"

  # Both logs stay OUT of $LOG. bench_e2e.sh greps $LOG for OOM/fatal markers,
  # watches its size for stalls, and fingerprints the served stack from it;
  # rank0's server output is the only thing that may appear there. (rank0 writes
  # it directly on the shared mount, so there is nothing to copy back.)
  echo ">>> multinode launcher: starting $(basename "$_MN_STATE") topology; launch log: $_launchlog"
  ( cd "$_MN_DIR" && python3 mn.py launch ) >> "$_launchlog" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! multinode launcher: launch failed (rc=$rc). Last launch log:" >&2
    tail -n 60 "$_launchlog" >&2 2>/dev/null || true
    echo "!!! multinode launcher: last rank0 log:" >&2
    tail -n 40 "$LOG" >&2 2>/dev/null || true
    return 2
  fi
  grep -E '^>>> multinode (rank|:)' "$_launchlog" 2>/dev/null | tail -n 8 || true

  # The supervisor is the pid bench_e2e.sh owns: `kill -0` on it answers "are the
  # remote ranks still there?", and the teardown's TERM is what reaps them.
  # Launched through $SERVER_LAUNCH_PREFIX (setsid) like every adapter, so
  # pgid == pid and teardown can prove the group is ours -- but note the proof
  # covers only this local process. The ranks themselves are reaped by the
  # supervisor's handler, never by a local signal, which is why the grace above
  # is 90s rather than 10.
  ${SERVER_LAUNCH_PREFIX:-} env \
    OUT_DIR="$_out_dir" \
    PYTHONPATH="$_MN_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    python3 "$_MN_DIR/mn.py" supervise >> "$_suplog" 2>&1 &
  SERVER_PID=$!

  # A supervisor that died on import (no ray, bad state file) must not be handed
  # over as a live server: bench_e2e.sh would read the immediate `kill -0` failure
  # as the SERVER dying early and blame the launch instead of the tooling.
  sleep 2
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!!! multinode launcher: the supervisor exited immediately; see $_suplog" >&2
    tail -n 30 "$_suplog" >&2 2>/dev/null || true
    echo "!!! The ranks may be RUNNING but unsupervised. Reaping them now." >&2
    ( cd "$_MN_DIR" && python3 -c 'import mn as m; m.teardown(m.load_state())' ) >&2 2>&1 || true
    SERVER_PID=""
    return 2
  fi
  echo ">>> multinode launcher: ranks spawned, supervisor pid $SERVER_PID," \
       "endpoint $BASE_URL (cold start is minutes; the health loop decides)."
}
