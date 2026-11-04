#!/usr/bin/env bash
# CPU-only test of adapters/launchers/atomesh.sh. Stub `python3 -m
# atom.entrypoints.openai_server` and `atomesh` with tiny HTTP servers, source
# the launcher the way bench_e2e.sh does, and check: the launch plan (GPU
# chunking, ports, per-worker flags/env/overlay), pd role checks, router args,
# EXTRA_ENV ATOMESH_* overrides, fail-fast on a dying worker, and that a TERM
# to the supervisor group leaves no component behind.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/../adapters/launchers/atomesh.sh"
REAL_PY="$(command -v python3)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fails=0
ok()   { echo "  ok   $*"; }
bad()  { echo "  FAIL $*"; fails=$((fails + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

mkdir -p "$WORK/bin"
cat > "$WORK/stub_server.py" <<'PY'
import http.server, json, os, sys
args = sys.argv[1:]
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
port = int(opt("--server-port") or opt("--port"))
role = None
if "--kv-transfer-config" in args:
    role = json.loads(opt("--kv-transfer-config"))["kv_role"]
if os.environ.get("STUB_DIE_PORT") == str(port):
    print("run_engine failed", flush=True)
print(f"stub up port={port} role={role} hip={os.environ.get('HIP_VISIBLE_DEVICES')} "
      f"pp={os.environ.get('PYTHONPATH','')} foo={os.environ.get('FOO')} "
      f"oqc={opt('--online_quant_config')} argv={' '.join(args)}", flush=True)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = b"ok"
        if self.path == "/kv_transfer_info":
            body = json.dumps({"kv_role": role}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
cat > "$WORK/bin/python3" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "-m" ] && [ "\${2:-}" = "atom.entrypoints.openai_server" ]; then
  shift 2; exec "$REAL_PY" "$WORK/stub_server.py" "\$@"
fi
exec "$REAL_PY" "\$@"
EOF
cat > "$WORK/bin/atomesh" <<EOF
#!/usr/bin/env bash
shift  # launch
echo "router argv: \$*"
exec "$REAL_PY" "$WORK/stub_server.py" "\$@"
EOF
chmod +x "$WORK/bin/python3" "$WORK/bin/atomesh"
export PATH="$WORK/bin:$PATH"

copy_function() { local f; f="$(declare -f "$1")"; eval "${f/$1/$2}"; }
# bench_e2e.sh sources the backend adapter before the launcher; the launcher splits
# EXTRA_SERVER_ARGS with its _atom_server_argv.
# shellcheck disable=SC1091
source "$HERE/../adapters/atom.sh"
adapter_profile_window() { echo "native profile window BASE_URL=$BASE_URL"; }
SERVER_LAUNCH_PREFIX=setsid
# shellcheck disable=SC1090
source "$LAUNCHER"

free_port() { "$REAL_PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }

wait_router() {  # wait_router SECONDS -> 0 when the router answers
  local i
  for ((i = 0; i < $1 * 5; i++)); do
    curl -sf --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 0.2
  done
  return 1
}

group_gone() {  # group_gone PGID SECONDS
  local i
  for ((i = 0; i < $2 * 5; i++)); do
    pgrep -g "$1" >/dev/null 2>&1 || return 0
    sleep 0.2
  done
  return 1
}

run_case() {  # run_case NAME ; uses the current env, leaves SERVER_PID running
  OUT_DIR="$WORK/$1"; LOG="$OUT_DIR/server.log"; PROFILE_DIR="$OUT_DIR/profile"
  mkdir -p "$OUT_DIR"; PORT="$(free_port)"; HOST=127.0.0.1
  ATOMESH_WORKER_PORT_BASE="${CASE_BASE:-$(free_port)}"
  export OUT_DIR LOG PROFILE_DIR PORT HOST ATOMESH_WORKER_PORT_BASE
  adapter_launch
}

stop_case() {
  local pgid; pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null
  check "teardown reaps every component" "group_gone '$pgid' 15"
}

BACKEND=atom; export BACKEND
MODEL=/models/stub; MAX_MODEL_LEN=4096; MEM_FRACTION=0.6; ISL=1024; OSL=256
OVERLAY_PYTHONPATH=/overlay/x; ATOMESH_STOP_GRACE_S=1
export MODEL MAX_MODEL_LEN MEM_FRACTION ISL OSL OVERLAY_PYTHONPATH ATOMESH_STOP_GRACE_S

echo "case 1: regular, GPU=0,1,6,7 TP=1, args/env/overlay on every worker, policy from EXTRA_ENV"
GPU=0,1,6,7 TP=1 EXTRA_ENV="FOO=bar ATOMESH_POLICY=cache_aware"
EXTRA_SERVER_ARGS="--max-num-seqs 64 --online_quant_config '{\"exclude_layer\": [\"*.mlp.gate\"]}'"
export GPU TP EXTRA_SERVER_ARGS EXTRA_ENV
unset ATOMESH_MODE
run_case regular
check "router healthy" "wait_router 20"
check "worker output reaches \$LOG while running" "grep -q '^\[worker-0\] stub up' '$LOG'"
check "4 workers planned" "[ \$(wc -l < '$OUT_DIR/atomesh/plan.tsv') -eq 4 ]"
check "GPU chunks 0/1/6/7" "grep -q 'worker-3=gpu\[7\]' '$OUT_DIR/atomesh/topology.txt'"
check "EXTRA_SERVER_ARGS on every worker" "[ \$(grep -c 'max-num-seqs 64' '$LOG') -eq 4 ]"
check "quoted JSON arg arrives as one unquoted token" "[ \$(grep -cF 'oqc={\"exclude_layer\": [\"*.mlp.gate\"]}' '$LOG') -eq 4 ]"
check "EXTRA_ENV on every worker" "[ \$(grep -c 'foo=bar' '$LOG') -eq 4 ]"
check "overlay on every worker" "[ \$(grep -c 'pp=/overlay/x' '$LOG') -eq 4 ]"
check "ATOMESH_* not forwarded to workers" "! grep -q 'ATOMESH_POLICY' '$OUT_DIR/atomesh/plan.tsv'"
check "policy from EXTRA_ENV reaches router" "grep -q -- '--policy cache_aware' '$LOG'"
check "router gets all 4 worker urls" "[ \$(grep 'router argv' '$LOG' | grep -o '127.0.0.1:[0-9]*' | wc -l) -eq 4 ]"
check "profiler only on worker-0" "[ \$(grep -c 'torch-profiler-dir' '$OUT_DIR/atomesh/plan.tsv') -eq 1 ] && grep '^worker-0' '$OUT_DIR/atomesh/plan.tsv' | grep -q torch-profiler-dir"
check "profile window targets worker-0" "adapter_profile_window | grep -q \"BASE_URL=http://127.0.0.1:\$ATOMESH_WORKER_PORT_BASE\$\""
stop_case

echo "case 2: pd 1P2D, TP=1"
GPU=0,1,6 TP=1 EXTRA_SERVER_ARGS="" EXTRA_ENV="" ATOMESH_MODE=pd
export GPU TP EXTRA_SERVER_ARGS EXTRA_ENV ATOMESH_MODE
run_case pd
check "router healthy" "wait_router 20"
check "roles verified" "grep -q 'prefill-0 kv_transfer_info' '$LOG' && grep -q 'decode-1 kv_transfer_info' '$LOG'"
check "router in pd mode with 1 prefill + 2 decode" "grep 'router argv' '$LOG' | grep -q -- '--pd-disaggregation' && [ \$(grep 'router argv' '$LOG' | grep -o -- '--decode http' | wc -l) -eq 2 ]"
check "prefix caching disabled on pd workers" "[ \$(grep -c 'no-enable_prefix_caching' '$OUT_DIR/atomesh/plan.tsv') -eq 3 ]"
check "mooncake tcp env" "[ \$(grep -c 'MC_FORCE_TCP=true' '$OUT_DIR/atomesh/plan.tsv') -eq 3 ]"
check "profiler on decode-0" "grep '^decode-0' '$OUT_DIR/atomesh/plan.tsv' | grep -q torch-profiler-dir"
stop_case

echo "case 3: a worker whose engine core fails after startup kills the leg fast"
GPU=0,1 TP=1 ATOMESH_MODE=regular
export GPU TP ATOMESH_MODE
CASE_BASE="$(free_port)"; unset ATOMESH_POLICY
export STUB_DIE_PORT="$((CASE_BASE + 10))"
run_case die
pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
check "supervisor exits on engine failure" "group_gone '$pgid' 20"
check "failure is logged" "grep -q 'FATAL: worker-1 engine core failed' '$LOG'"
unset STUB_DIE_PORT CASE_BASE

echo "case 4: a stale engine failure from a previous run in the same OUT_DIR is not re-read"
mkdir -p "$WORK/stale/atomesh"; echo "run_engine failed" > "$WORK/stale/atomesh/worker-0.log"
GPU=0,1 TP=1 ATOMESH_MODE=regular; export GPU TP ATOMESH_MODE
run_case stale
check "router healthy despite stale log" "wait_router 20"
stop_case

echo "case 5: default worker ports sit outside bench_e2e's PORT window"
OUT_DIR="$WORK/ports"; LOG="$OUT_DIR/server.log"; PORT=30123; unset ATOMESH_WORKER_PORT_BASE
export OUT_DIR LOG PORT; mkdir -p "$OUT_DIR"
adapter_launch >/dev/null
check "worker-0 on PORT+2000" "grep -q \"^worker-0	32123	\" '$OUT_DIR/atomesh/plan.tsv'"
pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"; kill -TERM -- "-$pgid" 2>/dev/null
group_gone "$pgid" 15

echo "case 6: invalid setups are refused"
check "EXTRA_SERVER_ARGS with unbalanced quotes" "! (EXTRA_SERVER_ARGS=\"--x 'oops\" OUT_DIR='$WORK/bad4' LOG='$WORK/bad4.log' adapter_launch) 2>/dev/null"
check "BACKEND other than atom" "! (BACKEND=vllm OUT_DIR='$WORK/bad0' LOG='$WORK/bad0.log' adapter_launch) 2>/dev/null"
check "no setsid prefix" "! (SERVER_LAUNCH_PREFIX='' OUT_DIR='$WORK/bad3' LOG='$WORK/bad3.log' adapter_launch) 2>/dev/null"
GPU=0,1,2 TP=2 ATOMESH_MODE=regular; export GPU TP ATOMESH_MODE
check "GPU count not a multiple of TP" "! (OUT_DIR='$WORK/bad1' LOG='$WORK/bad1.log' adapter_launch) 2>/dev/null"
GPU=0 TP=1 ATOMESH_MODE=pd; export GPU TP ATOMESH_MODE
check "pd with a single worker" "! (OUT_DIR='$WORK/bad2' LOG='$WORK/bad2.log' adapter_launch) 2>/dev/null"

echo
if [ "$fails" -eq 0 ]; then echo "PASS test_atomesh_launcher"; else echo "FAIL test_atomesh_launcher ($fails)"; exit 1; fi
