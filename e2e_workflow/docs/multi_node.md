# Multi-node serving (one server, N pinned machines)

`BENCH_LAUNCHER=multinode` serves **one** server instance whose TP group spans
several machines. Only *who starts the server* changes: the backend adapter, the
bench client, the metric and the acceptance contract are the single-host ones,
which is what keeps a two-node tok/s number comparable to the rest of GEAK's
results instead of making it a separate dialect.

## Where the framework knowledge lives

Nothing under `scripts/multi_node/` knows how to start a server. The split is:

| layer | owns | lives in |
| --- | --- | --- |
| **where** | which machines, which NIC, which master, teardown, trace gate | `launchers/multinode.sh`, `multi_node/*` |
| **how** | the launch line, and how the framework says *"I am rank i of n"* | `adapters/sglang.sh` |
| **what** | model recipe and the flags GEAK sweeps | `GEAK_MN_RECIPE_ARGS` / `EXTRA_SERVER_ARGS` |

That is three files, and the two seams between them are the only ones the
runtime forces:

| file | why it is separate |
| --- | --- |
| `adapters/launchers/multinode.sh` | `bench_e2e.sh` sources `adapters/launchers/$BENCH_LAUNCHER.sh` and calls the `adapter_*` functions it defines **in its own shell**. A launcher has to be a shell file at that path. |
| `multi_node/mn.py` | five subcommands — `preflight`, `launch`, `supervise`, `reap`, `trace-gate`. They are separate *processes* (`supervise` is the pid `bench_e2e.sh` kills; `reap` has to outlive it) but not separate files. |
| `multi_node/serve.sh.tmpl` | rendered per rank and run **on the far node**, so it is bash either way; kept as a file so `bash -n` and an editor see it as bash. |

Each rank sources `adapters/$BACKEND.sh` **on its own node** and calls
`adapter_dist_args RANK NNODES MASTER DIST_PORT`. Adding a backend is therefore
two lines in its adapter, not a fork of the launcher:

```bash
adapter_dist_mode() { echo per_rank; }   # one server process per node
adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }
```

`adapter_dist_mode` exists because the process model is not universal: vllm's ray
executor fans out to the other nodes from a *single* process (`head_only`), so
there is no per-rank launch to do. The launcher implements `per_rank` and refuses
anything else by name rather than launching sglang's shape and calling it vllm's.

Flags are layered in this order, the same one `launchers/magpie.sh` uses:
recipe (`GEAK_MN_RECIPE_ARGS`, plus `GEAK_MN_PD_ARGS_<ROLE>` on a disaggregated
engine) → GEAK's shared accepted flags (`EXTRA_SERVER_ARGS`) → that engine's
own accepted flags (`GEAK_MN_PD_SWEEP_ARGS_<ROLE>`, pd_disagg only) → the
rank/world or prefill/decode triple **last**. argparse is last-wins, so a
swept flag always beats the recipe, a per-engine sweep beats the shared sweep
on that engine only, and nothing a sweep sets can change which rank a process
is or which half of a disaggregated pair it is.

Overlays follow the same rule on `PYTHONPATH`: that engine's own overlay
(`GEAK_MN_PD_OVERLAY_<ROLE>`, a kernel accepted on that engine's shapes) comes
ahead of the shared `OVERLAY_PYTHONPATH`, so import resolution picks the
engine's copy of a module first and the other engine never loads it.

`mn_launch.json` records what each engine was actually launched with
(`pd.effective_args_by_role`) and every one-engine sweep key that the shared
layer also sets (`pd.sweep_leaks`, also printed at launch). A leaked key runs
on both engines, so the delta measured with it belongs to neither.

Measured on two MI300X nodes (DeepSeek-V4-Flash-FP8, TP16 DP2, ISL/OSL 1024,
CONC 8): **165.42 tok/s, spread 1.96%** over three timed rounds.

## Why there is no job-submission API here

Ray is used as a **process launcher, not a scheduler**. The distributed work is
the framework's own: each rank is told which rank it is (`adapter_dist_args`) and
the ranks form their own `torch.distributed` world over the cluster NIC. Ray only
picks the machine (`NodeAffinitySchedulingStrategy`) and spawns one process
there. Nothing in the measurement path needs `ray job submit`.

That choice is what preserves `bench_e2e.sh`'s lifecycle contract: the script
owns a **local pid** and `kill`ing it must stop the server, because every stall
check, teardown and acceptance gate is built on that. A job-id-based launcher
would have replaced the contract. Instead `mn.py supervise` *is* that local pid;
on SIGTERM it reaps the remote ranks through Ray.

| | Ray's role | What owns the collectives |
| --- | --- | --- |
| this launcher | pin a node, spawn a process | sglang / torch.distributed |
| a rayjob-based launcher | queue and manage a job | sglang / torch.distributed |

The difference is only in how the process gets started; nothing about the
distributed execution changes. So the rayjob path buys nothing here and costs the
local-pid contract.

## What the caller must hand over

GEAK does not create, resize or tear down the cluster.

1. **A running Ray cluster** covering the candidate machines. One `ray start
   --head --port=6779` plus `ray start --address='<head-ip>:6779'` on the others
   is enough; the head does not need to be one of the serving nodes and uses no
   GPU.
2. **A filesystem every node mounts** (`GEAK_MN_EXT_SHARED_FS`). Mandatory. It is
   where the overlay, the per-rank logs and the traces live, and the reason no
   step of this launcher ships files between nodes. The preflight proves it with
   a bidirectional marker exchange, because `df` showing the same size proves
   nothing.
3. **Idle machines.** Node selection refuses any box with GPU-resident
   processes, even at 0% utilization: a resident-but-idle neighbour still holds
   the dist-init port, and the launch would die on it.

## Running it

### 1. Pin the machines (once, in Setup)

```bash
export GEAK_MN_EXT_SHARED_FS=/workhome/geak      # mounted on every node
cd e2e_workflow/scripts/multi_node
python3 mn.py preflight --nnodes 2               # add --repin to re-select
```

Nothing here needs the machines' addresses in advance, because nobody has them:
the caller hands over a Ray cluster, `ray.nodes()` returns node **ids**, and each
candidate is then asked — by a task pinned to it — what interfaces it has. The
fabric is the subnet those candidates turn out to share, and `master_ip` only
exists once the selection is made. Set `GEAK_MN_CLUSTER_SUBNET` to override that
inference when a cluster has two shared networks and only one of them is the
fast one; `mn_preflight.json` records which way it was decided.

Writes the pin to `$GEAK_MN_EXT_SHARED_FS/mn_state.json` and the evidence to
`mn_preflight.json`. Re-running without `--repin` reuses the pin and only checks
that every pinned node is still alive — that is the intended behaviour, because a
pin that moves between the baseline leg and the candidate leg turns the A/B into
a measurement of which machines were free.

### 2. Run the benchmark

The command that produced the numbers above:

```bash
cd e2e_workflow/scripts
BACKEND=sglang \
MODEL=/apps/DeepSeek-V4-Flash-FP8 \
TP=16 GPU=0,1,2,3,4,5,6,7 \
MEM_FRACTION=0.90 \
BENCH_LAUNCHER=multinode \
GEAK_MN_EXT_SHARED_FS=/workhome/geak \
GEAK_MN_RECIPE_ARGS="--dp 2 --enable-dp-attention --attention-backend dsv4 ..." \
GEAK_MN_RECIPE_ENV="SYNC_TOKEN_IDS_ACROSS_TP=1 ..." \
OUT_DIR=/workhome/geak/exp/band_conc8 \
ISL=1024 OSL=1024 CONC=8 NUM_PROMPTS=40 \
GEAK_REPEAT_MODE=warm_server REPEATS=3 PROFILE=0 \
BENCH_OUTER_WARMUP_ROUNDS=2 \
SKILL_DIR=/path/to/GEAK/e2e_workflow \
  bash ./bench_e2e.sh
```

`TP=16` is the world size across both boxes; `GPU=0,...,7` is the **per-node**
device list. Those two disagreeing is not a mistake: once the TP group spans
machines, the only meaning a device list can have is per-node, and it is what the
GPU allow-list is checked against and what the serving lock is keyed on.

`BENCH_OUTER_WARMUP_ROUNDS=2` is not cosmetic on this stack — see
[the warmup section](#why-two-warmup-rounds).

### 3. Or run it under the workflow

`interface/run_e2e.py` is **not modified** for this: the two things it would
otherwise have to derive are both statable from outside, so a full workflow run
needs no interface change at all.

```bash
export BENCH_LAUNCHER=multinode          # apply_bench_launcher honours $BENCH_LAUNCHER
export GEAK_MN_EXT_SHARED_FS=/workhome/geak
```

```json
{
  "model_path": "/workhome/geak/models/Qwen3-8B-FP8",
  "tp": 16,
  "gpu_ids": "0,1,2,3,4,5,6,7"
}
```

Both are load-bearing, and both fail quietly if omitted:

* Without `BENCH_LAUNCHER`, the launcher is derived from the backend and lands on
  `native`, which starts ONE local server. The run then measures a TP group that
  fits on one box while reporting the topology that was asked for.
* Without `gpu_ids`, the device set defaults to `0..tp-1`. Across machines that
  names sixteen cards on a box that has eight — the per-node list is the only
  meaning a device set can have once the TP group spans hosts, and it is what the
  GPU allow-list is checked against and what the serving lock is keyed on.

Data-parallel width belongs in neither: it is an ordinary server flag, so it
rides the normal server-args channel and the launcher never learns its spelling.

Ports, the Ray address and the fabric prefix come from the pin, so the workflow
never states them; `mn.py preflight` wrote them into `mn_state.json`.

## Environment reference

| variable | default | meaning |
| --- | --- | --- |
| `GEAK_MN_EXT_SHARED_FS` | — | **required**; path mounted on every node |
| `GEAK_MN_NODES` | 2 | node count for the preflight |
| `GEAK_MN_GPUS_PER_NODE` | 8 | per-node device count |
| `GEAK_MN_SERVE_PORT` | 30890 | HTTP port on rank0 (inside the 30000–30999 window) |
| `GEAK_MN_DIST_INIT_PORT` | 30531 | torch.distributed rendezvous port |
| `GEAK_MN_RECIPE_ARGS` | — | model recipe, the BASE layer under `EXTRA_SERVER_ARGS` |
| `GEAK_MN_RECIPE_ENV` | — | recipe env, placed before `EXTRA_ENV` (`env` is last-wins) |
| `GEAK_MN_PD_ARGS_PREFILL` / `GEAK_MN_PD_ARGS_DECODE` | — | pd_disagg only. Experimenter's baseline for that engine, inside the recipe layer, so a shared sweep can still override it |
| `GEAK_MN_PD_SWEEP_ARGS_PREFILL` / `GEAK_MN_PD_SWEEP_ARGS_DECODE` | — | pd_disagg only. Flags the sweep accepted for that engine alone. Appended after `EXTRA_SERVER_ARGS` on that engine; the other engine does not receive them |
| `GEAK_MN_PD_SWEEP_ENV_PREFILL` / `GEAK_MN_PD_SWEEP_ENV_DECODE` | — | pd_disagg only. Same split for env. Appended after `EXTRA_ENV` on that engine |
| `GEAK_MN_PD_OVERLAY_PREFILL` / `GEAK_MN_PD_OVERLAY_DECODE` | — | pd_disagg only. Kernel overlay accepted for that engine alone. Placed ahead of `OVERLAY_PYTHONPATH` on that engine; recorded in `mn_launch.json` as `pd.sweep_by_role.<role>.overlay` |
| `GEAK_MN_BACKEND_ADAPTER` | `adapters/$BACKEND.sh` | override the adapter each rank sources |
| `GEAK_MN_RAY_ADDRESS` | `auto` | cluster address; `auto` = local head |
| `GEAK_MN_CLUSTER_SUBNET` | inferred | fabric prefix, e.g. `10.235.200.`; overrides the inference |
| `GEAK_MN_STATE` | `$SHARED_FS/mn_state.json` | the pin |
| `GEAK_MN_DIR` | auto-resolved | override if `multi_node/` is not next to the adapter |
| `GEAK_MN_DRY_RUN` | — | `1` = render the per-rank scripts and stop, cluster untouched |
| `GEAK_MN_TRACE_SETTLE_S` | 30 | trace bytes must be quiet this long to count as exported |
| `GEAK_MN_TRACE_QUIET_S` | 180 | give up on an incomplete trace set after this |
| `GEAK_MN_TRACE_TIMEOUT` | 1800 | hard cap on the trace wait |
| `GEAK_MN_EXPECT_TRACES` | `$TP` | expected per-rank trace count |
| `BENCH_OUTER_WARMUP_ROUNDS` | 1 | untimed full rounds before the timed ones |

## Artifacts

Under `$OUT_DIR`:

| file | what to check |
| --- | --- |
| `bench_summary.json` | `throughput_tok_s_median`, `..._spread_pct`, `outer_warmup_rounds`, `usable_for_acceptance` |
| `mn_launch.json` / `.log` | per-rank spawn record; `rc != 0` means no server was started |
| `mn_supervisor.json` | `verdict` (`torn_down` = ranks reaped), `cleared` |
| `server.log` / `server.rank1.log` | rank0's log is the one `bench_e2e.sh` greps for OOM/fatal |
| `profile/ranks/mn_trace_report.json` | the trace gate's verdict (see below) |

Under `$GEAK_MN_EXT_SHARED_FS`: `mn_state.json` (the pin) and
`mn_preflight.json` (selection + shared-mount evidence).

## The trace gate

With `PROFILE=1`, all N ranks export into the shared profile directory and the
gate decides whether that set can be believed. `verdict: pass` requires every
expected rank present and readable, **collective kernels in every rank** (a TP
group spanning machines that shows no collectives did not do what it claimed),
both nodes represented, and the parser's trace selection landing on the selected
rank. A failed gate is non-fatal — the bench that came with it is still valid —
but the verdict is on disk so a Top-N built on a partial set is never mistaken
for one built on the whole server.

Two things it exists to prevent, both of which look like a successful capture:

- **Waiting for the file count returns too early.** torch creates each rank's
  file when the export starts and then streams ~25 MB into it, so the count
  reaches N while all N are still writing. Measured: three traces truncated
  mid-write, six at zero bytes, and a reported 62% node imbalance that did not
  exist. The completion signal is total bytes going quiet instead.
- **A stray `.json` shadows the trace.** Selection downstream is
  `ls -t *.json.gz *.json | head -1`. The gate's own report used to be the
  newest file in that directory and got parsed *as* the trace, producing an
  empty Top-N that reads as a capture failure. The report now lives in `ranks/`,
  and the gate reproduces the selection and fails if anything would shadow the
  intended trace.

It also moves the non-selected ranks into `ranks/` so exactly one trace remains
at the top level: with N siblings, `ls -t | head -1` is decided by which rank's
NFS write landed first, and two legs of an A/B would compare different ranks.

## Why two warmup rounds

On this stack one warmup round is not enough, and the shortfall shows up as
noise rather than as an error. Three timed rounds after **one** warmup:
95.3 / 147.8 / 143.8 tok/s — the first round 34% below the next two, reported
spread 36.5%. rank0's `#cached-token` count is 0 through the warmup and the whole
first timed round, then jumps to 75%: the prefix cache only pays off once the
radix tree is populated, and the cold prefill also steals decode cycles
(TPOT 73.8 → 49.8 ms).

With **two** warmup rounds: 165.4 / 166.7 / 163.5 tok/s, spread 1.96%, TTFT
665 / 604 / 610 ms (4945 ms when cold). The default stays 1, so single-host paths
are unchanged; the value is recorded in `bench_summary.json` because two legs
compared against each other must use the same one.

## When comparing two legs

- Same pin. A different machine pair measured ~13% apart under an identical
  protocol — larger than any flag result worth reporting. Check that
  `mn_state.json`'s `pinned_at` is the same for both legs.
- Same `outer_warmup_rounds`.
- Beat the noise band by enough to mean it. At 1.96% over three rounds, treat
  differences under ~5% as "add rounds", not as a result.

## Failure modes and what they print

| symptom | cause | what to do |
| --- | --- | --- |
| `no pinned node set at …` | preflight never ran | run `mn.py preflight` |
| `only N idle node(s), need M` | a node has GPU-resident pids | the message says whether they are another container's; if so the box is shared, `--repin` |
| `… is NOT shared across the picked nodes` | the mount is local, or read-only from one side | fix the mount; do not proceed |
| `the pinned nodes did not come clear` | a previous scheduler still holds the dist-init port | wait, or `--repin` if the holder is a co-tenant |
| `the supervisor exited immediately` | no `ray` in the env, or a bad state file | see `mn_supervisor.log`; the launcher reaps the ranks it spawned |
| `N/M rank trace(s) landed` with N<M | a rank died before exporting | check that rank's `server.rank*.log` |
| a rank hangs on the first collective | `NCCL_SOCKET_IFNAME` unset, so RCCL picked calico/docker0/veth | `serve.sh.tmpl` exports it; a hand-rolled `torchrun` has no such protection |

## Tests

```bash
cd e2e_workflow/scripts/tests
python3 -m unittest test_mn
```

The gate and window tests run without a cluster: they fabricate rank traces,
including a streaming writer that drips gzip bytes over several seconds to prove
the wait does not return early.
