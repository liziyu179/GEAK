# Distributed Collectives as an Optimization Target

This is the contract for `execution_model = "distributed_collective"` — a kernel task whose operator
cannot be isolated on one card, because isolating it removes the thing being measured. Today that means
cross-node all-reduce; it means reduce-scatter, all-gather and all-to-all the moment someone writes the
`cases.py` for them.

**The one thing that changed.** Every other operator is measured by `harness_lib._run_leg` as one local
subprocess on one card. That is why collectives were previously routed to config levers only: an
all-reduce isolated on one rank has no peers, so `extract_op` had nothing to hand the kernel squad.
`mn.py fanout` removes that premise — a leg can now be sixteen processes across the pinned pair. Nothing
else about the two-leg contract moved: both legs still execute the same `leg_runner.py` and the same
`cases.py`, and which leg a process is is still decided ONLY by the overlay on `PYTHONPATH`.

Read this with the two measured cards, which are the state of the art on this topology and are what an
architect is expected to have already read:
`knowledge/learned/crossnode-tp-allreduce-gfx942.md` (what cannot engage, and why),
`knowledge/learned/crossnode-rccl-socket-tuning-gfx942.md` (the wire, and what the env already bought).

---

## 1. Admission — when a collective may become a kernel candidate

A collective is the most expensive candidate GEAK can spend a slot on: it needs both machines, it
forecloses the serving pool while it runs, and on this topology it is usually **already at the roof**.
All four conditions must hold, and each one is a number, not a judgement:

1. **The collective is genuinely hot.** From the trace, not from reputation. On the measured topology
   all-reduce is 66.45% of GPU time.
2. **Its time is WIRE time, not peer-wait.** Per-call skew ≈ 1.0–1.5× across ranks. The same kernel name
   is spin on one topology and wire time on another, and `roles/profiler.md`'s "route a collective as a
   comm-CONFIG lever, not a rewrite" is conditioned on `skew ≫ 3` for exactly this reason. **At high skew
   the collective is a symptom of imbalance elsewhere and optimizing it is optimizing a wait.**
3. **The env ladder is exhausted, and the baseline INCLUDES it.** `NCCL_SOCKET_NTHREADS=2
   NCCL_NSOCKS_PERTHREAD=2 NCCL_MIN_NCHANNELS=16` is worth 4–13× on the collective and +93% e2e on this
   fleet — it is banked, not hypothetical. A candidate measured against RCCL's *defaults* will report
   that env win as its own. **The baseline leg must carry the banked env; if it does not, the number is
   not a kernel result.**
4. **The wire roof is measured, and there is budget left.** Run the probe (`rccl-tests all_reduce_perf`,
   or `shell/ar_sweep.py` for a 16-rank `torch.distributed` probe that needs no server) at the SAME
   message sizes as the served ones, with the banked env set. `wire_budget = tuned_ms / probe_best_ms` at
   that size. The cross-node wire here is a shared 25 Gb/s management NIC, ≈3.1 GB/s, and at 8 MB the
   tuned collective is already sitting on it — **wire_budget ≈ 1.0 means zero headroom, and no kernel can
   create bandwidth.** Only two things help at the roof: move fewer bytes, or move them at a better time.

If 1–4 hold, the architect emits the candidate with `execution_model: "distributed_collective"` and an
`ENGAGEMENT_CHECK` that the integrator can run against a live server (see §6).

### Levers that are already falsified — do not spend a slot re-proving them

| lever | status on `nnodes > 1` |
|---|---|
| custom all-reduce, QuickReduce quantized AR, aiter `fused_allreduce_rmsnorm` | **self-disable**, and TP16 is outside `_SUPPORTED_WORLD_SIZES` anyway. Two independent gates. The banner is in every baseline `server.log`. |
| `--enable-quant-communications` | NPU-only; raises at startup on ROCm. |
| `SGLANG_ROCM_USE_MULTI_STREAM` / `TORCH_NCCL_HIGH_PRIORITY` overlap | measured **−5.10%**. There is nothing to overlap with when comm IS 66% of the time, and it is a no-op for dense models regardless. |
| `NCCL_ALGO=Tree` / `Ring` | Tree 0.83–1.00×; Ring is 10× *worse* at 128 KB. Leave the algorithm to RCCL. |
| `NTHREADS=4 NSOCKS=4`, `8×2`, `NCCL_BUFFSIZE=8MiB` | rank exits 1 during the first collective, on boxes with plenty of free VRAM. 2×2 is already at the knee. |

And the standing topology note, which is larger than anything in this file: **TP8 inside one xGMI island
with PP/DP across nodes avoids this wire entirely** (single-node AR sustains ~316–330 GB/s against
~3 GB/s here). Everything below is making the best of a fixed topology.

---

## 2. The task dir — what is different from a single-rank task

Same shape as `knowledge/shape_capture.md` describes, with these additions to `meta.json`:

```jsonc
{
  "execution_model": "distributed_collective",  // THE field. Absent or "single_rank" = the local path.
  "dist_backend": "nccl",                       // optional; rccl answers to "nccl" on ROCm
  "process_group": {                            // the FROZEN topology (see §3)
    "world_size": 16, "nnodes": 2, "ranks_per_node": 8,
    "tp": 16, "dp": 2, "group": "tp",
    "cluster_subnet": "10.235.200.", "transport": "socket/ens50f0",
    "banked_env": "NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=2 NCCL_MIN_NCHANNELS=16"
  },
  "wire_roof": { "probe": "ar_sweep", "gb_per_s": 3.1,
                 "per_size_ms": { "128KiB": 0.499, "8MiB": 4.649, "32MiB": 13.902 } }
}
```

`target_callable` is the collective entry the live stack actually calls (e.g. the tensor-parallel
all-reduce wrapper), resolved the same way as any other target: the baseline leg must resolve it OUTSIDE
the task dir, or the harness refuses to measure.

## 3. Freezing the process group, not just the shapes

A single-rank task freezes shapes and dtypes. A collective's inputs include **its topology**, and a
number measured on a different topology is not comparable — 16 ranks over 2 nodes and 16 ranks on one
node are different operators with the same name. So `process_group` above is part of the frozen contract
and must be captured from the live server, exactly as shapes are:

- `world_size` / `nnodes` / `ranks_per_node` — from the pin (`mn_state.json`) and the server's own log.
- which group — a TP16/DP2 server has more than one communicator; name the one the hot collective belongs
  to, or the candidate optimizes a group that is not the bottleneck.
- `transport` — from one `NCCL_DEBUG=INFO` boot. **Never inferred.** On this fleet the eight 400 Gb/s
  backend NICs are /31 point-to-point, 0 of 192 ping attempts reachable, so collectives fall to a shared
  25 Gb/s NIC. An architect who assumes the fast NIC is being used is off by 100×.
- `banked_env` — the env the baseline must carry (§1.3).

**If the pin moves, the frozen group is stale.** Re-derive it: `ip -o -4 addr` plus one ping matrix takes
under a minute, and `mn.py preflight` records the fabric it decided in `mn_preflight.json`.

## 4. `cases.py` — the two rules that are not obvious

The usual contract holds (`timing_cases(h, meta)`, `random_shapes(h, meta)`, `call(args)`), plus:

1. **`timing_cases` and `random_shapes` must be enumerable WITHOUT a process group.** The harness runs
   `--mode list` and `--mode resolve` locally on purpose: bucket labels and import identity are
   properties of the overlay, not of the topology, and making them need a 16-rank rendezvous would put
   the identity guard behind a live cluster. Build sig strings and shape descriptions with no collective
   and no `torch.distributed` call.
2. **`call(args)` runs on EVERY rank and must be symmetric.** It is entered by all sixteen processes with
   the same seed, so it must not contain a rank-conditional branch that changes the collective's shape.
   Per-rank data variation is fine and should come from the seeded generator; per-rank *control flow* is
   how a collective deadlocks with fifteen peers waiting on a barrier that one rank never reaches.

Bucket sizes should be the served message sizes, from the trace: 128 KB is the decode all-reduce, 32 MB
the prefill one. They behave differently enough (1.9× vs 10.8× from the same env change) that a single
bucket is not a measurement of the operator.

## 5. How a distributed leg is measured

```
measure_legs                     unchanged — ABBA interleaving, running median, undecided window
  └─ _run_leg(..., meta=meta)    THE dispatch, and the only place that knows a leg can span ranks
       ├─ resolve / list  -> local subprocess, one card, as always
       └─ time / oracle   -> mn.py fanout: 16 x (same leg_runner.py, same overlay, one card each)
```

What the fanout adds on top of the shared node facts (`multi_node/rank_preamble.sh.inc`, byte-identical
to what a server rank gets — if the two renderers ever disagreed about the NIC, the microbenchmark and
the serving A/B would be measuring different fabrics and nothing would fail while that was true):

- **the rank triple comes from the environment**, owned by `fanout` and unreachable from `EXTRA_ENV`. A
  swept `RANK` is two processes that both believe they are rank 0, which presents as a hang at
  rendezvous rather than as a bad flag.
- **one card per rank**, masked with `ROCR_VISIBLE_DEVICES`, so this rank's device is always `cuda:0`.
- **warm, then barrier, then time.** A candidate that JITs on first touch would otherwise have fifteen
  peers waiting inside the timed region and would report its own compile as wire time on all of them.
- **the reported time is the MAX over ranks.** A collective is not finished until its slowest rank is;
  each rank's own number is kept as `ms_this_rank` so skew stays auditable, but the operator's time is
  the worst rank's. Averaging them reports a time no rank experienced.

## 6. Correctness — the oracle is per rank

**Definition.** A candidate collective is correct when, for the same seed and the same frozen group, every
rank's output matches what RCCL produced on that rank, within `tol`.

The single-rank flow records only the baseline and calls the candidate in the unittest's own process.
A collective candidate cannot be called there at all — it needs its fifteen peers — so **both sides are
recorded under a fanout and compared file against file**:

```python
base = h.distributed_random_outputs(TASK, meta, "baseline",  seed=0, draws=3)
cand = h.distributed_random_outputs(TASK, meta, "candidate", seed=0, draws=3)
ok, rows = h.check_random_vs_baseline_per_rank(base, cand, tol)
```

All sixteen ranks are compared, not just rank0's. **Right on the root and wrong on a non-root rank is the
characteristic collective bug**, and reading the oracle only from rank0 is exactly how it survives the
gate. A rank or a draw present on one side only is a FAILURE, never a skip: a candidate that silently
produced nothing on half the ranks would otherwise compare zero cases and pass.

Each rank writes its own blob (`_baseline_random.pt.rank<N>`). Sixteen ranks writing one path on a shared
mount is a race whose loser supplies the golden.

The `ENGAGEMENT_CHECK` the architect attaches is an assertion about a LIVE server, evaluated by
`roles/e2e_integrator.md` with no new machinery — it is data, not code. For a collective the useful form
is a banner or counter that must appear on **all sixteen rank logs**, because the failure mode being
guarded against is engagement on rank0 only: e.g. `grep -c '<candidate marker>' server.rank*.log` must
report 16 non-zero counts. Zero on any rank ⇒ reject; the accepted-but-not-engaged case is what produced
a reported 1.416× on a comm lane that was actually closed.

## 7. What looks like a hang and is not

| symptom | what it actually is |
|---|---|
| `fanout` sits for minutes with no output | normal. 16 ranks × cold torch import over NFS; the per-rank logs in `OUT_DIR` are written live — read those, not the driver. |
| every rank waits at the first collective, forever | `NCCL_SOCKET_IFNAME` unset, so RCCL picked calico/docker0/veth and is blocking on an unroutable address. The preamble exports it from the discovered `CLUSTER_IF`; a bare `torchrun` started by hand has no such protection. |
| one rank exits 1 during the first collective, others hang | a socket setting past the knee (`4×4`, `8×2`, `BUFFSIZE=8MiB`). Not VRAM — reproduced on a box with 21.7 GiB free. |
| the leg times out and the NEXT leg reads as a slowdown | a leftover rank still holding a card. `_run_leg` hands `mn.py` the pattern that identifies them (`FANOUT_KILL_PATTERN`) precisely so this is cleaned up before the next spawn; if you invoke `fanout` by hand, pass it. |
| the same config measures 11 ms and 52 ms minutes apart | real. Run-to-run variance on this wire is ~5× because the transport choice is made once at communicator init, while within-run variance is a few percent. **Always repeat and print the spread** — `measure_legs`' ABBA pairing is what makes this survivable, and a single unpaired timing here is a coin flip. |
| the collective is 4–13× faster but e2e barely moves | check the Amdahl ceiling first: with 66.45% in the collective, the best possible e2e is `1/0.6645 = 1.50×` even if everything else went to zero. |
