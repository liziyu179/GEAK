# PD disaggregation over TCP mooncake — where the time actually goes (gfx942, CCS AUS fleet, sglang 0.5.19)

**Reuse key:** `(pd_disagg, gfx942, sglang, no-RDMA fleet)` · **Confidence: ★★** (4 measured legs, one
session, spreads called out per leg)

Applies when prefill and decode are SEPARATE single-node servers joined by a load balancer, and the KV
pages cross a network — i.e. `--disaggregation-mode prefill|decode` with
`--disaggregation-transfer-backend mooncake_tcp`. It does NOT apply to a TP group spanning nodes; that is
a different topology with a different bottleneck (see `crossnode-tp-allreduce-gfx942.md`).

## The headline, so nobody re-derives it

On this fleet, 1P1D across two boxes is **2.3× SLOWER than not disaggregating at all**:
**96.0 tok/s** vs **219.4 tok/s** for TP16-spanning on the identical model and workload
(Qwen3-8B-FP8, ISL512 / OSL32 / conc16 / 32 prompts). Disaggregation is not a free win here. If the
question is "fastest on two boxes", the answer measured so far is: don't disaggregate.

## What the bottleneck is NOT (each of these cost a server round)

Read the decode engine's own batch line before theorising. It carries `#transfer-req`, and it said:

    Decode batch, #running-req: 1, ..., #transfer-req: 15, ..., gen throughput (token/s): 75.34

**15 of 16 in-flight requests are parked in KV transfer while the GPU decodes one.** So the transfer is
where the time is. The trap is assuming that means bandwidth. It does not:

| leg | change | tok/s | spread | TTFT ms |
|---|---|---|---|---|
| base | none | 96.0 | 1.95% | 3932.6 |
| kvfp8 | `--kv-cache-dtype fp8_e4m3` (halves the bytes) | 93.1 | 4.06% | **3940.0** |
| pool | `MC_TCP_ENABLE_CONNECTION_POOL=1` | 110.3 | 13.5% | 3772.7 |
| page64 | `--page-size 64` + pool | 118.7 | 36.5% | 3651.4 |

**Halving the bytes on the wire moved TTFT by +0.2%.** Whatever costs ~245 ms per request does not care
how much is being sent, which rules out the wire width — and rules it out cheaply, in one leg. Note also
that the two "wins" carry 13.5% and 36.5% round spread against a 1.95% baseline: on this rig a +23%
median with 36% spread is a hypothesis, not a result. Repeat before banking.

Estimated from the numbers above, the KV channel runs at ~308 MB/s ≈ **2.5 Gb/s on a 25 Gb/s NIC**, i.e.
~10% of line rate. Unlike the RCCL case on this same fleet, the wall is NOT the wire — there is an order
of magnitude of headroom, and it is being eaten by per-transfer fixed cost.

## Levers, in the order the evidence supports

1. `MC_TCP_ENABLE_CONNECTION_POOL=1` — stop paying TCP setup per transfer. Measured +14.9% (noisy).
2. `--page-size 64` (or 32) — sglang defaults to `page_size: 1`, one token per page, so a 512-token
   request is 512 × 36 layers ≈ **18k non-contiguous regions** for the transfer engine to move. Page 64
   cuts that ~64×. Measured +23.6% (very noisy) on top of the pool.
3. `--kv-cache-dtype fp8_e4m3` — **measured no-win here** (−3%, inside spread). Keep it only if the
   bytes ever become the constraint; it is lossy and buys nothing while the cost is per-piece.

Still unexplained after those three: TTFT stays pinned at 3.65–3.94 s across every transfer-side lever
tried. A quantity that refuses to move under bytes, connections AND slice count is probably not the
transfer at all but a fixed wait (handshake timeout, poll interval, retry backoff). Next probe should be
`MC_TE_METRIC=1 MC_TE_METRIC_INTERVAL_SECONDS=5` (the transfer engine reports its own bandwidth, turning
this from arithmetic into measurement), and a conc=1 leg — if TTFT is still ~3.7 s at conc=1 it is a
fixed per-request wait and no amount of parallelism will touch it.

## Environment facts (checked, not assumed)

* **No RDMA on this fleet**: no `/dev/infiniband`. `mooncake_tcp` is the only viable backend. sglang's
  `pd_disaggregation_hook.py` rewrites the CLI value `mooncake_tcp` into `mooncake` +
  `MC_FORCE_TCP=1`, so the server_args you see logged say `mooncake` — that is correct, not a fallback.
* The full knob set the transfer engine reads is discoverable with
  `strings <mooncake>/*.so | grep -oE '\bMC_[A-Z0-9_]+'`. TCP-relevant: `MC_TCP_ENABLE_CONNECTION_POOL`,
  `MC_TCP_SLICE_SIZE`, `MC_SLICE_SIZE`, `MC_WORKERS_PER_CTX`, `MC_TRANSFER_TIMEOUT`,
  `MC_HANDSHAKE_CONNECT_TIMEOUT`, `MC_RPC_TIMEOUT_MS`, `MC_TE_METRIC*`.
* `E transfer_metadata.cpp:912] Local segment descriptor not found` appears once per engine at startup on
  a healthy run. It is noise; do not chase it.

## Trap: the load balancer renames itself

`python -m sglang_router.launch_router` overwrites its own argv with **`sglang::router`** shortly after
start. Anything that looks for it by the module path on the launch line (`pgrep -f
sglang_router.launch_router`) gets **zero** while the process is alive and answering `/health` with 200.
This cost a full run: a supervisor's liveness probe reported the front end dead and tore the pair down
mid-boot. Probe the LB by its **client port**, not its name; if you must match a name to kill it, match
`sglang::router`, and keep it distinct from `sglang::scheduler` so LB teardown does not reap the engines.
