# Cross-node TP all-reduce (gfx942 · sglang 0.5.19 · TP16 over 2×MI300X)

**Reuse key:** `(comm / all-reduce, gfx942, nnodes>1, any regime)`
**Confidence:** ★★★ — measured e2e on a pinned pair, plus source-level proof of every non-engagement.
**One-line:** when the TP process group spans nodes, **every byte-reducing all-reduce implementation in
sglang/aiter turns itself off**; the only candidate source left is the RCCL env table, and GEAK's router
does not currently send you there.

## What was measured

Qwen3-8B-FP8, TP16 (8+8 across two MI300X boxes), DP2, ISL 512 / OSL 32 / conc 16, warm_server.

| fact | value |
|---|---|
| all-reduce share of GPU time | **66.45%** (`ncclDevKernel_Generic_4`, top row of the profile) |
| per-call skew (mean/median) | **1.1× — uniform** |
| baseline | 213.586 tok/s |

**The skew is the load-bearing number.** The standard doctrine de-inflates collectives because their long
tail is peer-wait spin. At 1.1× there is no tail: this is **genuine wire time**, so de-inflation does not
apply and the raw 66.45% is the real cost. Check the skew before you decide which it is — the same kernel
name is spin on one topology and wire time on another.

## The three headline levers are all unavailable here — proven, not guessed

`allreduce/overview.md` advertises custom AR, QuickReduce quantized AR, and AR+RMSNorm fusion. All three
are **built on top of custom all-reduce**, which requires peers to read each other's HBM directly. That
is a same-node property, so the framework disables the whole family cross-node:

| lever | why it cannot engage | evidence |
|---|---|---|
| QuickReduce quantized AR (`ROCM_QUICK_REDUCE_QUANTIZATION=FP8/INT6/INT4`) | self-disables cross-node; quant level is read only **after** the communicator is already off, so the env binds to nothing | banner on **all 16 TP ranks**: `Custom quick allreduce is disabled because this process group spans across nodes.` (`quick_all_reduce.py:113`) |
| custom AR / aiter custom AR | same self-disable | `[aiter] Custom allreduce is disabled because this process group spans across nodes.` |
| aiter `fused_allreduce_rmsnorm[_quant]` | returns `None` unless `ca_comm` is live, and `ca_comm` is off per the row above; the `_quant` variant additionally hard-requires `is_gfx95_supported()` — this is gfx942 | source + banner |
| **world size** (independent second gate) | `QuickAllReduce._SUPPORTED_WORLD_SIZES = [2, 4, 8]`, `CustomAllreduce._SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]` — **TP16 is in neither** | `quick_all_reduce.py:54`, `custom_all_reduce.py:41` |
| `--enable-quant-communications` | **NPU-only.** Raises at startup on ROCm | `arg_groups/validation_hook.py`: `if cfg.enable_quant_communications and cfg.device != "npu": raise ValueError(...)` |

Two independent gates (cross-node, and world size) each individually exclude TP16. Do not spend a boot
reproducing these — read the baseline `server.log` for the banner instead; it is already there.

## What DOES remain, and what a round actually spent its budget on

The one engageable, parity-safe combination tried was
`SGLANG_ROCM_USE_MULTI_STREAM=1 TORCH_NCCL_HIGH_PRIORITY=1 GPU_MAX_HW_QUEUES=2` (comm/compute overlap +
torch-level scheduling). Measured **202.687 vs 213.586 = −5.10%**, a regression far outside the 0.5% noise
band. Two reasons, both worth carrying forward:

- **There is nothing to overlap with.** Overlap hides comm under independent compute; when comm *is* 66%
  of the time, the compute to hide it under does not exist, and raising collective stream priority just
  adds contention on the critical path.
- `SGLANG_ROCM_USE_MULTI_STREAM` is a **no-op for dense models** — it is only consumed on MoE/deepseek paths.

**The real gap in that round: not one RCCL knob was set.** The live ranks carried only
`NCCL_HIGH_PRIORITY=1` and an *empty* `NCCL_SOCKET_IFNAME` — nothing chose the algorithm, the protocol,
the channel count, or even the NIC. `NCCL_DEBUG` was off, so which transport carried the 66% was never
observed.

## Rule for the next cross-node round

1. **Observe the transport FIRST.** `NCCL_DEBUG=INFO` + `RCCL_DEBUG_SUBSYS=INIT,GRAPH` for one boot, and
   read which net/algo/protocol/channel count RCCL chose. Tuning a 66% cost without knowing its transport
   is guesswork. On a cluster whose fast NICs are point-to-point and do not route, collectives fall back
   to a management NIC and *that*, not software, is the roof — `rccl-tests all_reduce_perf` gives the
   number in a minute.
2. **Take candidates from the RCCL env table**, not from the all-reduce overview's backend landscape. The
   table is `perf_knowledge/backends/mori_rccl/rccl_tuning.md` and it carries explicit multi-node rows:
   `NCCL_NET_GDR_LEVEL=2`, `NCCL_IGNORE_CPU_AFFINITY=1`, `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` (RoCE),
   `NCCL_MIN_NCHANNELS`, `RCCL_MSCCLPP_THRESHOLD` (push more messages onto MSCCL++ fast kernels). For a
   socket fallback specifically, `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` and `NCCL_SOCKET_IFNAME`
   are the bandwidth knobs.
3. **Do not propose the single-node three.** Custom AR, QuickReduce and AR-norm fusion are off the table
   the moment `nnodes > 1`; proposing them burns a candidate slot on a guaranteed non-engagement.
4. **Know the Amdahl ceiling before promising anything.** With 66.45% untouchable, the best possible e2e
   result — every other kernel reduced to zero — is `1/0.6645 = 1.50×`. If the target is above that, the
   answer is a topology change, not a kernel.

## The topology note the cards already contained

`rccl_tuning.md` says it outright: **"Keep TP within one xGMI island (≤8); use PP across nodes."** TP16
spanning two boxes puts two all-reduces per layer on the slow wire and simultaneously disqualifies every
accelerated implementation. Folding TP back to 8 (with pipeline or data parallel across nodes) re-enables
custom AR, QuickReduce quantization and xGMI P2P all at once — a single-node AR sustains ~316–330 GB/s.
That is the lever with real headroom; everything in this card is making the best of a fixed topology.
