# kernel-microbench: GPU peak microbenchmark

Question: what bandwidth, FP64 / FP32 rate and latency does one Rubin GPU deliver to simple kernels: DRAM, L2 and
shared-memory bandwidth, FMA throughput (whole GPU and one SM), cuBLAS DGEMM / SGEMM at 8192³, 3×3 block-sparse SpMV
in FP64 and FP32 (in DRAM and held in one block's shared memory), dot products, and launch, graph-node, host
round-trip and grid-sync latency. The case list and how bytes and flops are counted are in the header of
`gpu_peaks.cu`; every figure is the median of 11 timed repetitions after 3 warm-ups.

`run.sbatch` (for the Rubin Slurm cluster) asks for one GPU on any node but cn05 for at most 20 minutes. The nodes have
no gcc, so `gpu_peaks.cu` is built for `sm_107a` with ~/cuda-13.4 inside the Rubin image through enroot,
then run on the node.

On Rubin: clone this repository, `cd` into it and `sbatch run.sbatch`.
The log, with one JSON line per case at the end, is `~/kcoral-probe/gpu-peaks-<job>.log`; the build and `out.jsonl`
go to `~/kcoral-probe/gpu-peaks-<job>/`.
