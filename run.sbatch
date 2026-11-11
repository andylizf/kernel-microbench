#!/bin/bash
# GPU peak microbenchmark on one Rubin GPU: bandwidth (DRAM, L2, shared memory), FP64 / FP32 throughput, cuBLAS
# DGEMM / SGEMM, 3x3 block-sparse SpMV, dot, launch and sync latency. Submit with
# sbatch run.sbatch from a clone of this repository. Builds and logs under ~/kcoral-probe.
# The nodes have no gcc, so the build happens in the Rubin image through enroot (nvidia hook skipped); the binary then
# runs on the node with ~/cuda-13.4's libraries. cn05 is excluded: its failed GPU 2 makes cuInit return 999 for every
# job on that node.
#SBATCH --job-name=kernel-microbench
#SBATCH --exclude=vr-r38-cn05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/vllm-yongye/kcoral-probe/gpu-peaks-%j.log
set -uo pipefail
IMAGE=$HOME/rubin-images/rubin-kernel-cu134-gpud.sqsh
WORK=$HOME/kcoral-probe/gpu-peaks-$SLURM_JOB_ID
C=$HOME/cuda-13.4
mkdir -p "$WORK" && cp "$SLURM_SUBMIT_DIR/gpu_peaks.cu" "$WORK/"
echo "host $(hostname) slurm CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,clocks.max.sm,memory.total --format=csv
export ENROOT_RUNTIME_PATH="$WORK/enroot" ENROOT_REMAP_ROOT=n
mkdir -p "$ENROOT_RUNTIME_PATH"
unset ${!PMIX_@}
enroot start --env NVIDIA_VISIBLE_DEVICES=void --mount "$HOME:$HOME" "$IMAGE" bash -c '
  cd "$1" && "$2/bin/nvcc" -O3 -std=c++17 -arch=sm_107a gpu_peaks.cu -o gpu_peaks -L"$2/lib64" -lcublas' bash "$WORK" "$C" \
  || { echo "build failed"; exit 1; }
LD_LIBRARY_PATH=$C/lib64:${LD_LIBRARY_PATH:-} "$WORK/gpu_peaks" > "$WORK/out.jsonl"
echo "run exit=$?"
cat "$WORK/out.jsonl"
