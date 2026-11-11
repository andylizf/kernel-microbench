// GPU peak microbenchmark: DRAM / L2 / shared-memory bandwidth, FP64 and FP32 throughput, block-sparse SpMV, launch latency.
// Build: nvcc -O3 -std=c++17 -arch=sm_XX -o gpu_peaks gpu_peaks.cu -lcublas      Run: ./gpu_peaks > out.jsonl
// One JSON line per case: {"case":..., "prec":..., "n":..., "ms":..., plus rate fields}. Every timing is the median of
// REPS timed repetitions (cudaEvent) after WARM untimed ones.
//
// Cases
//   dram_copy / dram_read   y = x (64-bit words, no arithmetic) over 2^28 words (2 GiB per array) / sum x*y (fp64):
//                           DRAM bandwidth, bytes counted once per element read or written.
//   dram_axpy               y = a*x + y in fp64 over the same arrays (one FMA per 24 bytes).
//   l2_copy / l2_axpy       the same two over 4 / 8 / 16 MiB per array, 200 passes in one launch: L2-resident.
//   fma                     independent FMA chains in registers, whole GPU (fp64, fp32): peak flops, 2 per FMA.
//   fma_1blk                the same kernel with 1 block of 256 threads (fp64, fp32): one SM's rate.
//   smem_1blk               one block streaming a 32 KiB shared-memory buffer (64-bit loads, XOR): one SM's shared bandwidth.
//   dgemm / sgemm           cuBLAS 8192^3 (DGEMM: FP64 tensor cores where the GPU has them; SGEMM: default math mode).
//   spmv_bsr3               3x3 block-sparse y = A x, NB blocks per block row (random columns within +-64 block rows),
//                           one thread per scalar row; matrix values and vectors in the case's precision, int32 columns.
//                           rows R = 1e3 .. 3e6 block rows; bytes = blocks*(9*s + 4) + 3R*s*(NB+1) (x gathered per
//                           block, counted once per use) and flops = 18 per block.
//   spmv_bsr3_1blk          the same matrix held in one block's shared memory (R = 256 block rows), one block of 768
//                           threads, repeated 100 times per launch: one SM's SpMV rate on shared-memory data.
//   dot                     sum x*y (fp64, fp32), n = 1e3 .. 1e8: two-pass block reduction.
//   launch_stream           empty kernel, back-to-back stream launches: time per launch.
//   launch_graph            empty kernel x 1000 nodes in one CUDA graph: time per node.
//   sync_roundtrip          empty kernel + cudaStreamSynchronize: one host round trip.
//   d2h_read                4-byte cudaMemcpy device-to-host after a kernel (the host read of a flag).
//   grid_sync               cooperative kernel, all SMs x 1 block, k grid.sync() calls: time per sync.
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace cg = cooperative_groups;
#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

static const int WARM = 3, REPS = 11;

template <class F>
double time_ms(F f) {  // median of REPS cudaEvent timings of f()
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < WARM; i++) f();
  CK(cudaDeviceSynchronize());
  std::vector<double> v;
  for (int i = 0; i < REPS; i++) {
    CK(cudaEventRecord(a));
    f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return v[v.size() / 2];
}

static void emit(const char* c, const char* p, double n, double ms, const char* k1, double v1, const char* k2 = nullptr,
                 double v2 = 0) {
  printf("{\"case\":\"%s\",\"prec\":\"%s\",\"n\":%.0f,\"ms\":%.6g,\"%s\":%.6g", c, p, n, ms, k1, v1);
  if (k2) printf(",\"%s\":%.6g", k2, v2);
  printf("}\n");
  fflush(stdout);
}

template <class T>
__global__ void axpy(int64_t n, T a, const T* __restrict__ x, T* __restrict__ y) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
}

__global__ void copy_rep(int64_t n, int reps, const unsigned long long* __restrict__ x, unsigned long long* __restrict__ y) {
  for (int r = 0; r < reps; r++)
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
      y[i] = x[i] ^ (unsigned long long)r;
}

template <class T>
__global__ void axpy_rep(int64_t n, int reps, T a, const T* __restrict__ x, T* __restrict__ y) {  // y re-read each pass
  for (int r = 0; r < reps; r++)
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
      y[i] = a * x[i] + y[i];
}

template <class T>
__global__ void reduce_dot(int64_t n, const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ part) {
  __shared__ T s[1024];
  T acc = 0;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    acc += x[i] * y[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}

template <class T>
__global__ void reduce_final(int n, T* part) {  // one block
  __shared__ T s[1024];
  T acc = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += part[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[0] = s[0];
}

template <class T>
__global__ void fma_kernel(int iters, T* out, T seed) {
  T a0 = seed + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const T b = (T)0.999999, c = (T)1e-7;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
      a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
  }
  T r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (r == (T)-12345) out[0] = r;  // never true; keeps the chains live
}

template <class T>
__global__ void smem_kernel(int iters, T* out) {  // integer XOR of the loaded words: no floating-point arithmetic
  extern __shared__ unsigned char sm_raw[];
  T* s = reinterpret_cast<T*>(sm_raw);
  const int n = 32768 / sizeof(T);
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = (T)i;
  __syncthreads();
  T acc = 0;
  for (int it = 0; it < iters; it++)
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc ^= s[i];
  if (acc == (T)12345) out[0] = acc;
}

// 3x3 BSR SpMV, one thread per scalar row; values row-major per block
template <class T>
__global__ void spmv_bsr3(int R, const int* __restrict__ off, const int* __restrict__ col, const T* __restrict__ val,
                          const T* __restrict__ x, T* __restrict__ y) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= 3 * R) return;
  int br = r / 3, k = r % 3;
  T acc = 0;
  for (int p = off[br]; p < off[br + 1]; p++) {
    const T* v = val + 9 * (int64_t)p + 3 * k;
    int c = 3 * col[p];
    acc += v[0] * x[c] + v[1] * x[c + 1] + v[2] * x[c + 2];
  }
  y[r] = acc;
}

// the same with the whole matrix and both vectors in shared memory, one block, `reps` products per launch
template <class T>
__global__ void spmv_bsr3_smem(int R, int nnzb, const int* __restrict__ off_g, const int* __restrict__ col_g,
                               const T* __restrict__ val_g, const T* __restrict__ x_g, T* __restrict__ y_g, int reps) {
  extern __shared__ unsigned char sm_raw[];
  T* val = reinterpret_cast<T*>(sm_raw);
  T* x = val + 9 * nnzb;
  T* y = x + 3 * R;
  int* off = reinterpret_cast<int*>(y + 3 * R);
  int* col = off + R + 1;
  for (int i = threadIdx.x; i < 9 * nnzb; i += blockDim.x) val[i] = val_g[i];
  for (int i = threadIdx.x; i < 3 * R; i += blockDim.x) x[i] = x_g[i];
  for (int i = threadIdx.x; i <= R; i += blockDim.x) off[i] = off_g[i];
  for (int i = threadIdx.x; i < nnzb; i += blockDim.x) col[i] = col_g[i];
  __syncthreads();
  for (int it = 0; it < reps; it++) {
    for (int r = threadIdx.x; r < 3 * R; r += blockDim.x) {
      int br = r / 3, k = r % 3;
      T acc = 0;
      for (int p = off[br]; p < off[br + 1]; p++) {
        const T* v = val + 9 * p + 3 * k;
        int c = 3 * col[p];
        acc += v[0] * x[c] + v[1] * x[c + 1] + v[2] * x[c + 2];
      }
      y[r] = acc;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < 3 * R; r += blockDim.x) x[r] = y[r] * (T)0.5;  // next product uses the result
    __syncthreads();
  }
  for (int i = threadIdx.x; i < 3 * R; i += blockDim.x) y_g[i] = y[i];
}

__global__ void empty_kernel() {}

__global__ void grid_sync_kernel(int k, int* out) {
  cg::grid_group g = cg::this_grid();
  for (int i = 0; i < k; i++) g.sync();
  if (g.thread_rank() == 0) out[0] = k;
}

struct Bsr {
  int R, nnzb;
  std::vector<int> off, col;
};
static Bsr make_bsr(int R, int NB, unsigned seed) {
  Bsr m;
  m.R = R;
  std::mt19937 g(seed);
  m.off.push_back(0);
  for (int r = 0; r < R; r++) {
    std::vector<int> cs{r};
    while ((int)cs.size() < NB) {
      int c = r + (int)(g() % 129) - 64;
      if (c < 0 || c >= R || std::find(cs.begin(), cs.end(), c) != cs.end()) {
        if (R <= NB) break;
        continue;
      }
      cs.push_back(c);
    }
    std::sort(cs.begin(), cs.end());
    m.col.insert(m.col.end(), cs.begin(), cs.end());
    m.off.push_back((int)m.col.size());
  }
  m.nnzb = (int)m.col.size();
  return m;
}

template <class T>
static void bench_spmv(const char* p, int NB) {
  for (int R : {1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000}) {
    Bsr m = make_bsr(R, NB, 7);
    int *doff, *dcol;
    T *dval, *dx, *dy;
    CK(cudaMalloc(&doff, (R + 1) * sizeof(int)));
    CK(cudaMalloc(&dcol, m.nnzb * sizeof(int)));
    CK(cudaMalloc(&dval, 9ll * m.nnzb * sizeof(T)));
    CK(cudaMalloc(&dx, 3ll * R * sizeof(T)));
    CK(cudaMalloc(&dy, 3ll * R * sizeof(T)));
    CK(cudaMemcpy(doff, m.off.data(), (R + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcol, m.col.data(), m.nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(dval, 0, 9ll * m.nnzb * sizeof(T)));
    CK(cudaMemset(dx, 0, 3ll * R * sizeof(T)));
    int thr = 256, blk = (3 * R + thr - 1) / thr;
    const int inner = 20;  // launches per timed repetition
    double ms = time_ms([&] {
                  for (int i = 0; i < inner; i++) spmv_bsr3<T><<<blk, thr>>>(R, doff, dcol, dval, dx, dy);
                }) / inner;
    CK(cudaGetLastError());
    double s = sizeof(T);
    double bytes = m.nnzb * (9 * s + 4) + 3.0 * R * s * (NB + 1) + 4.0 * (R + 1);
    double flops = 18.0 * m.nnzb;
    printf("{\"case\":\"spmv_bsr3\",\"prec\":\"%s\",\"n\":%d,\"nnzb\":%d,\"ms\":%.6g,\"GBps\":%.6g,\"GFLOPs\":%.6g}\n", p,
           R, m.nnzb, ms, bytes / ms / 1e6, flops / ms / 1e6);
    fflush(stdout);
    CK(cudaFree(doff)); CK(cudaFree(dcol)); CK(cudaFree(dval)); CK(cudaFree(dx)); CK(cudaFree(dy));
  }
  // one block, shared-memory resident (matrix held on chip)
  int R = 256;
  Bsr m = make_bsr(R, NB, 7);
  size_t smb = 9ull * m.nnzb * sizeof(T) + 6ull * R * sizeof(T) + (R + 1 + m.nnzb) * sizeof(int);
  int *doff, *dcol;
  T *dval, *dx, *dy;
  CK(cudaMalloc(&doff, (R + 1) * sizeof(int)));
  CK(cudaMalloc(&dcol, m.nnzb * sizeof(int)));
  CK(cudaMalloc(&dval, 9ll * m.nnzb * sizeof(T)));
  CK(cudaMalloc(&dx, 3ll * R * sizeof(T)));
  CK(cudaMalloc(&dy, 3ll * R * sizeof(T)));
  CK(cudaMemcpy(doff, m.off.data(), (R + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcol, m.col.data(), m.nnzb * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(dval, 0, 9ll * m.nnzb * sizeof(T)));
  CK(cudaMemset(dx, 0, 3ll * R * sizeof(T)));
  int dev = 0, optin = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  if (smb > (size_t)optin) {  // A100: 164 KB per block, below this case's layout (E77)
    printf("{\"case\":\"spmv_bsr3_1blk\",\"prec\":\"%s\",\"skipped\":\"smem %zu B > opt-in %d B\"}\n", p, smb, optin);
    CK(cudaFree(doff)); CK(cudaFree(dcol)); CK(cudaFree(dval)); CK(cudaFree(dx)); CK(cudaFree(dy));
    return;
  }
  CK(cudaFuncSetAttribute(spmv_bsr3_smem<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
  const int reps = 100;
  double ms = time_ms([&] { spmv_bsr3_smem<T><<<1, 768, smb>>>(R, m.nnzb, doff, dcol, dval, dx, dy, reps); });
  CK(cudaGetLastError());
  double flops = 18.0 * m.nnzb * reps;
  printf("{\"case\":\"spmv_bsr3_1blk\",\"prec\":\"%s\",\"n\":%d,\"nnzb\":%d,\"smem_B\":%zu,\"ms\":%.6g,\"us_per_spmv\":%.6g,"
         "\"GFLOPs\":%.6g}\n", p, R, m.nnzb, smb, ms, ms * 1e3 / reps, flops / ms / 1e6);
  fflush(stdout);
  CK(cudaFree(doff)); CK(cudaFree(dcol)); CK(cudaFree(dval)); CK(cudaFree(dx)); CK(cudaFree(dy));
}

template <class T>
static void bench_dot(const char* p, T* x, T* y, T* part) {
  for (int64_t n : {1000ll, 10000ll, 100000ll, 1000000ll, 10000000ll, 100000000ll}) {
    int blk = (int)std::min<int64_t>(1024, (n + 255) / 256);
    double ms = time_ms([&] {
      reduce_dot<T><<<blk, 256>>>(n, x, y, part);
      reduce_final<T><<<1, 256>>>(blk, part);
    });
    CK(cudaGetLastError());
    emit("dot", p, (double)n, ms, "GBps", 2.0 * n * sizeof(T) / ms / 1e6);
  }
}

int main() {
  int dev = 0, sms;
  CK(cudaSetDevice(dev));
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, dev));
  sms = pr.multiProcessorCount;
  int clk = 0, memclk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);
  printf("{\"case\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_MB\":%.1f,\"smem_per_block_optin_KB\":%.1f,"
         "\"mem_GB\":%.1f,\"clock_kHz\":%d,\"mem_clock_kHz\":%d}\n",
         pr.name, pr.major, pr.minor, sms, pr.l2CacheSize / 1048576.0, pr.sharedMemPerBlockOptin / 1024.0,
         pr.totalGlobalMem / 1e9, clk, memclk);
  fflush(stdout);

  // DRAM and L2 bandwidth
  const int64_t NB_ = 1ll << 28;  // doubles
  double *x, *y;
  CK(cudaMalloc(&x, NB_ * 8));
  CK(cudaMalloc(&y, NB_ * 8));
  CK(cudaMemset(x, 0, NB_ * 8));
  CK(cudaMemset(y, 0, NB_ * 8));
  {
    double ms = time_ms([&] { copy_rep<<<sms * 8, 512>>>(NB_, 1, (unsigned long long*)x, (unsigned long long*)y); });
    emit("dram_copy", "u64", (double)NB_, ms, "GBps", 2.0 * NB_ * 8 / ms / 1e6);
    ms = time_ms([&] { axpy<double><<<sms * 8, 512>>>(NB_, 1.0001, x, y); });
    emit("dram_axpy", "fp64", (double)NB_, ms, "GBps", 3.0 * NB_ * 8 / ms / 1e6);
    double* part;
    CK(cudaMalloc(&part, 4096 * 8));
    ms = time_ms([&] {
      reduce_dot<double><<<1024, 1024>>>(NB_, x, y, part);
      reduce_final<double><<<1, 1024>>>(1024, part);
    });
    emit("dram_read", "fp64", (double)NB_, ms, "GBps", 2.0 * NB_ * 8 / ms / 1e6);  // x and y read once
    for (int mb : {4, 8, 16}) {  // per array; two arrays resident in L2
      const int64_t nl = ((int64_t)mb << 20) / 8;
      const int reps = 200;
      ms = time_ms([&] { copy_rep<<<sms * 4, 512>>>(nl, reps, (unsigned long long*)x, (unsigned long long*)y); }) / reps;
      emit("l2_copy", "u64", (double)nl, ms, "GBps", 2.0 * nl * 8 / ms / 1e6);
      ms = time_ms([&] { axpy_rep<double><<<sms * 4, 512>>>(nl, reps, 1.0001, x, y); }) / reps;
      emit("l2_axpy", "fp64", (double)nl, ms, "GBps", 3.0 * nl * 8 / ms / 1e6);
    }
    // dot at scene sizes, fp64 and fp32 (fp32 views of the same buffers)
    bench_dot<double>("fp64", x, y, part);
    bench_dot<float>("fp32", (float*)x, (float*)y, (float*)part);
    CK(cudaFree(part));
  }
  // flops
  {
    int iters = 4096;
    double* o;
    CK(cudaMalloc(&o, 64));
    int blocks = sms * 8;
    double ms = time_ms([&] { fma_kernel<double><<<blocks, 256>>>(iters, o, 1.0); });
    double fl = 2.0 * 8 * 16 * (double)iters * blocks * 256;
    emit("fma", "fp64", blocks * 256.0, ms, "TFLOPs", fl / ms / 1e9);
    ms = time_ms([&] { fma_kernel<float><<<blocks, 256>>>(iters, (float*)o, 1.0f); });
    emit("fma", "fp32", blocks * 256.0, ms, "TFLOPs", fl / ms / 1e9);
    int it1 = 1024;
    ms = time_ms([&] { fma_kernel<double><<<1, 256>>>(it1, o, 1.0); });
    double fl1 = 2.0 * 8 * 16 * (double)it1 * 256;
    emit("fma_1blk", "fp64", 256, ms, "GFLOPs", fl1 / ms / 1e6);
    ms = time_ms([&] { fma_kernel<float><<<1, 256>>>(it1, (float*)o, 1.0f); });
    emit("fma_1blk", "fp32", 256, ms, "GFLOPs", fl1 / ms / 1e6);
    ms = time_ms([&] { fma_kernel<double><<<1, 1024>>>(it1, o, 1.0); });
    emit("fma_1blk1024", "fp64", 1024, ms, "GFLOPs", 4 * fl1 / ms / 1e6);
    ms = time_ms([&] { fma_kernel<float><<<1, 1024>>>(it1, (float*)o, 1.0f); });
    emit("fma_1blk1024", "fp32", 1024, ms, "GFLOPs", 4 * fl1 / ms / 1e6);
    int its = 2000;
    CK(cudaFuncSetAttribute(smem_kernel<unsigned long long>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768));
    ms = time_ms([&] { smem_kernel<unsigned long long><<<1, 1024, 32768>>>(its, (unsigned long long*)o); });
    emit("smem_1blk", "u64", 32768, ms, "GBps", 32768.0 * its / ms / 1e6);
    CK(cudaFree(o));
  }
  // cuBLAS GEMM
  {
    cublasHandle_t h;
    cublasCreate(&h);
    int n = 8192;
    double *A = x, *B = x + (int64_t)n * n, *C = y;
    double one = 1, zero = 0;
    double ms = time_ms([&] { cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, B, n, &zero, C, n); });
    emit("dgemm", "fp64", n, ms, "TFLOPs", 2.0 * n * n * (double)n / ms / 1e9);
    float onef = 1, zerof = 0;
    ms = time_ms([&] {
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &onef, (float*)A, n, (float*)B, n, &zerof, (float*)C, n);
    });
    emit("sgemm", "fp32", n, ms, "TFLOPs", 2.0 * n * n * (double)n / ms / 1e9);
    cublasDestroy(h);
  }
  CK(cudaFree(x));
  CK(cudaFree(y));
  // SpMV (11 blocks per block row: a soft-body row's neighbours, the bench's matrices have ~11 per row, E54 / sizeprobe)
  bench_spmv<double>("fp64", 11);
  bench_spmv<float>("fp32", 11);
  // latencies
  {
    const int K = 1000;
    double ms = time_ms([&] {
                  for (int i = 0; i < K; i++) empty_kernel<<<1, 32>>>();
                }) / K;
    emit("launch_stream", "-", K, ms, "us", ms * 1e3);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < K; i++) empty_kernel<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }) / K;
    emit("launch_graph", "-", K, ms, "us", ms * 1e3);
    // host round trip: wall clock over K (launch + sync)
    auto wall = [&](auto f) {
      std::vector<double> v;
      for (int r = 0; r < REPS; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; i++) f();
        auto t1 = std::chrono::steady_clock::now();
        v.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / 200);
      }
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    ms = wall([&] {
      empty_kernel<<<1, 32>>>();
      CK(cudaStreamSynchronize(0));
    });
    emit("sync_roundtrip", "-", 200, ms, "us", ms * 1e3);
    int* d;
    int hbuf;
    CK(cudaMalloc(&d, 64));
    ms = wall([&] {
      empty_kernel<<<1, 32>>>();
      CK(cudaMemcpy(&hbuf, d, 4, cudaMemcpyDeviceToHost));
    });
    emit("d2h_read", "-", 200, ms, "us", ms * 1e3);
    int k1 = 1, k2 = 1001;
    void* a1[] = {&k1, &d};
    void* a2[] = {&k2, &d};
    double t1 = time_ms([&] { CK(cudaLaunchCooperativeKernel((void*)grid_sync_kernel, sms, 256, a1, 0, 0)); });
    double t2 = time_ms([&] { CK(cudaLaunchCooperativeKernel((void*)grid_sync_kernel, sms, 256, a2, 0, 0)); });
    emit("grid_sync", "-", sms, (t2 - t1) / 1000, "us", (t2 - t1) / 1000 * 1e3);
    CK(cudaFree(d));
  }
  return 0;
}
