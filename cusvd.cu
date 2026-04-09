#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cusolverDn.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Error checking macro for cuSOLVER calls
#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error: " << status << " at "                \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Error checking macro for cuRAND calls
#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t status = call;                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            std::cerr << "cuRAND error: " << status << " at "                  \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

//references
//https://github.com/accelerated-computing-class/lab6
//https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvd/cusolver_gesvd_example.cu
//https://github.com/ROCm/rocm-examples/tree/f9d4e5e78325c36b319d91ec37c6410b2b6e12fb/Libraries/hipSOLVER/gesvd

constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }


template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    int k=0;
    while (elapsed_ms < target_time_ms || k<2) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
        k++;
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_in;
    bool compute_singular_vectors = false;
};

struct TestData {
    std::map<int32_t, float*> input;
    std::map<int32_t, float*> singvals;
};



enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    BenchmarkConfig const &config) {
    auto size_in = config.size_in;
    bool const sing_vectors = config.compute_singular_vectors;

    printf("  %6d  %3s%s", size_in, sing_vectors ? "yes" : "no",
           phase == Phase::BENCHMARK ? " " : " \n");
 
    curandGenerator_t curandGen;
    CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 12345ULL));

    float *a_gpu;
    float *svdout;
    float *u_gpu = nullptr;
    float *vt_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_in * size_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&svdout, size_in * sizeof(float)));
    if (sing_vectors) {
        CUDA_CHECK(cudaMalloc(&u_gpu, size_in * size_in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&vt_gpu, size_in * size_in * sizeof(float)));
    }
    
    cusolverDnHandle_t cusolverH = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize( cusolverH,  size_in,    size_in,   &lwork   ));
    float *d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

    double elapsed_ms = benchmark_ms(
        200.0,
        2,
        [&]() {
            CURAND_CHECK(curandGenerateUniform(curandGen, a_gpu, size_in*size_in)); 
        },
        [&]() {
            int const ldu = sing_vectors ? size_in : 1, ldvt = sing_vectors ? size_in : 1;
            CUSOLVER_CHECK(cusolverDnSgesvd(cusolverH, sing_vectors ? 'A' : 'N', sing_vectors ? 'A' : 'N',
                size_in, size_in, a_gpu, size_in, svdout, u_gpu, ldu, vt_gpu, ldvt, d_work, lwork, nullptr, d_info));
        });

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(svdout));
    if (u_gpu)
        CUDA_CHECK(cudaFree(u_gpu));
    if (vt_gpu)
        CUDA_CHECK(cudaFree(vt_gpu));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CURAND_CHECK(curandDestroyGenerator(curandGen));

    if (phase==Phase::BENCHMARK){
        printf("  %8.03f \n", elapsed_ms);
    }
}

void run_all_configs(
    Phase phase,
    std::vector<BenchmarkConfig> const &configs) {
    if (phase == Phase::WARMUP) {
        printf("warmup\n\n");
    }else {
        printf("\n\n");
        printf(
            "  %-6s  %-3s  %-9s \n",
            "size_i",
            "vec",
            "time (ms)");
        printf(
            "  %-6s  %-3s  %-9s  \n",
            "------",
            "---",
            "---------");
    }
    for (auto const &config : configs) {
        run_config( phase, config);
    }
    printf("\n");
}



int main() {
    std::vector<BenchmarkConfig> const configs_test = {
        {256, false},
        {256, true},
        {1024, false},
        {1024, true},
        {4096, false},
        {4096, true},
        {16384, false},
        {16384, true},
        {65536, false},
    };

    run_all_configs(Phase::WARMUP, configs_test);
    run_all_configs(Phase::BENCHMARK, configs_test);

    return 0;
}


