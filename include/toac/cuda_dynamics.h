#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

// SUNDIALS headers
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_context.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>

// Personal headers
#include <toac/symmetric_spacecraft.h>

// Define constants for batch processing
#define N_TOTAL_STATES (n_states * n_stp)

// Sparsity constants
#define NNZ_PER_BLOCK 37  // 4*6 + 3*2 = 24 + 6 + 7 diagonal = 37 nonzeros per block
#define TOTAL_NNZ (NNZ_PER_BLOCK * n_stp)

// Stream and buffer constants
#define N_STREAMS 2

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Step parameters structure
struct StepParams {
    // Control inputs
    sunrealtype tau_x, tau_y, tau_z;
    
    // Constructor for convenience
    StepParams(sunrealtype tx = 0.0, sunrealtype ty = 0.0, sunrealtype tz = 0.0) 
        : tau_x(tx), tau_y(ty), tau_z(tz) {}
};

// Timing utility class
class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMs() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// Forward declaration for CUDA kernels
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot, int systems_per_block);
__global__ void sparseBatchJacobian(int n_blocks, sunrealtype* block_data, sunrealtype* y);

// Device arrays for step parameters (constant during integration)
extern __device__ StepParams* d_step_params;

class OptimizedDynamicsIntegrator {
private:
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector y;
    int n_total, nnz;
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
    StepParams* d_step_params_ptr;
    
    void* cvode_mem;
    SUNContext sunctx;
    
    // Stream optimization
    cudaStream_t streams[N_STREAMS];
    int current_stream = 0;
    
    // Pinned memory (only for critical transfers)
    sunrealtype *h_y_pinned;
    
    // Timing
    PrecisionTimer timer;
    double setup_time;
    double solve_time;
    
    // Private helper methods
    void setupSparseJacobianStructure();
    void initializeCVODES();
    void setInitialConditions(const std::vector<std::vector<sunrealtype>>& initial_states);
    void setStepParams(const std::vector<StepParams>& step_params);
    
    // Static callback functions for SUNDIALS
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

    bool verbose = false; // Verbose output flag

public:
    // Constructor - performs one-time setup
    OptimizedDynamicsIntegrator(bool verb = false);
    
    // Destructor - cleanup
    ~OptimizedDynamicsIntegrator();
    
    // Main solve function - called repeatedly
    // initial_states: vector of n_stp initial states, each with 7 elements [q0,q1,q2,q3,wx,wy,wz]
    // step_params: vector of n_stp control parameters
    // delta_t: integration time step
    double solve(const std::vector<std::vector<sunrealtype>>& initial_states, 
                 const std::vector<StepParams>& step_params,
                 sunrealtype delta_t);
    
    // Utility functions to get results
    std::vector<std::vector<sunrealtype>> getAllSolutions();
    std::vector<sunrealtype> getQuaternionNorms();
    
    // Performance metrics
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return solve_time; }
    double getTotalTime() const { return setup_time + solve_time; }
    
    // Validation functions
    bool validateInputs(const std::vector<std::vector<sunrealtype>>& initial_states, 
                       const std::vector<StepParams>& step_params);
    void printSolutionStats();
};