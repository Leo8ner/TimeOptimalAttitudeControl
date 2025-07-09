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
#include <cassert>

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

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) \
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
} while(0)

#define SUNDIALS_CHECK(call, msg) \
    do { \
        auto retval = call; \
        if (retval != CV_SUCCESS) { \
            std::cerr << msg << ": " << retval \
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return retval; \
        } \
} while(0)

// Torque parameters structure
struct TorqueParams {
    // Control inputs
    sunrealtype tau_x, tau_y, tau_z;

};

// State parameters structure
struct StateParams {
    // Initial state inputs
    sunrealtype q0, q1, q2, q3; // Quaternion components
    sunrealtype wx, wy, wz;     // Angular velocities

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
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot);
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, sunrealtype* y);
__global__ void sensitivityRHS(int n_total, int Ns, sunrealtype* y, 
                                      N_Vector* yS_array, N_Vector* ySdot_array);

// Device arrays for step parameters (constant during integration)
extern __device__ TorqueParams* d_torque_params;
extern __device__ __constant__ sunrealtype d_inertia_constants[12];

class DynamicsIntegrator {
private:
    SUNMatrix Jac;
    SUNLinearSolver LS;
    N_Vector y;
    int n_total, nnz;
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
    TorqueParams* d_torque_params_ptr;
    
    void* cvode_mem;
    SUNContext sunctx;
    
    // Pinned memory (only for critical transfers)
    sunrealtype *h_y_pinned;

    // Sensitivity analysis members
    N_Vector* yS;                    // Sensitivity vectors
    int Ns;                          // Number of parameters
    bool sensitivity_enabled;
    
    // GPU memory for sensitivity computation
    sunrealtype* d_jacobian_workspace;
    
    // Timing
    PrecisionTimer timer;
    double setup_time;
    double solve_time;

    // Verbose output flag
    bool verbose = false;
    
    // Private helper methods
    void setupJacobianStructure();
    void setInitialConditions(const std::vector<StateParams>& initial_states,
                            const std::vector<TorqueParams>& torque_params);
    
    // Static callback functions for SUNDIALS
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    static int sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                            N_Vector* yS, N_Vector* ySdot, void* user_data,
                            N_Vector tmp1, N_Vector tmp2);
    // Sensitivity methods
    int setupSensitivityAnalysis();
    void initializeSensitivityVectors();

    // Validation functions
    bool validateInputs(const std::vector<StateParams>& initial_states, 
                       const std::vector<TorqueParams>& torque_params);
    void printSolutionStats();

public:
    // Constructor - performs one-time setup
    DynamicsIntegrator(bool verb = false);
    
    // Destructor - cleanup
    ~DynamicsIntegrator();
    
    // Main solve function - called repeatedly
    // initial_states: vector of n_stp initial states, each with 7 elements [q0,q1,q2,q3,wx,wy,wz]
    // torque_params: vector of n_stp control parameters
    // delta_t: integration time step
    int solve(const std::vector<StateParams>& initial_states, 
            const std::vector<TorqueParams>& torque_params,
            const sunrealtype& delta_t, bool enable_sensitivity);


    
    // Utility functions to get results
    std::vector<StateParams> getSolution();
    std::vector<sunrealtype> getQuaternionNorms();
    std::vector<std::vector<StateParams>> getSensitivities();
    // Performance metrics
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return solve_time; }
    double getTotalTime() const { return setup_time + solve_time; }
};