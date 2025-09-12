#pragma once

//==============================================================================
// DYNAMICS INTEGRATOR HEADER (Updated for C-Array API)
//==============================================================================
// This header defines a CUDA-accelerated dynamics integrator for batch 
// processing of spacecraft attitude dynamics using C-style arrays for 
// maximum compatibility with CasADi external functions.
//==============================================================================

//==============================================================================
// SYSTEM INCLUDES
//==============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <atomic>

//==============================================================================
// CUDA INCLUDES
//==============================================================================
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

//==============================================================================
// SUNDIALS INCLUDES
//==============================================================================
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_context.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>

//==============================================================================
// PROJECT INCLUDES
//==============================================================================
#include "symmetric_spacecraft.h"

using namespace std;

//==============================================================================
// CONSTANTS AND MACROS
//==============================================================================

// Default parameters for integration
inline constexpr sunrealtype DEFAULT_RTOL = 1e-6;
inline constexpr sunrealtype DEFAULT_ATOL = 1e-8;
inline constexpr sunrealtype SENSITIVITY_ATOL = 1e-2;
inline constexpr sunrealtype SENSITIVITY_RTOL = 1e-2;
inline constexpr int MAX_CVODE_STEPS = 100000;
inline constexpr sunrealtype QUAT_NORM_TOLERANCE = 1e-8;

//------------------------------------------------------------------------------
// Error checking macros
//------------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            cerr << "CUDA kernel error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << endl; \
            return -1; \
        } \
    } while(0)

#define SUNDIALS_CHECK(call, msg) \
    do { \
        auto retval = call; \
        if (retval != CV_SUCCESS) { \
            cerr << msg << ": " << retval \
                      << " at " << __FILE__ << ":" << __LINE__ << endl; \
            return retval; \
        } \
    } while(0)

//==============================================================================
// CUDA KERNEL DECLARATIONS
//==============================================================================

/**
 * @brief CUDA kernel for computing dynamics right-hand side
 */
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot, 
                            sunrealtype* torque_params, sunrealtype* dt_params);

/**
 * @brief CUDA kernel for computing sparse Jacobian matrix
 */
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, 
                               sunrealtype* y, sunrealtype* dt_params);

/**
 * @brief CUDA kernel for computing sensitivity right-hand side
 */
__global__ void sensitivityRHS(int Ns, sunrealtype* y, sunrealtype** yS_data_array, 
                              sunrealtype** ySdot_data_array, sunrealtype* dt_params);

//==============================================================================
// VALIDATION STRUCTURES
//==============================================================================

/**
 * Validation error metrics between computed and expected solutions
 */
struct ValidationStats {
    sunrealtype max_absolute_error;
    sunrealtype mean_absolute_error;
    sunrealtype max_relative_error;
    sunrealtype mean_relative_error;
    sunrealtype rms_error;
    vector<sunrealtype> per_state_max_errors;
    vector<sunrealtype> per_system_max_errors;
};

//==============================================================================
// MAIN INTEGRATOR CLASS (Updated for C-Array API)
//==============================================================================

/**
 * @brief CUDA-accelerated dynamics integrator with C-style array interface
 * 
 * This class provides high-performance batch integration of spacecraft attitude
 * dynamics using SUNDIALS CVODES with CUDA acceleration. Updated to use C-style
 * arrays for maximum compatibility with CasADi external functions.
 * 
 * Key features:
 * - Batch processing of multiple trajectories
 * - Sparse Jacobian computation on GPU  
 * - Forward sensitivity analysis
 * - Quaternion-based attitude representation
 * - C-style array API for CasADi compatibility
 * - Caller-managed memory for all inputs/outputs
 */
class DynamicsIntegrator {
private:
    //--------------------------------------------------------------------------
    // SUNDIALS Integration Components
    //--------------------------------------------------------------------------
    void* cvode_mem;                ///< CVODES memory structure
    SUNContext sunctx;              ///< SUNDIALS context
    SUNMatrix Jac;                  ///< Sparse Jacobian matrix
    SUNLinearSolver LS;             ///< Linear solver
    N_Vector y;                     ///< Solution vector

    //--------------------------------------------------------------------------
    // CUDA Components
    //--------------------------------------------------------------------------
    cusparseHandle_t cusparse_handle;   ///< cuSPARSE handle for sparse operations
    cusolverSpHandle_t cusolver_handle; ///< cuSOLVER handle for linear algebra
    cudaStream_t compute_stream;        ///< CUDA stream for computations
    
    //--------------------------------------------------------------------------
    // Memory Management (Updated for C-arrays)
    //--------------------------------------------------------------------------
    sunrealtype* h_y_pinned;           ///< Pinned host memory for efficient transfers
    sunrealtype* h_tau_pinned;         ///< Pinned host memory for torque parameters
    sunrealtype* h_dt_pinned;          ///< Pinned host memory for time step parameters
    sunrealtype* d_torque_params_ptr;  ///< Device memory for torque parameters
    sunrealtype* d_dt_params_ptr;      ///< Device memory for time step parameters

    //--------------------------------------------------------------------------
    // Problem Dimensions
    //--------------------------------------------------------------------------
    int n_total;                    ///< Total number of states (n_states * n_stp)
    
    //--------------------------------------------------------------------------
    // Sensitivity Analysis (Updated for C-arrays)
    //--------------------------------------------------------------------------
    N_Vector* yS;                   ///< Array of sensitivity vectors
    int Ns;                         ///< Number of sensitivity parameters
    bool sensitivity_enabled;       ///< Flag indicating if sensitivity is active
    sunrealtype** d_yS_ptrs;        ///< Device array of yS vector pointers
    sunrealtype** d_ySdot_ptrs;     ///< Device array of ySdot vector pointers
    bool sens_was_setup;            ///< Flag indicating if sensitivity was set up

    // Internal sparsity storage (C-style arrays)
    long long int* y0_row_idx;         ///< Row indices for Y0 sparsity pattern
    int y0_nnz;                     ///< Number of non-zeros in Y0 Jacobian
    long long int* u_row_idx;          ///< Row indices for U sparsity pattern  
    int u_nnz;                      ///< Number of non-zeros in U Jacobian
    long long int* dt_row_idx;         ///< Row indices for dt sparsity pattern
    int dt_nnz;                     ///< Number of non-zeros in dt Jacobian

    //--------------------------------------------------------------------------
    // Performance Monitoring
    //--------------------------------------------------------------------------
    float setup_time;              ///< Time spent in setup operations
    float solve_time;              ///< Time spent in integration
    
    //--------------------------------------------------------------------------
    // Private Helper Methods (Updated for C-arrays)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Setup the sparsity pattern for the Jacobian matrix
     */
    void setupJacobianStructure();
    
    /**
     * @brief Set initial conditions for all integration steps
     */
    void setInitialConditions(const sunrealtype* initial_states,
                              const sunrealtype* torque_params,
                              const sunrealtype* delta_t);
    
    /**
     * @brief Setup sensitivity analysis vectors and parameters
     */
    int setupSensitivityAnalysis();
    
    /**
     * @brief Initialize sensitivity vectors with appropriate values
     */
    void initializeSensitivityVectors();

    /**
     * @brief Pre-compute sparse Jacobian patterns
     */
    void computeSparsities();
    
    //--------------------------------------------------------------------------
    // Static SUNDIALS Callback Functions
    //--------------------------------------------------------------------------
    
    /**
     * @brief Right-hand side function callback for SUNDIALS
     */
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    
    /**
     * @brief Jacobian computation callback for SUNDIALS
     */
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                                SUNMatrix Jac, void* user_data, 
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    
    /**
     * @brief Sensitivity RHS function callback for SUNDIALS
     */
    static int sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                                      N_Vector* yS, N_Vector* ySdot, void* user_data,
                                      N_Vector tmp1, N_Vector tmp2);

    /**
     * @brief Comprehensive cleanup of all allocated resources
     */
    void cleanup();

    //--------------------------------------------------------------------------
    // Test and Validation Methods (Updated for C-arrays)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Generate random test data
     */
    void generateRandomQuaternion(sunrealtype* quaternion);
    void generateRandomAngularVelocity(sunrealtype* omega);
    void generateRandomTorque(sunrealtype* torque);
    void generateBatchInputs(int batch_size, sunrealtype* initial_states,
                             sunrealtype* torque_params, sunrealtype* dt_params);

    /**
     * @brief Physics validation methods
     */
    void computeAngularMomentum(const sunrealtype* states, sunrealtype* angular_momentum) const;
    void computeRotationalEnergy(const sunrealtype* states, sunrealtype* energy) const;
    void computePowerInput(const sunrealtype* states,
                                           const sunrealtype* torques, 
                                           sunrealtype* power) const;
    void getQuaternionNorms(const sunrealtype* states, sunrealtype* norms) const;
    void verifyPhysics(const sunrealtype* initial_states,
                                       const sunrealtype* final_states,
                                       const sunrealtype* torque_params,
                                       const sunrealtype* integration_time) const;
    void validatePhysics(const sunrealtype* initial_states,
                         const sunrealtype* final_states,
                         const sunrealtype* torque_params,
                         const sunrealtype* integration_time) const;

    /**
     * @brief Performance and validation tests
     */
    void profileIntegration(int batch_size, int num_iterations);
    void validateSensitivityAnalysis(const sunrealtype* initial_states,
                                   const sunrealtype* torque_params,
                                   const sunrealtype* integration_time,
                                   int num_systems_to_test = 3);

public:
    void measureSensitivityCost(int n_iterations);

    //--------------------------------------------------------------------------
    // Constructor and Destructor
    //--------------------------------------------------------------------------
    
    /**
     * @brief Constructor - performs one-time setup operations
     * @param enable_sensitivity Enable forward sensitivity analysis capability
     */
    explicit DynamicsIntegrator(bool enable_sensitivity = false);
    
    /**
     * @brief Destructor - cleanup GPU and SUNDIALS resources
     */
    ~DynamicsIntegrator();
    
    //--------------------------------------------------------------------------
    // Main Interface Methods (Updated for C-arrays)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Solve the batch dynamics problem using C-style arrays
     * 
     * This is the main method called repeatedly with different initial conditions
     * and control inputs. All arrays are caller-allocated.
     * 
     * @param initial_states Caller-allocated array [n_stp × n_states]
     * @param torque_params Caller-allocated array [n_stp × n_controls]
     * @param delta_t Caller-allocated array [n_stp]
     * @param enable_sensitivity Enable forward sensitivity analysis
     * @return Success/failure code (0 for success)
     */
    int solve(const sunrealtype* initial_states, 
              const sunrealtype* torque_params,
              const sunrealtype* delta_t, 
              bool enable_sensitivity = false);
    
    //--------------------------------------------------------------------------
    // Result Access Methods (Updated for C-arrays)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Copy final integrated states to caller-allocated buffer
     * @param next_state Caller-allocated array [n_stp × n_states]
     * @return Success/failure code (0 for success)
     */
    int getSolution(sunrealtype* next_state) const;
    
    //--------------------------------------------------------------------------
    // Sparse Sensitivity Access Methods (New C-array API)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Extract ∂y_final/∂y_initial sensitivities to caller-allocated array
     * @param values Caller-allocated array for sensitivity values
     * @return Success/failure code (0 for success)
     */
    int getSensitivitiesY0(sunrealtype* values) const;
    
    /**
     * @brief Extract ∂y_final/∂u sensitivities to caller-allocated array
     * @param values Caller-allocated array for sensitivity values
     * @return Success/failure code (0 for success)
     */
    int getSensitivitiesU(sunrealtype* values) const;
    
    /**
     * @brief Extract ∂y_final/∂dt sensitivities to caller-allocated array
     * @param values Caller-allocated array for sensitivity values
     * @return Success/failure code (0 for success)
     */
    int getSensitivitiesDt(sunrealtype* values) const;
    
    //--------------------------------------------------------------------------
    // Sparsity Pattern Access Methods (New C-array API)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Get sparsity pattern for ∂y_final/∂y_initial Jacobian
     * @param row_indices Caller-allocated array for row indices
     * @param col_pointers Caller-allocated array for column pointers
     * @param nnz_out Output: number of non-zero entries
     * @return Success/failure code (0 for success)
     */
    int getSparsityY0(long long int* row_indices, long long int* col_pointers, int* nnz_out) const;
    
    /**
     * @brief Get sparsity pattern for ∂y_final/∂u Jacobian
     * @param row_indices Caller-allocated array for row indices
     * @param col_pointers Caller-allocated array for column pointers
     * @param nnz_out Output: number of non-zero entries
     * @return Success/failure code (0 for success)
     */
    int getSparsityU(long long int* row_indices, long long int* col_pointers, int* nnz_out) const;
    
    /**
     * @brief Get sparsity pattern for ∂y_final/∂dt Jacobian
     * @param row_indices Caller-allocated array for row indices
     * @param col_pointers Caller-allocated array for column pointers
     * @param nnz_out Output: number of non-zero entries
     * @return Success/failure code (0 for success)
     */
    int getSparsityDt(long long int* row_indices, long long int* col_pointers, int* nnz_out) const;
    
    //--------------------------------------------------------------------------
    // Sparsity Size Query Methods (New C-array API)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂y_initial Jacobian
     * @return Number of non-zeros, or negative on error
     */
    int getSparsitySizeY0() const;
    
    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂u Jacobian
     * @return Number of non-zeros, or negative on error
     */
    int getSparsitySizeU() const;
    
    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂dt Jacobian
     * @return Number of non-zeros, or negative on error
     */
    int getSparsitySizeDt() const;
    
    //--------------------------------------------------------------------------
    // Performance Metrics
    //--------------------------------------------------------------------------
    
    /**
     * @brief Get time spent in setup operations
     * @return Setup time in milliseconds
     */
    float getSetupTime() const { return setup_time; }
    
    /**
     * @brief Get time spent in integration
     * @return Integration time in milliseconds
     */
    float getSolveTime() const { return solve_time; }
    
    /**
     * @brief Get total computation time
     * @return Total time in milliseconds
     */
    float getTotalTime() const { return setup_time + solve_time; }

    //--------------------------------------------------------------------------
    // Test and Validation Interface
    //--------------------------------------------------------------------------
    
    /**
     * @brief Run comprehensive test suite with C-array API
     * @param batch_sizes Vector of batch sizes to test
     * @param iterations Number of iterations per test
     */
    void runComprehensiveTest(const vector<int>& batch_sizes, int iterations = 5);
};