#pragma once

//==============================================================================
// DYNAMICS INTEGRATOR HEADER
//==============================================================================
// This header defines a CUDA-accelerated dynamics integrator for batch 
// processing of spacecraft attitude dynamics using SUNDIALS CVODES with 
// sensitivity analysis capabilities.
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
#include <symmetric_spacecraft.h>

//==============================================================================
// NAMESPACE USAGE
//==============================================================================
using namespace std;

//==============================================================================
// CONSTANTS AND MACROS
//==============================================================================

// Batch processing constants
inline constexpr int N_TOTAL_STATES = (n_states * n_stp);

// Sparsity pattern constants
inline constexpr int NNZ_PER_BLOCK = 37;  // Nonzeros per block: 4*6 + 3*2 + 7 diagonal = 37

// Default parameters for integration
inline constexpr sunrealtype DEFAULT_RTOL = 1e-8;
inline constexpr sunrealtype DEFAULT_ATOL = 1e-10;
inline constexpr sunrealtype SENSITIVITY_ATOL = 1e-12;
inline constexpr sunrealtype SENSITIVITY_RTOL = 1e-10;
inline constexpr int MAX_CVODE_STEPS = 100000;

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
 * @param n_total Total number of states across all integration steps
 * @param y State vector
 * @param ydot Time derivative of state vector
 */
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot);

/**
 * @brief CUDA kernel for computing sparse Jacobian matrix
 * @param n_blocks Number of blocks in the batch
 * @param block_data Jacobian matrix data in block format
 * @param y State vector
 */
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, sunrealtype* y);

/**
 * @brief CUDA kernel for computing sensitivity right-hand side
 * @param n_total Total number of states
 * @param Ns Number of sensitivity parameters
 * @param y State vector
 * @param yS_array Array of sensitivity vectors
 * @param ySdot_array Array of sensitivity time derivatives
 */
__global__ void sensitivityRHS(int Ns, sunrealtype* y, sunrealtype** yS_data_array, sunrealtype** ySdot_data_array);

//==============================================================================
// MAIN INTEGRATOR CLASS
//==============================================================================

/**
 * @brief CUDA-accelerated dynamics integrator for batch spacecraft simulation
 * 
 * This class provides high-performance batch integration of spacecraft attitude
 * dynamics using SUNDIALS CVODES with CUDA acceleration. It supports:
 * - Batch processing of multiple trajectories
 * - Sparse Jacobian computation on GPU
 * - Forward sensitivity analysis
 * - Quaternion-based attitude representation
 * 
 * The integrator is designed for repeated use with different initial conditions
 * and control inputs, with expensive setup operations performed only once.
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
    sunrealtype* d_torque_params_ptr;  ///< Device pointer to torque parameters
    
    //--------------------------------------------------------------------------
    // Memory Management
    //--------------------------------------------------------------------------
    sunrealtype* h_y_pinned;           ///< Pinned host memory for efficient state transfers
    sunrealtype* h_tau_pinned;         ///< Pinned host memory for efficient control transfers

    //--------------------------------------------------------------------------
    // Problem Dimensions
    //--------------------------------------------------------------------------
    int n_total;                    ///< Total number of states (n_states * n_stp)
    int nnz;                        ///< Number of nonzeros in Jacobian
    int n_batch;                    ///< Number of batches (integration steps)
    
    //--------------------------------------------------------------------------
    // Sensitivity Analysis
    //--------------------------------------------------------------------------
    N_Vector* yS;                   ///< Array of sensitivity vectors
    int Ns;                         ///< Number of sensitivity parameters
    bool sensitivity_enabled;       ///< Flag indicating if sensitivity is active
    sunrealtype** d_yS_ptrs;        ///< Device array of yS vector pointers
    sunrealtype** d_ySdot_ptrs;     ///< Device array of ySdot vector pointers
    bool sens_was_setup;            ///< Flag indicating if sensitivity was set up
    //--------------------------------------------------------------------------
    // Performance Monitoring
    //--------------------------------------------------------------------------
    float setup_time;              ///< Time spent in setup operations
    float solve_time;              ///< Time spent in integration
    bool verbose;                  ///< Verbose output flag
    
    //--------------------------------------------------------------------------
    // Private Helper Methods
    //--------------------------------------------------------------------------
    
    /**
     * @brief Setup the sparsity pattern for the Jacobian matrix
     */
    void setupJacobianStructure();
    
    /**
     * @brief Set initial conditions for all integration steps
     * @param initial_states Vector of initial state parameters
     * @param torque_params Vector of torque control parameters
     */
    void setInitialConditions(const vector<vector<sunrealtype>>& initial_states,
                              const vector<vector<sunrealtype>>& torque_params);
    
    /**
     * @brief Setup sensitivity analysis vectors and parameters
     * @return Success/failure code
     */
    int setupSensitivityAnalysis();
    
    /**
     * @brief Initialize sensitivity vectors with appropriate values
     */
    void initializeSensitivityVectors();
    
    /**
     * @brief Update sensitivity vector pointers on the device
     * 
     * This is called after sensitivity vectors are initialized or modified.
     */
    void initializeSensitivityPointers();
    
    /**
     * @brief Print solution statistics (when verbose mode is enabled)
     */
    void printSolutionStats();
    
    //--------------------------------------------------------------------------
    // Static SUNDIALS Callback Functions
    //--------------------------------------------------------------------------
    
    /**
     * @brief Right-hand side function callback for SUNDIALS
     * @param t Current time
     * @param y Current state vector
     * @param ydot Time derivative vector (output)
     * @param user_data User-defined data pointer
     * @return Success/failure code
     */
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    
    /**
     * @brief Jacobian computation callback for SUNDIALS
     * @param t Current time
     * @param y Current state vector
     * @param fy Current RHS vector
     * @param Jac Jacobian matrix (output)
     * @param user_data User-defined data pointer
     * @param tmp1,tmp2,tmp3 Temporary vectors
     * @return Success/failure code
     */
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                                SUNMatrix Jac, void* user_data, 
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    
    /**
     * @brief Sensitivity RHS function callback for SUNDIALS
     * @param Ns Number of sensitivity parameters
     * @param t Current time
     * @param y Current state vector
     * @param ydot Current RHS vector
     * @param yS Array of sensitivity vectors
     * @param ySdot Array of sensitivity RHS vectors (output)
     * @param user_data User-defined data pointer
     * @param tmp1,tmp2 Temporary vectors
     * @return Success/failure code
     */
    static int sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                                      N_Vector* yS, N_Vector* ySdot, void* user_data,
                                      N_Vector tmp1, N_Vector tmp2);

    /**
     * @brief Cleanup all allocated resources
     * This is called in the destructor and on error.
     * It releases all GPU and SUNDIALS resources.
     */
    void cleanup ();

public:
    //--------------------------------------------------------------------------
    // Constructor and Destructor
    //--------------------------------------------------------------------------
    
    /**
     * @brief Constructor - performs one-time setup operations
     * @param verb Enable verbose output for debugging
     */
    explicit DynamicsIntegrator(bool enable_sensitivity = false, bool verb = false);
    
    /**
     * @brief Destructor - cleanup GPU and SUNDIALS resources
     */
    ~DynamicsIntegrator();
    
    //--------------------------------------------------------------------------
    // Main Interface Methods
    //--------------------------------------------------------------------------
    
    /**
     * @brief Solve the batch dynamics problem
     * 
     * This is the main method called repeatedly with different initial conditions
     * and control inputs. It performs the integration from t=0 to t=delta_t.
     * 
     * @param initial_states Vector of initial state parameters (size n_stp)
     * @param torque_params Vector of control torque parameters (size n_stp)
     * @param delta_t Integration time step
     * @param enable_sensitivity Enable forward sensitivity analysis
     * @return Success/failure code (0 for success)
     */
    int solve(const vector<vector<sunrealtype>>& initial_states, 
              const vector<vector<sunrealtype>>& torque_params,
              const sunrealtype& delta_t, 
              bool enable_sensitivity = false);
    
    //--------------------------------------------------------------------------
    // Result Access Methods
    //--------------------------------------------------------------------------
    
    /**
     * @brief Get the integrated solution states
     * @return Vector of final state parameters
     */
    vector<vector<sunrealtype>> getSolution() const;
    
    /**
     * @brief Get quaternion norms for validation
     * @return Vector of quaternion norms (should be close to 1.0)
     */
    vector<sunrealtype> getQuaternionNorms() const;
    
    /**
     * @brief Get sensitivity analysis results
     * @return Vector of sensitivity vectors for each parameter
     */
    tuple<vector<sunrealtype>, vector<int>, vector<int>, int, int> 
    getSensitivities() const;
    
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
    
};