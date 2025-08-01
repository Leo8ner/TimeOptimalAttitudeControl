#pragma once

//==============================================================================
// CUDA-ACCELERATED SPACECRAFT DYNAMICS INTEGRATOR
//==============================================================================
/**
 * @file cuda_dynamics.h
 * @brief High-performance batch spacecraft attitude dynamics integrator using CUDA and SUNDIALS
 *
 * This header defines a GPU-accelerated integrator for spacecraft attitude dynamics
 * that supports batch processing, sparse linear algebra, and forward sensitivity
 * analysis. The implementation uses:
 * - SUNDIALS CVODES for adaptive time stepping
 * - CUDA kernels for dynamics computation
 * - cuSPARSE for sparse Jacobian operations
 * - cuSOLVER for efficient linear system solving
 *
 * Key features:
 * - Quaternion-based attitude representation
 * - Batch processing of multiple spacecraft systems
 * - GPU-accelerated sparse Jacobian computation
 * - Forward sensitivity analysis for optimization
 * - Block-diagonal structure exploitation
 * - C-style API with caller-managed memory
 */
//==============================================================================

//==============================================================================
// SYSTEM INCLUDES
//==============================================================================
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
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
#include <toac/symmetric_spacecraft.h>

using namespace std;

//==============================================================================
// INTEGRATION PARAMETERS AND TOLERANCES
//==============================================================================

/// Default relative tolerance for ODE integration
inline constexpr sunrealtype DEFAULT_RTOL = 1e-12;

/// Default absolute tolerance for ODE integration
inline constexpr sunrealtype DEFAULT_ATOL = 1e-14;

/// Absolute tolerance for sensitivity analysis (relaxed for numerical stability)
inline constexpr sunrealtype SENSITIVITY_ATOL = 1e-2;

/// Relative tolerance for sensitivity analysis
inline constexpr sunrealtype SENSITIVITY_RTOL = 1e-2;

/// Maximum number of internal integration steps before failure
inline constexpr int MAX_CVODE_STEPS = 100000;

//==============================================================================
// ERROR CHECKING MACROS
//==============================================================================

/**
 * @brief Check CUDA runtime API calls and exit on error
 * @param call CUDA function call to check
 */
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl;       \
            exit(1);                                                \
        }                                                           \
    } while (0)

/**
 * @brief Check CUDA kernel launch errors and return from function on error
 */
#define CUDA_CHECK_KERNEL()                                          \
    do                                                               \
    {                                                                \
        cudaError_t err = cudaGetLastError();                        \
        if (err != cudaSuccess)                                      \
        {                                                            \
            cerr << "CUDA kernel error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;   \
            return -1;                                               \
        }                                                            \
    } while (0)

/**
 * @brief Check SUNDIALS function calls and return error code on failure
 * @param call SUNDIALS function call to check
 * @param msg Error message to display
 */
#define SUNDIALS_CHECK(call, msg)                                  \
    do                                                             \
    {                                                              \
        auto retval = call;                                        \
        if (retval != CV_SUCCESS)                                  \
        {                                                          \
            cerr << msg << ": " << retval                          \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            return retval;                                         \
        }                                                          \
    } while (0)

//==============================================================================
// CUDA KERNEL FUNCTION DECLARATIONS
//==============================================================================

/**
 * @brief GPU kernel for computing spacecraft dynamics right-hand side
 *
 * Computes time derivatives for quaternion kinematics and Euler's equations
 * for rigid body rotation. Each thread processes one spacecraft system.
 *
 * @param n_total Total number of state variables across all systems
 * @param y State vector [q0,q1,q2,q3,wx,wy,wz] for each system (device memory)
 * @param ydot Time derivatives output vector (device memory)
 * @param torque_params Applied torques [τx,τy,τz] for each system (device memory)
 * @param dt_params Integration time steps for each system (device memory)
 */
__global__ void dynamicsRHS(int n_total, sunrealtype *y, sunrealtype *ydot,
                            sunrealtype *torque_params, sunrealtype *dt_params);

/**
 * @brief GPU kernel for computing sparse Jacobian matrix blocks
 *
 * Computes analytical Jacobian ∂f/∂y for each spacecraft system using
 * block-diagonal CSR format. Exploits sparsity structure of dynamics equations.
 *
 * @param n_blocks Number of spacecraft systems (blocks)
 * @param block_data Sparse Jacobian matrix data in CSR format (device memory)
 * @param y Current state vector (device memory)
 * @param dt_params Time step parameters for each system (device memory)
 */
__global__ void sparseJacobian(int n_blocks, sunrealtype *block_data,
                               sunrealtype *y, sunrealtype *dt_params);

/**
 * @brief GPU kernel for computing forward sensitivity equations
 *
 * Computes sensitivity derivatives ∂ẏ/∂p using Jacobian-vector products
 * for forward sensitivity analysis. Used for gradient-based optimization.
 *
 * @param Ns Total number of sensitivity parameters
 * @param y Current state vector (device memory)
 * @param yS_data_array Array of pointers to sensitivity vectors (device memory)
 * @param ySdot_data_array Array of pointers to sensitivity derivatives (device memory)
 * @param dt_params Time step parameters for each system (device memory)
 */
__global__ void sensitivityRHS(int Ns, sunrealtype *y, sunrealtype **yS_data_array,
                               sunrealtype **ySdot_data_array, sunrealtype *torque_params, sunrealtype *dt_params);

//==============================================================================
// MAIN DYNAMICS INTEGRATOR CLASS
//==============================================================================

/**
 * @brief High-performance CUDA-accelerated spacecraft dynamics integrator
 *
 * This class provides batch integration of spacecraft attitude dynamics using
 * quaternion representation and Euler's equations. Key capabilities include:
 *
 * Features:
 * - GPU-accelerated batch processing of multiple spacecraft systems
 * - Adaptive time stepping with SUNDIALS CVODES
 * - Sparse Jacobian computation with cuSPARSE
 * - Forward sensitivity analysis for optimization applications
 * - Block-diagonal structure exploitation for efficiency
 * - Quaternion norm preservation
 * - C-style API with caller-managed memory allocation
 *
 * Design Philosophy:
 * - Expensive setup operations performed once in constructor
 * - Repeated solve() calls with different parameters are efficient
 * - Memory transfers minimized through pinned host memory
 * - GPU kernels optimized for coalesced memory access
 * - All array memory allocation responsibility lies with caller
 *
 * Thread Safety: Not thread-safe - create separate instances for concurrent use
 */
class DynamicsIntegrator
{
public:
    //==========================================================================
    // CONSTRUCTOR AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructor - performs comprehensive one-time setup
     *
     * Initializes all GPU contexts, SUNDIALS solvers, and internal memory allocations.
     * This is an expensive operation designed to be called once per integrator
     * instance. Caller remains responsible for all input/output array allocations.
     *
     * Setup operations include:
     * - CUDA context and library handle creation
     * - SUNDIALS CVODES initialization
     * - Sparse matrix structure setup
     * - Pinned memory allocation for efficient transfers
     * - Sensitivity analysis preparation (if enabled)
     *
     * @param enable_sensitivity Enable forward sensitivity analysis capability
     * @throws std::runtime_error on GPU or SUNDIALS initialization failure
     */
    explicit DynamicsIntegrator(bool enable_sensitivity = false);

    /**
     * @brief Destructor - cleanup all GPU and SUNDIALS resources
     *
     * Ensures proper cleanup of all allocated resources including GPU memory,
     * SUNDIALS contexts, and library handles. Does not free caller-allocated arrays.
     */
    ~DynamicsIntegrator();

    //==========================================================================
    // MAIN INTERFACE METHODS
    //==========================================================================

    /**
     * @brief Solve batch spacecraft dynamics problem
     *
     * This is the primary method called repeatedly during optimization or
     * simulation. Integrates from t=0 to t=delta_t for all spacecraft systems
     * simultaneously using GPU acceleration.
     *
     * Integration process:
     * 1. Transfer initial conditions and parameters to GPU
     * 2. Setup/reinitialize sensitivity analysis (if requested)
     * 3. Perform adaptive integration using CVODES
     * 4. Extract final states and sensitivities
     *
     * Memory layout expectations:
     * - initial_states: [sys0_states, sys1_states, ..., sysN_states]
     * - torque_params: [sys0_controls, sys1_controls, ..., sysN_controls]
     * - delta_t: [dt0, dt1, ..., dtN]
     *
     * @param initial_states Caller-allocated array of initial quaternions and angular velocities
     *                       (size: n_stp × n_states elements)
     * @param torque_params Caller-allocated array of applied torques for each system
     *                      (size: n_stp × n_controls elements)
     * @param delta_t Caller-allocated array of integration time steps for each system
     *                (size: n_stp elements)
     * @param enable_sensitivity Compute forward sensitivities for optimization
     * @return 0 on success, negative error code on failure
     */
    int solve(const sunrealtype *initial_states,
              const sunrealtype *torque_params,
              const sunrealtype *delta_t,
              bool enable_sensitivity = false);

    //==========================================================================
    // RESULT ACCESS METHODS
    //==========================================================================

    /**
     * @brief Copy final integrated states from GPU to caller-allocated buffer
     *
     * Copies the final quaternions and angular velocities after integration.
     * Data is copied from GPU memory in the same layout as input.
     *
     * @param next_state Caller-allocated buffer for final states
     *                   (must have space for n_stp × n_states elements)
     *                   Layout: [sys0_final, sys1_final, ..., sysN_final]
     *                   where each system has [q0,q1,q2,q3,wx,wy,wz]
     * @return 0 on success, negative error code on failure
     */
    int getSolution(sunrealtype *next_state) const;

    /**
     * @brief Extract ∂y_final/∂y_initial sensitivities in sparse CCS format
     *
     * Provides sensitivity values for state-to-state derivatives in Compressed
     * Column Storage format for efficient use with sparse matrix libraries.
     * Caller must pre-allocate values array with sufficient space.
     *
     * @param values Caller-allocated array for non-zero sensitivity values
     *               (size determined by sparsity pattern)
     * @return 0 on success, negative on error
     * @note Requires sensitivity analysis enabled and sparsity pre-computed
     * @note Use getSparsityY0() first to determine required array size
     */
    int getSensitivitiesY0(sunrealtype *values) const;

    /**
     * @brief Extract ∂y_final/∂u sensitivities in sparse CCS format
     *
     * Provides sensitivity values for control-to-state derivatives.
     * Caller must pre-allocate values array with sufficient space.
     *
     * @param values Caller-allocated array for non-zero sensitivity values
     *               (size determined by sparsity pattern)
     * @return 0 on success, negative on error
     * @note Use getSparsityU() first to determine required array size
     */
    int getSensitivitiesU(sunrealtype *values) const;

    /**
     * @brief Extract ∂y_final/∂dt sensitivities in sparse CCS format
     *
     * Provides sensitivity values for time-step-to-state derivatives.
     * Caller must pre-allocate values array with sufficient space.
     *
     * @param values Caller-allocated array for non-zero sensitivity values
     *               (size determined by sparsity pattern)
     * @return 0 on success, negative on error
     * @note Use getSparsityDt() first to determine required array size
     */
    int getSensitivitiesDt(sunrealtype *values) const;

    //==========================================================================
    // SPARSITY PATTERN ACCESS METHODS
    //==========================================================================

    /**
     * @brief Get sparsity pattern for ∂y_final/∂y_initial Jacobian
     *
     * Returns the CCS sparsity structure that remains constant across
     * different parameter values. Caller must pre-allocate arrays.
     *
     * @param row_indices Caller-allocated array for row indices of non-zero entries
     * @param col_pointers Caller-allocated array for column pointers in CCS format
     *                     (size: n_stp × n_states + 1 elements)
     * @param nnz_out Output parameter: number of non-zero entries
     * @return 0 on success, negative on error
     */
    int getSparsityY0(long long int *row_indices, long long int *col_pointers);

    /**
     * @brief Get sparsity pattern for ∂y_final/∂u Jacobian
     *
     * @param row_indices Caller-allocated array for row indices of non-zero entries
     * @param col_pointers Caller-allocated array for column pointers in CCS format
     *                     (size: n_stp × n_controls + 1 elements)
     * @param nnz_out Output parameter: number of non-zero entries
     * @return 0 on success, negative on error
     */
    int getSparsityU(long long int *row_indices, long long int *col_pointers);

    /**
     * @brief Get sparsity pattern for ∂y_final/∂dt Jacobian
     *
     * @param row_indices Caller-allocated array for row indices of non-zero entries
     * @param col_pointers Caller-allocated array for column pointers in CCS format
     *                     (size: n_stp + 1 elements)
     * @param nnz_out Output parameter: number of non-zero entries
     * @return 0 on success, negative on error
     */
    int getSparsityDt(long long int *row_indices, long long int *col_pointers);

    //==========================================================================
    // SPARSITY SIZE QUERY METHODS
    //==========================================================================

    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂y_initial Jacobian
     * @return Number of non-zero entries, negative on error
     */
    int getSparsitySizeY0();

    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂u Jacobian
     * @return Number of non-zero entries, negative on error
     */
    int getSparsitySizeU();

    /**
     * @brief Get the number of non-zero entries in ∂y_final/∂dt Jacobian
     * @return Number of non-zero entries, negative on error
     */
    int getSparsitySizeDt();

    //==========================================================================
    // PERFORMANCE MONITORING METHODS
    //==========================================================================

    /**
     * @brief Get time spent in one-time setup operations
     * @return Setup time in milliseconds
     */
    float getSetupTime() const { return setup_time; }

    /**
     * @brief Get time spent in most recent integration
     * @return Integration time in milliseconds
     */
    float getSolveTime() const { return solve_time; }

    /**
     * @brief Get total computation time (setup + solve)
     * @return Total time in milliseconds
     */
    float getTotalTime() const { return setup_time + solve_time; }

private:
    //==========================================================================
    // SUNDIALS INTEGRATION COMPONENTS
    //==========================================================================

    void *cvode_mem;    ///< CVODES integrator memory structure
    SUNContext sunctx;  ///< SUNDIALS execution context
    SUNMatrix Jac;      ///< Sparse Jacobian matrix for Newton iterations
    SUNLinearSolver LS; ///< Linear solver for Newton systems
    N_Vector y;         ///< Main solution vector on GPU

    //==========================================================================
    // CUDA LIBRARY HANDLES
    //==========================================================================

    cusparseHandle_t cusparse_handle;   ///< cuSPARSE handle for sparse operations
    cusolverSpHandle_t cusolver_handle; ///< cuSOLVER handle for linear algebra
    cudaStream_t compute_stream;        ///< CUDA stream for sensitivity kernels

    //==========================================================================
    // GPU MEMORY MANAGEMENT
    //==========================================================================

    /// Pinned host memory for initial states (fast GPU transfer)
    sunrealtype *h_y_pinned;

    /// Pinned host memory for torque parameters (fast GPU transfer)
    sunrealtype *h_tau_pinned;

    /// Pinned host memory for time step parameters (fast GPU transfer)
    sunrealtype *h_dt_pinned;

    /// Device memory for torque parameters accessed by GPU kernels
    sunrealtype *d_torque_params_ptr;

    /// Device memory for time step parameters accessed by GPU kernels
    sunrealtype *d_dt_params_ptr;

    //==========================================================================
    // SENSITIVITY ANALYSIS COMPONENTS
    //==========================================================================

    N_Vector *yS;             ///< Array of sensitivity vectors on GPU
    int Ns;                   ///< Total number of sensitivity parameters
    bool sensitivity_enabled; ///< Flag indicating if sensitivity is currently active
    bool sens_was_setup;      ///< Flag indicating if sensitivity was ever initialized
    bool sparsity_computed;   ///< Flag indicating if sparsity patterns were computed

    /// Device arrays of pointers to sensitivity vectors (for kernel access)
    sunrealtype **d_yS_ptrs;
    sunrealtype **d_ySdot_ptrs;

    //==========================================================================
    // SPARSE JACOBIAN STRUCTURE (INTERNAL STORAGE)
    //==========================================================================

    /// Internal storage for ∂y/∂y₀ row indices (populated during setup)
    long long int *y0_row_idx;
    int y0_nnz; ///< Number of non-zeros in ∂y/∂y₀

    /// Internal storage for ∂y/∂u row indices (populated during setup)
    long long int *u_row_idx;
    int u_nnz; ///< Number of non-zeros in ∂y/∂u

    /// Internal storage for ∂y/∂dt row indices (populated during setup)
    long long int *dt_row_idx;
    int dt_nnz; ///< Number of non-zeros in ∂y/∂dt

    //==========================================================================
    // PERFORMANCE MONITORING
    //==========================================================================

    float setup_time; ///< Time spent in constructor setup (milliseconds)
    float solve_time; ///< Time spent in most recent solve() call (milliseconds)

    //==========================================================================
    // PRIVATE SETUP AND UTILITY METHODS
    //==========================================================================

    /**
     * @brief Setup sparse Jacobian structure for SUNDIALS
     *
     * Configures the CSR sparsity pattern for the Jacobian matrix used in
     * Newton iterations. Pattern is fixed and reflects coupling structure.
     */
    void setupJacobianStructure();

    /**
     * @brief Initialize sensitivity analysis vectors and SUNDIALS module
     *
     * Sets up sensitivity parameter vectors and configures CVODES for
     * forward sensitivity analysis.
     *
     * @return 0 on success, negative error code on failure
     */
    int setupSensitivityAnalysis();

    /**
     * @brief Initialize sensitivity vectors with identity structure
     *
     * Sets initial conditions for sensitivity ODEs: ∂y(t=0)/∂p
     * - Identity for initial condition sensitivities
     * - Zero for control and time sensitivities
     */
    void initializeSensitivityVectors();

    /**
     * @brief Transfer initial conditions and parameters to GPU memory
     *
     * Copies problem data from host to device using pinned memory for
     * optimal transfer performance.
     *
     * @param initial_states Initial quaternions and angular velocities
     * @param torque_params Applied torques for each system
     * @param delta_t Integration time steps for each system
     */
    void setInitialConditions(const sunrealtype *initial_states,
                              const sunrealtype *torque_params,
                              const sunrealtype *delta_t);

    /**
     * @brief Compute and cache sparse Jacobian patterns internally
     *
     * Pre-computes the sparsity structure for all sensitivity Jacobians.
     * Called once during sensitivity setup to avoid repeated computation.
     * Allocates internal storage for sparsity patterns.
     */
    void computeSparsities();

    /**
     * @brief Cleanup all allocated GPU and SUNDIALS resources
     *
     * Safely releases all memory and handles. Called by destructor and
     * on error conditions to prevent resource leaks.
     */
    void cleanup();

    //==========================================================================
    // STATIC SUNDIALS CALLBACK FUNCTIONS
    //==========================================================================

    /**
     * @brief SUNDIALS-compatible right-hand side function callback
     *
     * Called by CVODES during integration to compute time derivatives.
     * Launches GPU kernel for batch dynamics computation.
     *
     * @param t Current integration time
     * @param y Current state vector
     * @param ydot Output time derivative vector
     * @param user_data Pointer to DynamicsIntegrator instance
     * @return 0 on success, negative on error
     */
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

    /**
     * @brief SUNDIALS-compatible Jacobian computation callback
     *
     * Called by CVODES for Newton iteration setup. Launches GPU kernel
     * for sparse Jacobian matrix computation.
     *
     * @param t Current integration time
     * @param y Current state vector
     * @param fy Current RHS evaluation (unused)
     * @param Jac Output Jacobian matrix
     * @param user_data Pointer to DynamicsIntegrator instance
     * @param tmp1,tmp2,tmp3 Temporary vectors (unused)
     * @return 0 on success, negative on error
     */
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy,
                                SUNMatrix Jac, void *user_data,
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

    /**
     * @brief SUNDIALS-compatible sensitivity RHS function callback
     *
     * Called by CVODES during sensitivity integration. Launches GPU kernel
     * for batch sensitivity derivative computation.
     *
     * @param Ns Number of sensitivity parameters
     * @param t Current integration time
     * @param y Current state vector
     * @param ydot Current RHS evaluation
     * @param yS Array of sensitivity vectors
     * @param ySdot Output sensitivity derivative vectors
     * @param user_data Pointer to DynamicsIntegrator instance
     * @param tmp1,tmp2 Temporary vectors (unused)
     * @return 0 on success, negative on error
     */
    static int sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                                      N_Vector *yS, N_Vector *ySdot, void *user_data,
                                      N_Vector tmp1, N_Vector tmp2);
};