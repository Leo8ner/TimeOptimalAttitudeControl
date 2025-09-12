// external_dynamics.h
#ifndef EXTERNAL_DYNAMICS_H
#define EXTERNAL_DYNAMICS_H

#include <toac/cuda_dynamics.h>
#include <toac/symmetric_spacecraft.h>

// C standard library includes
#include <stdio.h>
#include <string.h>

// C++ standard library includes (for implementation)
#ifdef __cplusplus
#include <mutex>
#include <unordered_map>
#include <atomic>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// TYPE DEFINITIONS AND CONSTANTS
//==============================================================================

// CasADi type definitions (must match CasADi exactly)
#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

// Problem dimensions (from symmetric_spacecraft.h)
#define n_total_states (n_states * n_stp)
#define n_total_controls (n_controls * n_stp)

// Symbol visibility for shared libraries (matching CasADi pattern)
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

//==============================================================================
// FORWARD DYNAMICS FUNCTION API
//==============================================================================
/**
 * @brief Forward dynamics function: X_next = dynamics(X, U, dt)
 * 
 * Integrates spacecraft attitude dynamics from initial state X to final state X_next
 * over time step dt with control inputs U.
 * 
 * Input dimensions:
 *   X:  [n_states × n_stp] - Initial quaternions and angular velocities  
 *   U:  [n_controls × n_stp] - Applied torques
 *   dt: [n_stp] - Integration time steps
 * 
 * Output dimensions:
 *   X_next: [n_states × n_stp] - Final quaternions and angular velocities
 * 
 * Memory Layout: Column-major (CasADi standard)
 * Thread Safety: Functions are thread-safe through memory handle system
 */

// Reference counting for shared integrator instance
CASADI_SYMBOL_EXPORT void dynamics_incref(void);
CASADI_SYMBOL_EXPORT void dynamics_decref(void);

// Memory management (CasADi-compatible)
CASADI_SYMBOL_EXPORT int dynamics_alloc_mem(void);
CASADI_SYMBOL_EXPORT int dynamics_init_mem(int mem);
CASADI_SYMBOL_EXPORT void dynamics_free_mem(int mem);

// Thread-safe memory checkout/release for non-thread-safe integrator
CASADI_SYMBOL_EXPORT int dynamics_checkout(void);
CASADI_SYMBOL_EXPORT void dynamics_release(int mem);

// Function introspection
CASADI_SYMBOL_EXPORT casadi_int dynamics_n_in(void);    // Returns 3: [X, U, dt]
CASADI_SYMBOL_EXPORT casadi_int dynamics_n_out(void);   // Returns 1: [X_next]

// Default input values
CASADI_SYMBOL_EXPORT casadi_real dynamics_default_in(casadi_int i);

// Input/output names for debugging and introspection
CASADI_SYMBOL_EXPORT const char* dynamics_name_in(casadi_int ind);   // ["X", "U", "dt"]
CASADI_SYMBOL_EXPORT const char* dynamics_name_out(casadi_int ind);  // ["X_next"]

// Sparsity patterns (dense matrices for forward function)
CASADI_SYMBOL_EXPORT const casadi_int* dynamics_sparsity_in(casadi_int ind);
CASADI_SYMBOL_EXPORT const casadi_int* dynamics_sparsity_out(casadi_int ind);

// Work vector sizes for CasADi memory allocation
CASADI_SYMBOL_EXPORT int dynamics_work(casadi_int* sz_arg, casadi_int* sz_res, 
                                      casadi_int* sz_iw, casadi_int* sz_w);
CASADI_SYMBOL_EXPORT int dynamics_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, 
                                            casadi_int* sz_iw, casadi_int* sz_w);

// Main evaluation function
CASADI_SYMBOL_EXPORT int dynamics(const casadi_real** arg, casadi_real** res, 
                                 casadi_int* iw, casadi_real* w, int mem);

//==============================================================================
// JACOBIAN DYNAMICS FUNCTION API  
//==============================================================================
/**
 * @brief Jacobian function: [dX_next/dX, dX_next/dU, dX_next/dt] = jac_dynamics(X, U, dt, X_next)
 * 
 * Computes sparse Jacobian matrices for spacecraft dynamics using forward 
 * sensitivity analysis on GPU.
 * 
 * Input dimensions:
 *   X:      [n_states × n_stp] - Initial states
 *   U:      [n_controls × n_stp] - Controls  
 *   dt:     [n_stp] - Time steps
 *   X_next: [n_states × n_stp] - Final states (for CasADi compatibility)
 * 
 * Output dimensions (sparse CCS format):
 *   jac_X_next_X:  [n_total_states × n_total_states] - State-to-state Jacobian
 *   jac_X_next_U:  [n_total_states × n_total_controls] - Control-to-state Jacobian  
 *   jac_X_next_dt: [n_total_states × n_stp] - Time-to-state Jacobian
 * 
 * Sparsity: Block-diagonal structure exploited for efficiency
 */

// Reference counting (shared with forward dynamics)
CASADI_SYMBOL_EXPORT void jac_dynamics_incref(void);
CASADI_SYMBOL_EXPORT void jac_dynamics_decref(void);

// Memory management  
CASADI_SYMBOL_EXPORT int jac_dynamics_alloc_mem(void);
CASADI_SYMBOL_EXPORT int jac_dynamics_init_mem(int mem);
CASADI_SYMBOL_EXPORT void jac_dynamics_free_mem(int mem);

// Thread-safe memory checkout/release
CASADI_SYMBOL_EXPORT int jac_dynamics_checkout(void);
CASADI_SYMBOL_EXPORT void jac_dynamics_release(int mem);

// Function introspection
CASADI_SYMBOL_EXPORT casadi_int jac_dynamics_n_in(void);    // Returns 4: [X, U, dt, X_next]
CASADI_SYMBOL_EXPORT casadi_int jac_dynamics_n_out(void);   // Returns 3: [jac_X_next_X, jac_X_next_U, jac_X_next_dt]

// Default input values
CASADI_SYMBOL_EXPORT casadi_real jac_dynamics_default_in(casadi_int i);

// Input/output names
CASADI_SYMBOL_EXPORT const char* jac_dynamics_name_in(casadi_int ind);   // ["X", "U", "dt", "X_next"]
CASADI_SYMBOL_EXPORT const char* jac_dynamics_name_out(casadi_int ind);  // ["jac_X_next_X", "jac_X_next_U", "jac_X_next_dt"]

// Sparse sparsity patterns (CCS format with header [nrow, ncol, colptr..., rowidx...])
CASADI_SYMBOL_EXPORT const casadi_int* jac_dynamics_sparsity_in(casadi_int ind);
CASADI_SYMBOL_EXPORT const casadi_int* jac_dynamics_sparsity_out(casadi_int ind);

// Work vector sizes
CASADI_SYMBOL_EXPORT int jac_dynamics_work(casadi_int* sz_arg, casadi_int* sz_res, 
                                          casadi_int* sz_iw, casadi_int* sz_w);
CASADI_SYMBOL_EXPORT int jac_dynamics_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, 
                                                casadi_int* sz_iw, casadi_int* sz_w);

// Main evaluation function
CASADI_SYMBOL_EXPORT int jac_dynamics(const casadi_real** arg, casadi_real** res, 
                                     casadi_int* iw, casadi_real* w, int mem);

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Initialize the global CUDA dynamics integrator
 * 
 * Creates a single shared DynamicsIntegrator instance with sensitivity analysis
 * enabled. This function is thread-safe and uses reference counting.
 * 
 * @return 0 on success, non-zero on failure
 * @note Expensive operation - performs GPU context setup, SUNDIALS initialization
 */
CASADI_SYMBOL_EXPORT int initialize_dynamics_integrator(void);

/**
 * @brief Cleanup the global dynamics integrator
 * 
 * Decrements reference count and destroys integrator when count reaches zero.
 * Frees all GPU memory, SUNDIALS contexts, and sparsity pattern storage.
 * 
 * @note Thread-safe, should be called once for each initialize_dynamics_integrator() call
 */
CASADI_SYMBOL_EXPORT void cleanup_dynamics_integrator(void);

//==============================================================================
// PERFORMANCE AND DEBUGGING
//==============================================================================

/**
 * @brief Get performance timing information from most recent integration
 * 
 * @param setup_time_ms Output: Time spent in one-time setup (milliseconds) 
 * @param solve_time_ms Output: Time spent in most recent solve (milliseconds)
 * @return 0 on success, negative on error
 * @note Only available if integrator is initialized
 */
CASADI_SYMBOL_EXPORT int dynamics_get_timing(float* setup_time_ms, float* solve_time_ms);

/**
 * @brief Get integrator statistics  
 * 
 * @param forward_calls Output: Number of forward dynamics evaluations
 * @param jacobian_calls Output: Number of Jacobian evaluations  
 * @return 0 on success, negative on error
 */
CASADI_SYMBOL_EXPORT int dynamics_get_stats(int* forward_calls, int* jacobian_calls);

//==============================================================================
// ERROR CODES
//==============================================================================

#define DYNAMICS_SUCCESS           0   // Success
#define DYNAMICS_ERROR_INIT       -1   // Initialization failed
#define DYNAMICS_ERROR_MEMORY     -2   // Memory allocation failed  
#define DYNAMICS_ERROR_INVALID    -3   // Invalid input parameters
#define DYNAMICS_ERROR_INTEGRATION -4   // Integration failed
#define DYNAMICS_ERROR_SENSITIVITY -5   // Sensitivity computation failed
#define DYNAMICS_ERROR_SPARSITY    -6   // Sparsity pattern error

//==============================================================================
// VERSION INFORMATION
//==============================================================================

#define EXTERNAL_DYNAMICS_VERSION_MAJOR 1
#define EXTERNAL_DYNAMICS_VERSION_MINOR 0  
#define EXTERNAL_DYNAMICS_VERSION_PATCH 0

/**
 * @brief Get version string
 * @return Static version string "1.0.0"
 */
CASADI_SYMBOL_EXPORT const char* dynamics_version(void);

#ifdef __cplusplus
}
#endif

#endif // EXTERNAL_DYNAMICS_H