#pragma once
#include <casadi/casadi.hpp>
#include <toac/cuda_dynamics.h>
#include <memory>
#include <vector>
#include <iostream>
#include <filesystem>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

// Forward declaration for circular dependency resolution
class JacobianCallback;

//==============================================================================
// DYNAMICS CALLBACK CLASS
//==============================================================================

/**
 * CasADi callback interface for GPU-accelerated spacecraft dynamics integration
 *
 * This class provides a CasADi-compatible interface to the CUDA-based spacecraft
 * dynamics integrator, enabling seamless integration with optimization frameworks
 * like IPOPT. Supports sparse Jacobian computation for efficient gradient-based
 * optimization.
 *
 * Input format (column-major for CasADi compatibility):
 * - arg[0]: initial_states (n_states × n_stp) - Initial quaternions and angular velocities
 * - arg[1]: torque_params (n_controls × n_stp) - Applied torques for each system
 * - arg[2]: delta_t (n_stp × 1) - Integration time steps for each system
 *
 * Output format:
 * - res[0]: final_states (n_states × n_stp) - Final quaternions and angular velocities
 *
 * Features:
 * - Batch processing of multiple spacecraft systems
 * - GPU-accelerated integration using SUNDIALS/CVODES
 * - Sparse analytical Jacobian computation
 * - Forward sensitivity analysis for optimization
 * - C-style array API with caller-managed memory
 */
class DynamicsCallback : public Callback {
private:
    std::string name_;                                    ///< Callback function name for CasADi
    Dict opts_;                                          ///< CasADi options dictionary
    std::unique_ptr<DynamicsIntegrator> integrator_;     ///< GPU dynamics integrator instance
    
    /// Shared pointer to Jacobian callback to maintain proper lifetime management
    mutable std::shared_ptr<JacobianCallback> jac_callback_;

public:
    /**
     * Constructor for dynamics callback
     *
     * Initializes the GPU integrator with sensitivity analysis enabled and
     * constructs the CasADi callback interface.
     *
     * @param name Function name for CasADi registration
     * @param opts CasADi options dictionary (default: empty)
     */
    DynamicsCallback(std::string name, Dict opts = Dict());

    /// Default destructor - unique_ptr handles cleanup automatically
    ~DynamicsCallback() = default;

    // CasADi callback interface methods (required overrides)
    
    /// Return number of input arguments (3: states, controls, time)
    casadi_int get_n_in() override { return 3; }
    
    /// Return number of output arguments (1: final states)
    casadi_int get_n_out() override { return 1; }
    
    /**
     * Get symbolic name for input argument
     * @param i Input index (0=X, 1=U, 2=dt)
     * @return Human-readable input name
     */
    std::string get_name_in(casadi_int i) override;
    
    /**
     * Get symbolic name for output argument
     * @param i Output index (0=X_next)
     * @return Human-readable output name
     */
    std::string get_name_out(casadi_int i) override;
    
    /**
     * Define sparsity pattern for input arguments
     * @param i Input index
     * @return Sparsity pattern (all inputs are dense matrices)
     */
    Sparsity get_sparsity_in(casadi_int i) override;
    
    /**
     * Define sparsity pattern for output arguments
     * @param i Output index
     * @return Sparsity pattern (output is dense matrix)
     */
    Sparsity get_sparsity_out(casadi_int i) override;

    /**
     * Main function evaluation - performs spacecraft dynamics integration
     *
     * Converts CasADi matrices to integrator format, calls GPU integration,
     * and converts results back to CasADi format.
     *
     * @param arg Vector of input matrices [states, controls, time]
     * @return Vector containing final states matrix
     */
    std::vector<DM> eval(const std::vector<DM>& arg) const override;

    /// Enable analytical Jacobian computation for optimization
    bool has_jacobian() const override { return true; }

    /**
     * Create analytical Jacobian callback function
     *
     * Constructs a separate callback that computes sparse Jacobian matrices
     * using forward sensitivity analysis on the GPU.
     *
     * @param name Jacobian function name
     * @param inames Input argument names
     * @param onames Output argument names  
     * @param opts CasADi options for Jacobian computation
     * @return CasADi Function object for Jacobian evaluation
     */
    Function get_jacobian(const std::string& name,
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames,
                         const Dict& opts) const override;

    /**
     * Convert CasADi DM matrix to flat array format
     *
     * Transforms column-major CasADi matrix to system-major array format
     * expected by the GPU integrator.
     *
     * @param dm Input CasADi matrix
     * @param output Pre-allocated output array (caller-managed)
     */
    void convertDMtoArray(const DM& dm, double* output) const;
    
    /**
     * Convert integrator array format back to CasADi DM matrix
     *
     * @param data Input array in integrator format
     * @param rows Number of rows in output matrix
     * @param cols Number of columns in output matrix
     * @return CasADi DM matrix in column-major format
     */
    DM convertArrayToDM(const double* data, int rows, int cols) const;
};

//==============================================================================
// JACOBIAN CALLBACK CLASS  
//==============================================================================

/**
 * Specialized callback for computing analytical Jacobian matrices
 *
 * This nested class handles the computation of sparse Jacobian matrices using
 * forward sensitivity analysis. It maintains the same input interface as the
 * parent dynamics callback but returns three sparse Jacobian matrices.
 *
 * Output Jacobians:
 * - ∂X_next/∂X: (n_stp×n_states) × (n_stp×n_states) - State-to-state derivatives
 * - ∂X_next/∂U: (n_stp×n_states) × (n_stp×n_controls) - Control-to-state derivatives  
 * - ∂X_next/∂dt: (n_stp×n_states) × n_stp - Time-to-state derivatives
 *
 * Each Jacobian maintains block-diagonal structure reflecting system independence.
 * Uses C-style array API with proper memory management.
 */
class JacobianCallback : public Callback {
private:
    const DynamicsCallback* parent_;        ///< Reference to parent callback for utility methods
    DynamicsIntegrator* integrator_;        ///< Direct access to GPU integrator for sensitivity data
    
    /// Pre-computed sparsity patterns for efficient Jacobian construction
    Sparsity Jac_y0;   ///< Sparsity pattern for ∂X_next/∂X
    Sparsity Jac_u;    ///< Sparsity pattern for ∂X_next/∂U  
    Sparsity Jac_dt;   ///< Sparsity pattern for ∂X_next/∂dt

    /**
     * Initialize sparse Jacobian patterns from integrator
     *
     * Retrieves pre-computed sparsity structures from the GPU integrator
     * and constructs CasADi sparsity patterns for efficient matrix operations.
     * Uses the new C-style array API with proper memory management.
     */
    void setupSparsity();

public:
    /**
     * Constructor for Jacobian callback
     *
     * @param parent Pointer to parent dynamics callback
     * @param integrator Pointer to GPU integrator with sensitivity capability
     * @param name Function name for CasADi registration
     * @param opts CasADi options dictionary
     */
    JacobianCallback(const DynamicsCallback* parent, 
                    DynamicsIntegrator* integrator, 
                    const std::string& name, 
                    const Dict& opts);

    /// Return number of inputs (4: states, controls, time, final_states)
    casadi_int get_n_in() override { return 4; }
    
    /// Return number of outputs (3: three Jacobian matrices)
    casadi_int get_n_out() override { return 3; }

    /**
     * Define sparsity patterns for Jacobian inputs
     * @param i Input index (same as parent callback plus final states)
     * @return Input sparsity pattern
     */
    Sparsity get_sparsity_in(casadi_int i) override;

    /**
     * Define sparsity patterns for Jacobian outputs
     * @param i Output index (0=∂X/∂X, 1=∂X/∂U, 2=∂X/∂dt)
     * @return Sparse Jacobian pattern
     */
    Sparsity get_sparsity_out(casadi_int i) override;

    /**
     * Compute analytical Jacobian matrices using forward sensitivity analysis
     *
     * Performs GPU integration with sensitivity analysis enabled, extracts
     * sensitivity data in sparse format, and constructs CasADi sparse matrices.
     * Uses C-style array API with proper memory management.
     *
     * @param arg Input arguments [states, controls, time, final_states]
     * @return Vector of three sparse Jacobian matrices
     */
    std::vector<DM> eval(const std::vector<DM>& arg) const override;
};