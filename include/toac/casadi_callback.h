#pragma once
#include <casadi/casadi.hpp>
#include <toac/cuda_dynamics.h>
#include <memory>
#include <vector>
#include <iostream>
#include <filesystem>
#include <toac/symmetric_spacecraft.h>
#include <sundials/sundials_types.h>

using namespace casadi;

// Forward declaration of JacobianCallback
class JacobianCallback;

//==============================================================================
// DYNAMICS CALLBACK CLASS
//==============================================================================

/**
 * Updated DynamicsCallback with sparse Jacobian support for IPOPT optimization
 * 
 * Inputs:
 *   - arg[0]: initial_states (n_states x n_stp)  - all initial states
 *   - arg[1]: torque_params (n_controls x n_stp) - all control inputs  
 *   - arg[2]: delta_t (1 x 1)                    - integration time step
 * 
 * Outputs:
 *   - res[0]: final_states (n_states x n_stp)    - all final states
 */

class DynamicsCallback : public Callback {
private:
    std::string name_;
    bool verbose_;
    Dict opts_;
    std::unique_ptr<DynamicsIntegrator> integrator_;

    // Keep reference to Jacobian callback to maintain lifetime
    mutable std::shared_ptr<JacobianCallback> jac_callback_;
    
public:
    // Constructor
    DynamicsCallback(std::string name, bool verbose = false, Dict opts = Dict());

    ~DynamicsCallback() = default;

    // Required CasADi callback methods
    casadi_int get_n_in() override { return 3; }
    casadi_int get_n_out() override { return 1; }

    std::string get_name_in(casadi_int i) override;

    std::string get_name_out(casadi_int i) override;

    Sparsity get_sparsity_in(casadi_int i) override;

    Sparsity get_sparsity_out(casadi_int i) override;

    // Main evaluation function
    std::vector<DM> eval(const std::vector<DM>& arg) const override;

    // Enable Jacobian computation
    bool has_jacobian() const override { return true; }

    // Create Jacobian callback
    Function get_jacobian(const std::string& name, 
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames, 
                         const Dict& opts) const override;
};
                         
// Inner Jacobian callback class
class JacobianCallback : public Callback {
private:
    const DynamicsCallback* parent_;
    const DynamicsIntegrator* integrator_;

    /**
     * Convert CSR sensitivity data to CasADi sparse format
     * Maps block-diagonal structure to [all_x, all_u] input layout
     */
    std::tuple<std::vector<casadi_int>, std::vector<casadi_int>, std::vector<double>>
    csrToCasADiTriplets(const std::vector<sunrealtype>& csr_data,
                        const std::vector<int>& csr_indices, 
                        const std::vector<int>& csr_indptr) const;
    
public:
    JacobianCallback(const DynamicsCallback* parent, const DynamicsIntegrator* integrator, const std::string& name, const Dict& opts);

    casadi_int get_n_in() override { return 4; }  // Same inputs as parent
    casadi_int get_n_out() override { return 3; } // Jacobian matrix

    Sparsity get_sparsity_in(casadi_int i) override;

    Sparsity get_sparsity_out(casadi_int i) override;

    std::vector<DM> eval(const std::vector<DM>& arg) const override;
};
