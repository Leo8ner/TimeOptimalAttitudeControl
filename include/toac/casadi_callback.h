#pragma once
#include <casadi/casadi.hpp>
#include <toac/cuda_dynamics.h>
#include <memory>
#include <vector>
#include <iostream>
#include <filesystem>
#include <toac/symmetric_spacecraft.h>

// Forward declaration of your integrator
class DynamicsIntegrator;

// Forward declarations for parameter structures
struct StateParams;
struct TorqueParams;

using namespace casadi;

/**
 * Base Callback for spacecraft dynamics integration
 * 
 * Inputs:
 *   - arg[0]: initial_states (n_stp * N_STATES x 1) - flattened initial states
 *   - arg[1]: torque_params (n_stp * N_CONTROLS x 1) - flattened control inputs
 *   - arg[2]: delta_t (1 x 1) - integration time step
 * 
 * Outputs:
 *   - res[0]: final_states (n_stp * N_STATES x 1) - flattened final states
 */
class DynamicsCallback : public Callback {
protected:
    std::unique_ptr<DynamicsIntegrator> integrator_;
    bool verbose_ = false;
    Dict opts_ = Dict();
    std::string name_ = "DynamicsCallback";
    
public:
    // Constructor
    DynamicsCallback(std::string name, bool verbose = false, Dict opts = Dict());
    
    // Virtual destructor
    virtual ~DynamicsCallback();
    
    // Required CasADi callback methods
    casadi_int get_n_in() override;
    casadi_int get_n_out() override;
    
    std::string get_name_in(casadi_int i) override;
    std::string get_name_out(casadi_int i) override;
    
    Sparsity get_sparsity_in(casadi_int i) override;
    Sparsity get_sparsity_out(casadi_int i) override;
    
    // Main evaluation function
    std::vector<DM> eval(const std::vector<DM>& arg) const override;
    
// protected:
//     // Helper methods
//     void convertInputsToStructures(const std::vector<DM>& arg,
//                                   std::vector<StateParams>& initial_states,
//                                   std::vector<TorqueParams>& torque_params,
//                                   double& delta_t) const;
    
//     std::vector<double> convertStatesToVector(const std::vector<StateParams>& states) const;
};

// /**
//  * Extended Callback with Jacobian computation (sparse)
//  */
// class DynamicsCallbackWithJacobian : public DynamicsCallback {
// private:
//     // Inner class for Jacobian computation
//     class JacobianCallback : public Callback {
//     private:
//         DynamicsCallbackWithJacobian* parent_;
//         mutable Sparsity jacobian_sparsity_;
//         mutable bool sparsity_computed_;
        
//     public:
//         JacobianCallback(DynamicsCallbackWithJacobian* parent);
        
//         casadi_int get_n_in() override;
//         casadi_int get_n_out() override;
        
//         Sparsity get_sparsity_in(casadi_int i) override;
//         Sparsity get_sparsity_out(casadi_int i) override;
        
//         std::vector<DM> eval(const std::vector<DM>& arg) const override;
        
//     private:
//         void computeJacobianSparsity() const;
//         Sparsity createBlockDiagonalSparsity(int n_blocks, int block_size, const std::vector<int>& block_pattern) const;
//     };
    
//     std::unique_ptr<JacobianCallback> jac_callback_;
    
// public:
//     DynamicsCallbackWithJacobian(int n_stp, bool verbose = false);
    
//     bool has_jacobian() const override;
    
//     Function get_jacobian(const std::string& name, const std::vector<std::string>& inames,
//                          const std::vector<std::string>& onames, const Dict& opts) const override;
// };

// /**
//  * Extended Callback with both Jacobian and Hessian computation (sparse)
//  */
// class DynamicsCallbackWithHessian : public DynamicsCallbackWithJacobian {
// private:
//     // Inner class for Hessian computation  
//     class HessianCallback : public Callback {
//     private:
//         DynamicsCallbackWithHessian* parent_;
//         mutable Sparsity hessian_sparsity_;
//         mutable bool sparsity_computed_;
        
//     public:
//         HessianCallback(DynamicsCallbackWithHessian* parent);
        
//         casadi_int get_n_in() override;
//         casadi_int get_n_out() override;
        
//         Sparsity get_sparsity_in(casadi_int i) override;
//         Sparsity get_sparsity_out(casadi_int i) override;
        
//         std::vector<DM> eval(const std::vector<DM>& arg) const override;
        
//     private:
//         void computeHessianSparsity() const;
//     };
    
//     std::unique_ptr<HessianCallback> hess_callback_;
    
// public:
//     DynamicsCallbackWithHessian(int n_stp, bool verbose = false);
    
//     bool has_hessian() const override;
    
//     Function get_hessian(const std::string& name, const std::vector<std::string>& inames,
//                         const std::vector<std::string>& onames, const Dict& opts) const override;
// };

// Factory functions for easy creation
// void create_dynamics(bool verbose = false);
// Function get_dynamics();
//Function create_spacecraft_dynamics_with_jacobian(int n_stp, bool verbose = false);
//Function create_spacecraft_dynamics_with_hessian(int n_stp, bool verbose = false);
