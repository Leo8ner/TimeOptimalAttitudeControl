#include <toac/casadi_callback.hpp>

using namespace casadi;
/**
 * Base Callback for spacecraft dynamics integration
 * 
 * Inputs:
 *   - arg[0]: initial_states (n_stp * n_states x 1) - flattened initial states
 *   - arg[1]: torque_params (n_stp * n_controls x 1) - flattened control inputs
 *   - arg[2]: delta_t (1 x 1) - integration time step
 * 
 * Outputs:
 *   - res[0]: final_states (n_stp * n_states x 1) - flattened final states
 */
class DynamicsCallback : public Callback {
protected:
    std::unique_ptr<OptimizedDynamicsIntegrator> integrator_;
    bool verbose_;
    
public:
    // Constructor
    DynamicsCallback(bool verbose = false) 
        : verbose_(verbose) {
        integrator_ = std::make_unique<OptimizedDynamicsIntegrator>(verbose);
    }
    
    // Virtual destructor
    virtual ~DynamicsCallback() = default;
    
    // Required CasADi callback methods
    casadi_int get_n_in() override { return 3; }
    casadi_int get_n_out() override { return 1; }
    
    std::string get_name_in(casadi_int i) override {
        switch(i) {
            case 0: return "X";
            case 1: return "U"; 
            case 2: return "dt";
            default: return "unknown";
        }
    }
    
    std::string get_name_out(casadi_int i) override {
        switch(i) {
            case 0: return "X_next";
            default: return "unknown";
        }
    }
    
    Sparsity get_sparsity_in(casadi_int i) override {
        switch(i) {
            case 0: // initial_states
                return Sparsity::dense(n_stp * n_states, 1);
            case 1: // torque_params
                return Sparsity::dense(n_stp * n_controls, 1);
            case 2: // delta_t
                return Sparsity::dense(1, 1);
            default:
                casadi_error("Invalid input index");
        }
    }
    
    Sparsity get_sparsity_out(casadi_int i) override {
        switch(i) {
            case 0: // final_states
                return Sparsity::dense(n_stp * n_states, 1);
            default:
                casadi_error("Invalid output index");
        }
    }
    
    // Main evaluation function
    std::vector<DM> eval(const std::vector<DM>& arg) const override {
        if (arg.size() != 3) {
            casadi_error("Expected 3 inputs");
        }
        
        // Extract inputs
        DM initial_states_dm = arg[0];
        DM torque_params_dm = arg[1];
        DM delta_t_dm = arg[2];
        
        // Convert to vectors for C++ integrator
        std::vector<sunrealtype> initial_states_vec = initial_states_dm.get_elements();
        std::vector<sunrealtype> torque_params_vec = torque_params_dm.get_elements();
        sunrealtype delta_t = static_cast<sunrealtype>(delta_t_dm.scalar());
        
        // Convert to StateParams and TorqueParams structures
        std::vector<StateParams> initial_states;
        std::vector<TorqueParams> torque_params;
        
        // Convert flattened vectors to structured parameters
        for (int i = 0; i < n_stp; ++i) {
            StateParams state;
            state.q0 = initial_states_vec[i * n_states + 0];
            state.q1 = initial_states_vec[i * n_states + 1];
            state.q2 = initial_states_vec[i * n_states + 2];
            state.q3 = initial_states_vec[i * n_states + 3];
            state.wx = initial_states_vec[i * n_states + 4];
            state.wy = initial_states_vec[i * n_states + 5];
            state.wz = initial_states_vec[i * n_states + 6];
            initial_states.push_back(state);
            
            TorqueParams torque;
            torque.tau_x = torque_params_vec[i * n_controls + 0];
            torque.tau_y = torque_params_vec[i * n_controls + 1];  
            torque.tau_z = torque_params_vec[i * n_controls + 2];
            torque_params.push_back(torque);
        }
        
        // Call integrator
        int ret_code = integrator_->solve(initial_states, torque_params, delta_t);
        
        if (ret_code != 0) {
            casadi_error("Integration failed with code " + std::to_string(ret_code));
        }
        
        // Get solution and convert back to DM
        std::vector<StateParams> final_states = integrator_->getSolution();
        std::vector<sunrealtype> final_states_vec;
        final_states_vec.reserve(n_stp * n_states);
        
        for (const auto& state : final_states) {
            final_states_vec.push_back(state.q0);
            final_states_vec.push_back(state.q1);
            final_states_vec.push_back(state.q2);
            final_states_vec.push_back(state.q3);
            final_states_vec.push_back(state.wx);
            final_states_vec.push_back(state.wy);
            final_states_vec.push_back(state.wz);
        }
        
        DM result = DM(final_states_vec);
        return {result};
    }
};

/**
 * Extended Callback with Jacobian computation
 */
class DynamicsCallbackWithJacobian : public DynamicsCallback {
private:
    // Inner class for Jacobian computation
    class JacobianCallback : public Callback {
    private:
        DynamicsCallbackWithJacobian* parent_;
        
    public:
        JacobianCallback(DynamicsCallbackWithJacobian* parent) 
            : parent_(parent) {}
            
        casadi_int get_n_in() override { return 4; } // 3 inputs + 1 nominal output
        casadi_int get_n_out() override { return 1; }
        
        Sparsity get_sparsity_in(casadi_int i) override {
            switch(i) {
                case 0: // initial_states
                    return Sparsity::dense(n_stp * n_states, 1);
                case 1: // torque_params
                    return Sparsity::dense(n_stp * n_controls, 1);
                case 2: // delta_t
                    return Sparsity::dense(1, 1);
                case 3: // nominal output
                    return Sparsity(n_stp * n_states, 1);
                default:
                    casadi_error("Invalid input index");
            }
        }
        
        Sparsity get_sparsity_out(casadi_int i) override {
            switch(i) {
                case 0: {
                    // Jacobian sparsity: output_size x input_size
                    int n_outputs = n_stp * n_states;
                    int n_inputs = n_stp * (n_states + n_controls) + 1;
                    
                    // You might want to use actual sparsity pattern from your integrator
                    // For now, using dense pattern
                    return Sparsity::dense(n_outputs, n_inputs);
                }
                default:
                    casadi_error("Invalid output index");
            }
        }
        
        std::vector<DM> eval(const std::vector<DM>& arg) const override {
            if (arg.size() != 4) {
                casadi_error("Expected 4 inputs for Jacobian evaluation");
            }
            
            // Extract inputs (first 3 are the same as main function)
            DM initial_states_dm = arg[0];
            DM torque_params_dm = arg[1]; 
            DM delta_t_dm = arg[2];
            // arg[3] is nominal output (not used in this implementation)
            
            // Convert inputs same as main function
            std::vector<sunrealtype> initial_states_vec = initial_states_dm.get_elements();
            std::vector<sunrealtype> torque_params_vec = torque_params_dm.get_elements();
            sunrealtype delta_t = static_cast<sunrealtype>(delta_t_dm.scalar());
            
            // Convert to structured parameters
            std::vector<StateParams> initial_states;
            std::vector<TorqueParams> torque_params;
            
            for (int i = 0; i < n_stp; ++i) {
                StateParams state;
                state.q0 = initial_states_vec[i * n_states + 0];
                state.q1 = initial_states_vec[i * n_states + 1];
                state.q2 = initial_states_vec[i * n_states + 2];
                state.q3 = initial_states_vec[i * n_states + 3];
                state.wx = initial_states_vec[i * n_states + 4];
                state.wy = initial_states_vec[i * n_states + 5];
                state.wz = initial_states_vec[i * n_states + 6];
                initial_states.push_back(state);
                
                TorqueParams torque;
                torque.tau_x = torque_params_vec[i * n_controls + 0];
                torque.tau_y = torque_params_vec[i * n_controls + 1];
                torque.tau_z = torque_params_vec[i * n_controls + 2];
                torque_params.push_back(torque);
            }
            
            // Call integrator (this should compute both solution and Jacobian)
            int ret_code = parent_->integrator_->solve(initial_states, torque_params, delta_t);
            
            if (ret_code != 0) {
                casadi_error("Integration failed with code " + std::to_string(ret_code));
            }
            
            // Get Jacobian from integrator
            // Option 1: Get sparse Jacobian
            auto [jac_values, row_indices, col_pointers] = parent_->integrator_->getJacobianFull();
            
            int n_outputs = n_stp * n_states;
            int n_inputs = n_stp * (n_states + n_controls) + 1;
            
            // Convert sparse CSR to dense matrix for CasADi
            // (You might want to keep it sparse for efficiency)
            std::vector<sunrealtype> jac_dense(n_outputs * n_inputs, 0.0);
            
            for (int i = 0; i < n_outputs; ++i) {
                for (int j = col_pointers[i]; j < col_pointers[i+1]; ++j) {
                    int col = row_indices[j];
                    jac_dense[i * n_inputs + col] = jac_values[j];
                }
            }
            
            DM jacobian = DM(jac_dense).reshape(n_outputs, n_inputs);
            return {jacobian};
        }
    };
    
    std::unique_ptr<JacobianCallback> jac_callback_;
    
public:
    DynamicsCallbackWithJacobian(bool verbose = false)
        : DynamicsCallback(verbose) {
        jac_callback_ = std::make_unique<JacobianCallback>(this);
    }
    
    bool has_jacobian() const override { return true; }
    
    Function get_jacobian(const std::string& name, const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames, const Dict& opts) const override {
        return jac_callback_->create(name + "_jac", opts);
    }
};

/**
 * Extended Callback with both Jacobian and Hessian computation
 */
class DynamicsCallbackWithHessian : public DynamicsCallbackWithJacobian {
private:
    // Inner class for Hessian computation  
    class HessianCallback : public Callback {
    private:
        DynamicsCallbackWithHessian* parent_;
        
    public:
        HessianCallback(DynamicsCallbackWithHessian* parent)
            : parent_(parent) {}
            
        casadi_int get_n_in() override { return 5; } // 3 inputs + nominal output + output weights
        casadi_int get_n_out() override { return 1; }
        
        Sparsity get_sparsity_in(casadi_int i) override {
            switch(i) {
                case 0: // initial_states
                    return Sparsity::dense(n_stp * n_states, 1);
                case 1: // torque_params
                    return Sparsity::dense(n_stp * n_controls, 1);
                case 2: // delta_t
                    return Sparsity::dense(1, 1);
                case 3: // nominal output
                    return Sparsity(n_stp * n_states, 1);
                case 4: // output weights
                    return Sparsity::dense(n_stp * n_states, 1);
                default:
                    casadi_error("Invalid input index");
            }
        }
        
        Sparsity get_sparsity_out(casadi_int i) override {
            switch(i) {
                case 0: {
                    // Hessian sparsity: input_size x input_size
                    int n_inputs = n_stp * (n_states + n_controls) + 1;
                    return Sparsity::dense(n_inputs, n_inputs);
                }
                default:
                    casadi_error("Invalid output index");
            }
        }
        
        std::vector<DM> eval(const std::vector<DM>& arg) const override {
            // For now, return zero Hessian
            // You'll need to implement Hessian computation in your integrator
            int n_inputs = n_stp * (n_states + n_controls) + 1;
            DM hessian = DM::zeros(n_inputs, n_inputs);
            
            // TODO: Implement actual Hessian computation
            // This would require second-order derivatives from your integrator
            
            return {hessian};
        }
    };
    
    std::unique_ptr<HessianCallback> hess_callback_;
    
public:
    DynamicsCallbackWithHessian(bool verbose = false)
        : DynamicsCallbackWithJacobian(verbose) {
        hess_callback_ = std::make_unique<HessianCallback>(this);
    }
    
    bool has_hessian() const override { return true; }
    
    Function get_hessian(const std::string& name, const std::vector<std::string>& inames,
                        const std::vector<std::string>& onames, const Dict& opts) const override {
        return hess_callback_->create(name + "_hess", opts);
    }
};

// Factory functions for easy creation
inline Function create_spacecraft_dynamics(bool verbose = false) {
    auto callback = std::make_shared<DynamicsCallback>(verbose);
    return callback->create("spacecraft_dynamics");
}

inline Function create_spacecraft_dynamics_with_jacobian(bool verbose = false) {
    auto callback = std::make_shared<DynamicsCallbackWithJacobian>(verbose);
    return callback->create("spacecraft_dynamics_jac");
}

inline Function create_spacecraft_dynamics_with_hessian(bool verbose = false) {
    auto callback = std::make_shared<DynamicsCallbackWithHessian>(verbose);
    return callback->create("spacecraft_dynamics_hess");
}

