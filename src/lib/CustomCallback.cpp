#include <toac/casadi_callback.h>

using namespace casadi;
/**
 * Base Callback for spacecraft dynamics integration
 * 
 * Inputs:
 *   - arg[0]: initial_states (n_stp x n_states)  - flattened initial states
 *   - arg[1]: torque_params (n_stp x n_controls) - flattened control inputs
 *   - arg[2]: delta_t (1 x 1)                    - integration time step
 * 
 * Outputs:
 *   - res[0]: final_states (n_stp x n_states)    - flattened final states
 */

// Constructor
DynamicsCallback::DynamicsCallback(std::string name, bool verbose, Dict opts) 
    : name_(name), verbose_(verbose), opts_(opts) {
    integrator_ = std::make_unique<DynamicsIntegrator>(verbose_);
    Callback::construct(name_, opts_);
}

DynamicsCallback::~DynamicsCallback() {
    // Can be empty, but must be defined
}

// Required CasADi callback methods
casadi_int DynamicsCallback::get_n_in() { return 3; }
casadi_int DynamicsCallback::get_n_out() { return 1; }

std::string DynamicsCallback::get_name_in(casadi_int i) {
    switch(i) {
        case  0: return "X";
        case  1: return "U"; 
        case  2: return "dt";
        default: return "unknown";
    }
}

std::string DynamicsCallback::get_name_out(casadi_int i) {
    switch(i) {
        case  0: return "X_next";
        default: return "unknown";
    }
}

Sparsity DynamicsCallback::get_sparsity_in(casadi_int i) {
    switch(i) {
        case 0: // initial_states
            return Sparsity::dense(n_states, n_stp);
        case 1: // torque_params
            return Sparsity::dense(n_controls, n_stp);
        case 2: // delta_t
            return Sparsity::dense(1, 1);
        default:
            casadi_error("Invalid input index");
    }
}

Sparsity DynamicsCallback::get_sparsity_out(casadi_int i) {
    switch(i) {
        case  0: // final_states
            return Sparsity::dense(n_states, n_stp);
        default:
            casadi_error("Invalid output index");
    }
}

// Main evaluation function
std::vector<DM> DynamicsCallback::eval(const std::vector<DM>& arg) const {
    if (arg.size() != 3) {
        casadi_error("Expected 3 inputs");
    }
    
    // Extract inputs - now they are matrices, not flattened vectors
    DM initial_states_dm = arg[0];  // n_states x n_stp
    DM torque_params_dm  = arg[1];   // n_controls x n_stp
    DM delta_t_dm        = arg[2];   // 1 x 1
    
    // Check dimensions
    if (initial_states_dm.size1() != n_states || initial_states_dm.size2() != n_stp) {
        casadi_error("Initial states dimension mismatch. Expected " + 
                    std::to_string(n_states) + "x" + std::to_string(n_stp) + 
                    ", got " + std::to_string(initial_states_dm.size1()) + "x" + 
                    std::to_string(initial_states_dm.size2()));
    }
    
    if (torque_params_dm.size1() != n_controls || torque_params_dm.size2() != n_stp) {
        casadi_error("Torque params dimension mismatch. Expected " + 
                    std::to_string(n_controls) + "x" + std::to_string(n_stp) + 
                    ", got " + std::to_string(torque_params_dm.size1()) + "x" + 
                    std::to_string(torque_params_dm.size2()));
    }
    
    // Extract scalar delta_t
    sunrealtype delta_t = static_cast<sunrealtype>(delta_t_dm.scalar());
    
    // Convert matrices to structured parameters
    std::vector<StateParams> initial_states;
    std::vector<TorqueParams> torque_params;
    
    initial_states.reserve(n_stp);
    torque_params.reserve(n_stp);
    
    // Process each time step (each column)
    for (int i = 0; i < n_stp; ++i) {
        // Extract state for step i (column i)
        StateParams state;
        state.q0 = static_cast<sunrealtype>(initial_states_dm(0, i).scalar());
        state.q1 = static_cast<sunrealtype>(initial_states_dm(1, i).scalar());
        state.q2 = static_cast<sunrealtype>(initial_states_dm(2, i).scalar());
        state.q3 = static_cast<sunrealtype>(initial_states_dm(3, i).scalar());
        state.wx = static_cast<sunrealtype>(initial_states_dm(4, i).scalar());
        state.wy = static_cast<sunrealtype>(initial_states_dm(5, i).scalar());
        state.wz = static_cast<sunrealtype>(initial_states_dm(6, i).scalar());
        initial_states.push_back(state);
        
        // Extract torque for step i (column i)
        TorqueParams torque;
        torque.tau_x = static_cast<sunrealtype>(torque_params_dm(0, i).scalar());
        torque.tau_y = static_cast<sunrealtype>(torque_params_dm(1, i).scalar());
        torque.tau_z = static_cast<sunrealtype>(torque_params_dm(2, i).scalar());
        torque_params.push_back(torque);
    }
    
    // Call integrator
    int ret_code = integrator_->solve(initial_states, torque_params, delta_t);
    
    if (ret_code != 0) {
        casadi_error("Integration failed with code " + std::to_string(ret_code));
    }
    
    // Get solution and convert back to matrix format
    std::vector<StateParams> final_states = integrator_->getSolution();
    
    // Create output matrix: n_states x n_stp
    DM result = DM::zeros(n_states, n_stp);
    
    for (int i = 0; i < n_stp; ++i) {
        const auto& state = final_states[i];
        result(0, i) = state.q0;
        result(1, i) = state.q1;
        result(2, i) = state.q2;
        result(3, i) = state.q3;
        result(4, i) = state.wx;
        result(5, i) = state.wy;
        result(6, i) = state.wz;
    }
    
    return {result};
}

// /**
//  * Extended Callback with Jacobian computation
//  */
// class DynamicsCallbackWithJacobian : public DynamicsCallback {
// private:
//     // Inner class for Jacobian computation
//     class JacobianCallback : public Callback {
//     private:
//         DynamicsCallbackWithJacobian* parent_;
        
//     public:
//         JacobianCallback(DynamicsCallbackWithJacobian* parent) 
//             : parent_(parent) {}
            
//         casadi_int get_n_in() override { return 4; } // 3 inputs + 1 nominal output
//         casadi_int get_n_out() override { return 1; }
        
//         Sparsity get_sparsity_in(casadi_int i) override {
//             switch(i) {
//                 case 0: // initial_states
//                     return Sparsity::dense(n_stp * n_states, 1);
//                 case 1: // torque_params
//                     return Sparsity::dense(n_stp * n_controls, 1);
//                 case 2: // delta_t
//                     return Sparsity::dense(1, 1);
//                 case 3: // nominal output
//                     return Sparsity(n_stp * n_states, 1);
//                 default:
//                     casadi_error("Invalid input index");
//             }
//         }
        
//         Sparsity get_sparsity_out(casadi_int i) override {
//             switch(i) {
//                 case 0: {
//                     // Jacobian sparsity: output_size x input_size
//                     int n_outputs = n_stp * n_states;
//                     int n_inputs = n_stp * (n_states + n_controls) + 1;
                    
//                     // You might want to use actual sparsity pattern from your integrator
//                     // For now, using dense pattern
//                     return Sparsity::dense(n_outputs, n_inputs);
//                 }
//                 default:
//                     casadi_error("Invalid output index");
//             }
//         }
        
//         std::vector<DM> eval(const std::vector<DM>& arg) const override {
//             if (arg.size() != 4) {
//                 casadi_error("Expected 4 inputs for Jacobian evaluation");
//             }
            
//             // Extract inputs (first 3 are the same as main function)
//             DM initial_states_dm = arg[0];
//             DM torque_params_dm = arg[1]; 
//             DM delta_t_dm = arg[2];
//             // arg[3] is nominal output (not used in this implementation)
            
//             // Convert inputs same as main function
//             std::vector<sunrealtype> initial_states_vec = initial_states_dm.get_elements();
//             std::vector<sunrealtype> torque_params_vec = torque_params_dm.get_elements();
//             sunrealtype delta_t = static_cast<sunrealtype>(delta_t_dm.scalar());
            
//             // Convert to structured parameters
//             std::vector<StateParams> initial_states;
//             std::vector<TorqueParams> torque_params;
            
//             for (int i = 0; i < n_stp; ++i) {
//                 StateParams state;
//                 state.q0 = initial_states_vec[i * n_states + 0];
//                 state.q1 = initial_states_vec[i * n_states + 1];
//                 state.q2 = initial_states_vec[i * n_states + 2];
//                 state.q3 = initial_states_vec[i * n_states + 3];
//                 state.wx = initial_states_vec[i * n_states + 4];
//                 state.wy = initial_states_vec[i * n_states + 5];
//                 state.wz = initial_states_vec[i * n_states + 6];
//                 initial_states.push_back(state);
                
//                 TorqueParams torque;
//                 torque.tau_x = torque_params_vec[i * n_controls + 0];
//                 torque.tau_y = torque_params_vec[i * n_controls + 1];
//                 torque.tau_z = torque_params_vec[i * n_controls + 2];
//                 torque_params.push_back(torque);
//             }
            
//             // Call integrator (this should compute both solution and Jacobian)
//             int ret_code = parent_->integrator_->solve(initial_states, torque_params, delta_t);
            
//             if (ret_code != 0) {
//                 casadi_error("Integration failed with code " + std::to_string(ret_code));
//             }
            
//             // Get Jacobian from integrator
//             // Option 1: Get sparse Jacobian
//             auto [jac_values, row_indices, col_pointers] = parent_->integrator_->getJacobianFull();
            
//             int n_outputs = n_stp * n_states;
//             int n_inputs = n_stp * (n_states + n_controls) + 1;
            
//             // Convert sparse CSR to dense matrix for CasADi
//             // (You might want to keep it sparse for efficiency)
//             std::vector<sunrealtype> jac_dense(n_outputs * n_inputs, 0.0);
            
//             for (int i = 0; i < n_outputs; ++i) {
//                 for (int j = col_pointers[i]; j < col_pointers[i+1]; ++j) {
//                     int col = row_indices[j];
//                     jac_dense[i * n_inputs + col] = jac_values[j];
//                 }
//             }
            
//             DM jacobian = DM(jac_dense).reshape(n_outputs, n_inputs);
//             return {jacobian};
//         }
//     };
    
//     std::unique_ptr<JacobianCallback> jac_callback_;
    
// public:
//     DynamicsCallbackWithJacobian(bool verbose = false)
//         : DynamicsCallback(verbose) {
//         jac_callback_ = std::make_unique<JacobianCallback>(this);
//     }
    
//     bool has_jacobian() const override { return true; }
    
//     Function get_jacobian(const std::string& name, const std::vector<std::string>& inames,
//                          const std::vector<std::string>& onames, const Dict& opts) const override {
//         return jac_callback_->create(name + "_jac", opts);
//     }
// };

// /**
//  * Extended Callback with both Jacobian and Hessian computation
//  */
// class DynamicsCallbackWithHessian : public DynamicsCallbackWithJacobian {
// private:
//     // Inner class for Hessian computation  
//     class HessianCallback : public Callback {
//     private:
//         DynamicsCallbackWithHessian* parent_;
        
//     public:
//         HessianCallback(DynamicsCallbackWithHessian* parent)
//             : parent_(parent) {}
            
//         casadi_int get_n_in() override { return 5; } // 3 inputs + nominal output + output weights
//         casadi_int get_n_out() override { return 1; }
        
//         Sparsity get_sparsity_in(casadi_int i) override {
//             switch(i) {
//                 case 0: // initial_states
//                     return Sparsity::dense(n_stp * n_states, 1);
//                 case 1: // torque_params
//                     return Sparsity::dense(n_stp * n_controls, 1);
//                 case 2: // delta_t
//                     return Sparsity::dense(1, 1);
//                 case 3: // nominal output
//                     return Sparsity(n_stp * n_states, 1);
//                 case 4: // output weights
//                     return Sparsity::dense(n_stp * n_states, 1);
//                 default:
//                     casadi_error("Invalid input index");
//             }
//         }
        
//         Sparsity get_sparsity_out(casadi_int i) override {
//             switch(i) {
//                 case 0: {
//                     // Hessian sparsity: input_size x input_size
//                     int n_inputs = n_stp * (n_states + n_controls) + 1;
//                     return Sparsity::dense(n_inputs, n_inputs);
//                 }
//                 default:
//                     casadi_error("Invalid output index");
//             }
//         }
        
//         std::vector<DM> eval(const std::vector<DM>& arg) const override {
//             // For now, return zero Hessian
//             // You'll need to implement Hessian computation in your integrator
//             int n_inputs = n_stp * (n_states + n_controls) + 1;
//             DM hessian = DM::zeros(n_inputs, n_inputs);
            
//             // TODO: Implement actual Hessian computation
//             // This would require second-order derivatives from your integrator
            
//             return {hessian};
//         }
//     };
    
//     std::unique_ptr<HessianCallback> hess_callback_;
    
// public:
//     DynamicsCallbackWithHessian(bool verbose = false)
//         : DynamicsCallbackWithJacobian(verbose) {
//         hess_callback_ = std::make_unique<HessianCallback>(this);
//     }
    
//     bool has_hessian() const override { return true; }
    
//     Function get_hessian(const std::string& name, const std::vector<std::string>& inames,
//                         const std::vector<std::string>& onames, const Dict& opts) const override {
//         return hess_callback_->create(name + "_hess", opts);
//     }
// };

// void create_dynamics(bool verbose) {
//     std::cout << "Step 1: Creating callback instance..." << std::endl;
//     Function F = DynamicsCallback("F", verbose);
//     std::cout << "Step 1 completed." << F << std::endl;
//     Function F_ext = external("F", F);
//     try {        
//         std::cout << "Dynamics function created." << std::endl;
        
//         // Options for C code generation
//         Dict opts = Dict();
//         opts["cpp"] = false;
//         opts["with_header"] = true;
//         opts["with_mem"] = true;  // Include memory management
//         opts["main"] = false;     // Don't generate main function
        
//         // Get paths for code generation
//         std::string prefix_code = std::filesystem::current_path().parent_path().string() + "/code_gen/";
//         std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        
//         // Create directories if they don't exist
//         std::filesystem::create_directories(prefix_code);
//         std::filesystem::create_directories(prefix_lib);
        
//         // Generate C code
//         CodeGenerator myCodeGen = CodeGenerator("dynamics.c", opts);
//         std::cout << "Step 2: Generating C code..." << std::endl;
//         myCodeGen.add(F);
//         std::cout << "Step 2: C code generation started..." << std::endl;
//         myCodeGen.generate(prefix_code);
//         std::cout << "Step 2: C code generated at: " << prefix_code << "dynamics.c" << std::endl;
        
//         // Compile C code to a shared library
//         std::string compile_command = "gcc -fPIC -shared -O3 " +
//                                     prefix_code + "dynamics.c -o " +
//                                     prefix_lib + "lib_dynamics.so";
        
//         std::cout << "Compilation command: " << compile_command << std::endl;
        
//         int compile_flag = std::system(compile_command.c_str());
//         casadi_assert(compile_flag == 0, "Compilation failed!");
        
//         std::cout << "Compilation succeeded!" << std::endl;
//         std::cout << "Generated files:" << std::endl;
//         std::cout << "  - C code: " << prefix_code << "dynamics.c" << std::endl;
//         std::cout << "  - Header: " << prefix_code << "dynamics.h" << std::endl;
//         std::cout << "  - Library: " << prefix_lib << "lib_dynamics.so" << std::endl;
        
//         return;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error in steps 3-4: " << e.what() << std::endl;
//         throw;
//     }
// }

// // Function to load the generated shared library
// Function get_dynamics() {
//     std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
//     std::string lib_path = prefix_lib + "lib_dynamics.so";
    
//     // Check if library exists
//     if (!std::filesystem::exists(lib_path)) {
//         casadi_error("Shared library not found: " + lib_path);
//     }
    
//     // Load the external function
//     Function F = external("dynamics", lib_path);
    
//     std::cout << "Loaded dynamics function from: " << lib_path << std::endl;
    
//     return F;
// }

// inline Function create_spacecraft_dynamics_with_jacobian(bool verbose = false) {
//     auto callback = std::make_shared<DynamicsCallbackWithJacobian>(verbose);
//     return callback->create("spacecraft_dynamics_jac");
// }

// inline Function create_spacecraft_dynamics_with_hessian(bool verbose = false) {
//     auto callback = std::make_shared<DynamicsCallbackWithHessian>(verbose);
//     return callback->create("spacecraft_dynamics_hess");
// }

