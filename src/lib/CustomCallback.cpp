#include <toac/dynamics.h>

using namespace casadi;

// Constructor
CUDADynamicsCallback::CUDADynamicsCallback(const std::string& name, int n_states, int n_controls)
    : n_X(n_states), n_U(n_controls) {
    
    // Initialize CUDA integrator
    cuda_integrator = std::make_unique<CUDADynamics>();
    
    // Define callback signature
    construct(name, {});
}


// Required: Get number of inputs
casadi_int CUDADynamicsCallback::get_n_in() {
    return 3;  // X_current, U_current, dt_current
}

// Required: Get number of outputs  
casadi_int CUDADynamicsCallback::get_n_out() {
    return 1;  // Final states
}

// Required: Main evaluation function
std::vector<DM> CUDADynamicsCallback::eval(const std::vector<DM>& arg) const {
    // Input validation
    if (arg.size() != 3) {
        throw std::runtime_error("CUDADynamicsCallback expects exactly 3 inputs");
    }
    
    // Extract inputs
    DM X_current = arg[0];  // Current states (n_X x n_stp)
    DM U_current = arg[1];  // Controls (n_U x n_stp)  
    DM dt_current = arg[2]; // Time steps (n_stp x 1)
    
    // Validate input dimensions
    if (X_current.size1() != n_X || X_current.size2() != n_stp) {
        throw std::runtime_error("X_current has incorrect dimensions");
    }
    if (U_current.size1() != n_U || U_current.size2() != n_stp) {
        throw std::runtime_error("U_current has incorrect dimensions");
    }
    if (dt_current.size1() != n_stp || dt_current.size2() != 1) {
        throw std::runtime_error("dt_current has incorrect dimensions");
    }
    
    // Convert CasADi DM to std::vector<double>
    std::vector<double> initial_states = X_current.get_elements();
    std::vector<double> controls = U_current.get_elements();
    std::vector<double> dt_values = dt_current.get_elements();
    
    // Call CUDA integrator
    std::vector<double> final_states;
    int status = cuda_integrator->integrate_parallel(
        initial_states, controls, dt_values, final_states);
        
    if (status != 0) {
        throw std::runtime_error("CUDA integration failed with status: " + std::to_string(status));
    }
    
    // Convert back to CasADi format and reshape
    DM result = reshape(DM(final_states), n_X, n_stp);
    
    return {result};
}

// Required: Main evaluation function
std::vector<MX> CUDADynamicsCallback::eval(const std::vector<MX>& arg) const {
    // Input validation
    if (arg.size() != 3) {
        throw std::runtime_error("CUDADynamicsCallback expects exactly 3 inputs");
    }
    
    // Extract inputs
    MX X_current = arg[0];  // Current states (n_X x n_stp)
    MX U_current = arg[1];  // Controls (n_U x n_stp)  
    MX dt_current = arg[2]; // Time steps (n_stp x 1)
    
    // Validate input dimensions
    if (X_current.size1() != n_X || X_current.size2() != n_stp) {
        throw std::runtime_error("X_current has incorrect dimensions");
    }
    if (U_current.size1() != n_U || U_current.size2() != n_stp) {
        throw std::runtime_error("U_current has incorrect dimensions");
    }
    if (dt_current.size1() != n_stp || dt_current.size2() != 1) {
        throw std::runtime_error("dt_current has incorrect dimensions");
    }
    
    return {arg[0]};  // Return the current states as output
}

// Factory method to create the callback
Function CUDADynamicsCallback::create_function(const std::string& name, int n_states, int n_controls) {
    // Create callback instance
    auto callback = std::make_shared<CUDADynamicsCallback>(
        name, n_states, n_controls);
    
    // Create CasADi function from callback
    std::vector<MX> inputs = {
        MX::sym("X_current", n_states, n_stp),
        MX::sym("U_current", n_controls, n_stp), 
        MX::sym("dt_current", n_stp, 1)
    };
    
    std::vector<MX> outputs = callback->eval(inputs);
    
    return Function(name, inputs, outputs, 
                    {"X_current", "U_current", "dt_current"}, 
                    {"X_final"});
}