#include <toac/casadi_callback.h>

using namespace casadi;


CUDACallback::CUDACallback(const std::string& name) {
    // Initialize the callback with the given name
    construct(name);
    integrator = std::make_unique<OptimizedDynamicsIntegrator>(false);
}

// Input/output dimensions
casadi_int CUDACallback::get_n_in() { return 3; }  // X_current, U, dt
casadi_int CUDACallback::get_n_out() { return 1; } // X_next
    
std::string CUDACallback::get_name_in(casadi_int i) {
    switch(i) {
        case 0: return "X_current";  // 7 x n_stp
        case 1: return "U";          // 3 x n_stp
        case 2: return "dt";         // scalar
        default: return "";
    }
}
    
std::string CUDACallback::get_name_out(casadi_int i) {
    return i == 0 ? "X_next" : "";  // 7 x n_stp
}
    
Sparsity CUDACallback::get_sparsity_in(casadi_int i) {
    switch(i) {
        case 0: return Sparsity::dense(7, n_stp);    // X_current
        case 1: return Sparsity::dense(3, n_stp);    // U
        case 2: return Sparsity::dense(1, 1);        // dt
        default: return Sparsity();
    }
}
    
Sparsity CUDACallback::get_sparsity_out(casadi_int i) {
    return i == 0 ? Sparsity::dense(7, n_stp) : Sparsity();
}

// Main evaluation function - calls batch CUDA integrator
DMVector CUDACallback::eval(const DMVector& arg) const {
    // Extract inputs
    DM X_current = arg[0];  // 7 x n_stp
    DM U = arg[1];          // 3 x n_stp  
    DM dt_scalar = arg[2];  // scalar
    
    double dt_val = static_cast<double>(dt_scalar);
    
    // Convert to batch integrator format
    std::vector<std::vector<sunrealtype>> initial_states(n_stp);
    std::vector<StepParams> step_params(n_stp);
    
    for(int i = 0; i < n_stp; i++) {
        // Extract initial state for step i: [q0, q1, q2, q3, wx, wy, wz]
        initial_states[i].resize(7);
        for(int j = 0; j < 7; j++) {
            initial_states[i][j] = static_cast<sunrealtype>(X_current(j, i));
        }
        
        // Extract control inputs for step i: [tau_x, tau_y, tau_z]
        step_params[i] = StepParams(
            static_cast<sunrealtype>(U(0, i)),  // tau_x
            static_cast<sunrealtype>(U(1, i)),  // tau_y
            static_cast<sunrealtype>(U(2, i))   // tau_z
        );
    }
    
    // Call batch CUDA integrator
    integrator->solve(initial_states, step_params, dt_val);
    
    // Get results and convert back to CasADi format
    auto solutions = integrator->getAllSolutions();
    DM X_next = DM::zeros(7, n_stp);
    
    for(int i = 0; i < n_stp; i++) {
        for(int j = 0; j < 7; j++) {
            X_next(j, i) = solutions[i][j];
        }
    }
    
    return {X_next};
}
