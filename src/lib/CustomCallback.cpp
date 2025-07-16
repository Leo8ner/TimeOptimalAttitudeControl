#include <toac/casadi_callback.h>

using namespace casadi;

// Constructor
DynamicsCallback::DynamicsCallback(std::string name, bool verbose, Dict opts) 
    : name_(name), verbose_(verbose), opts_(opts) {
    integrator_ = std::make_unique<DynamicsIntegrator>(true, verbose_);  // Enable sensitivity
    Callback::construct(name_, opts_);
}

std::string DynamicsCallback::get_name_in(casadi_int i) {
    switch(i) {
        case 0: return "X";
        case 1: return "U"; 
        case 2: return "dt";
        default: return "unknown";
    }
}

std::string DynamicsCallback::get_name_out(casadi_int i) {
    switch(i) {
        case 0: return "X_next";
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
            return Sparsity::dense(n_stp, 1);
        default:
            casadi_error("Invalid input index");
    }
}

Sparsity DynamicsCallback::get_sparsity_out(casadi_int i) {
    switch(i) {
        case 0: // final_states
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
    
    // Extract inputs
    DM initial_states_dm = arg[0];  // n_states x n_stp
    DM torque_params_dm  = arg[1];  // n_controls x n_stp
    DM delta_t_dm        = arg[2];  // 1 x 1
    
    // Validate dimensions
    if (initial_states_dm.size1() != n_states || initial_states_dm.size2() != n_stp) {
        casadi_error("Initial states dimension mismatch");
    }
    
    if (torque_params_dm.size1() != n_controls || torque_params_dm.size2() != n_stp) {
        casadi_error("Torque params dimension mismatch");
    }
    
    // Extract scalar delta_t
    sunrealtype delta_t = static_cast<sunrealtype>(delta_t_dm.scalar());
    
    // Convert matrices to integrator format
    std::vector<std::vector<sunrealtype>> initial_states(n_stp);
    std::vector<std::vector<sunrealtype>> torque_params(n_stp);
    
    for (int i = 0; i < n_stp; ++i) {
        // Extract state for system i (column i)
        initial_states[i].reserve(n_states);
        for (int j = 0; j < n_states; ++j) {
            initial_states[i].push_back(static_cast<sunrealtype>(initial_states_dm(j, i).scalar()));
        }
        
        // Extract torque for system i (column i)
        torque_params[i].reserve(n_controls);
        for (int j = 0; j < n_controls; ++j) {
            torque_params[i].push_back(static_cast<sunrealtype>(torque_params_dm(j, i).scalar()));
        }
    }
    
    // Call integrator with sensitivity enabled
    int ret_code = integrator_->solve(initial_states, torque_params, delta_t, true);
    
    if (ret_code != 0) {
        casadi_error("Integration failed with code " + std::to_string(ret_code));
    }
    
    // Get solution and convert back to matrix format
    std::vector<std::vector<sunrealtype>> final_states = integrator_->getSolution();
    
    // Create output matrix: n_states x n_stp
    DM result = DM::zeros(n_states, n_stp);
    
    for (int i = 0; i < n_stp; ++i) {
        const auto& state = final_states[i];
        for (int j = 0; j < n_states; ++j) {
            result(j, i) = state[j];
        }
    }
    
    return {result};
}

// Create Jacobian callback
Function DynamicsCallback::get_jacobian(const std::string& name, 
                        const std::vector<std::string>& inames,
                        const std::vector<std::string>& onames, 
                        const Dict& opts) const {
    
    // Create and store Jacobian callback
    jac_callback_ = std::make_shared<JacobianCallback>(this, integrator_.get(), name, opts);
    return *jac_callback_;
}

JacobianCallback::JacobianCallback(const DynamicsCallback* parent, const DynamicsIntegrator* integrator, const std::string& name, const Dict& opts)
    : parent_(parent), integrator_(integrator) {
    Callback::construct(name, opts);
}

Sparsity JacobianCallback::get_sparsity_in(casadi_int i) {
    switch(i) {
        case 0: // initial_states
            return Sparsity::dense(n_states, n_stp);
        case 1: // torque_params
            return Sparsity::dense(n_controls, n_stp);
        case 2: // delta_t
            return Sparsity::dense(n_stp, 1);
        case 3: // final_states
            return Sparsity::dense(n_states, n_stp);
        default:
            casadi_error("Invalid input index");
    }
}

Sparsity JacobianCallback::get_sparsity_out(casadi_int i) {
    std::vector<casadi_int> rows, cols;
    
    switch(i) {
        case 0: { // jac_X_next_X: (n_stp*n_states) × (n_stp*n_states)
            for (int sys = 0; sys < n_stp; sys++) {
                int row_offset = sys * n_states;
                int col_offset = sys * n_states;
                
                for (int r = 0; r < n_states; r++) {
                    for (int c = 0; c < n_states; c++) {
                        rows.push_back(row_offset + r);
                        cols.push_back(col_offset + c);
                    }
                }
            }
            return Sparsity::triplet(n_stp * n_states, n_stp * n_states, rows, cols);
        }
        
        case 1: { // jac_X_next_U: (n_stp*n_states) × (n_stp*n_controls)
            for (int sys = 0; sys < n_stp; sys++) {
                int row_offset = sys * n_states;
                int col_offset = sys * n_controls;
                
                for (int r = 0; r < n_states; r++) {
                    for (int c = 0; c < n_controls; c++) {
                        rows.push_back(row_offset + r);
                        cols.push_back(col_offset + c);
                    }
                }
            }
            return Sparsity::triplet(n_stp * n_states, n_stp * n_controls, rows, cols);
        }
        
        case 2: { // jac_X_next_dt: (n_stp*n_states) × n_stp
            for (int sys = 0; sys < n_stp; sys++) {
                int row_offset = sys * n_states;
                
                for (int r = 0; r < n_states; r++) {
                    rows.push_back(row_offset + r);
                    cols.push_back(sys);  // Each system depends on its own dt
                }
            }
            return Sparsity::triplet(n_stp * n_states, n_stp, rows, cols);
        }
        
        default:
            casadi_error("Invalid output index");
    }
}

std::vector<DM> JacobianCallback::eval(const std::vector<DM>& arg) const {
    // Get sensitivities from integrator (assumes eval() was called first)
    auto [csr_data, csr_indices, csr_indptr, n_rows, n_cols] = 
        integrator_->getSensitivities();
    
    // Convert CSR to CasADi triplet format
    auto [rows, cols, values] = csrToCasADiTriplets(csr_data, csr_indices, csr_indptr);
    
    // Create sparse Jacobian matrix
    casadi_int jac_rows = n_stp * n_states;
    casadi_int jac_cols = n_stp * (n_states + n_controls);
    
    Sparsity jac_sparsity = Sparsity::triplet(jac_rows, jac_cols, rows, cols);
    DM jacobian = DM(jac_sparsity, values);
    
    return {jacobian};
}

/**
 * Convert CSR sensitivity data to CasADi sparse format
 * Maps block-diagonal structure to [all_x, all_u] input layout
 */
std::tuple<std::vector<casadi_int>, std::vector<casadi_int>, std::vector<double>>
JacobianCallback::csrToCasADiTriplets(const std::vector<sunrealtype>& csr_data,
                    const std::vector<int>& csr_indices, 
                    const std::vector<int>& csr_indptr) const {
    
    std::vector<casadi_int> rows, cols;
    std::vector<double> values;
    
    // Reserve space for efficiency
    rows.reserve(csr_data.size());
    cols.reserve(csr_data.size());
    values.reserve(csr_data.size());
    
    // Convert CSR to triplet format with proper indexing for [all_x, all_u] layout
    for (int global_row = 0; global_row < n_stp * n_states; global_row++) {
        int start = csr_indptr[global_row];
        int end = csr_indptr[global_row + 1];
        
        for (int idx = start; idx < end; idx++) {
            int param_idx = csr_indices[idx];
            sunrealtype value = csr_data[idx];
            
            if (std::abs(value) > 1e-15) {  // Skip numerical zeros
                // Decode parameter info
                int param_sys = param_idx / (n_states + n_controls);
                int param_type = param_idx % (n_states + n_controls);
                
                // Map to CasADi column index for [all_x, all_u] layout:
                // all_x: [0, n_stp * n_states)
                // all_u: [n_stp * n_states, n_stp * (n_states + n_controls))
                casadi_int casadi_col;
                if (param_type < n_states) {
                    // Initial condition parameter: maps to all_x section
                    casadi_col = param_sys * n_states + param_type;
                } else {
                    // Control parameter: maps to all_u section  
                    int control_idx = param_type - n_states;
                    casadi_col = n_stp * n_states + param_sys * n_controls + control_idx;
                }
                
                rows.push_back(global_row);
                cols.push_back(casadi_col);
                values.push_back(static_cast<double>(value));
            }
        }
    }
    
    return std::make_tuple(std::move(rows), std::move(cols), std::move(values));
}