#include <toac/casadi_callback.h>

using namespace casadi;

//==============================================================================
// DYNAMICS CALLBACK IMPLEMENTATION
//==============================================================================

/**
 * Constructor: Initialize CasADi callback with GPU-accelerated dynamics integrator
 *
 * Creates a DynamicsIntegrator instance with sensitivity analysis enabled for
 * Jacobian computation. The integrator is configured for batch processing of
 * spacecraft attitude dynamics with sparse linear algebra acceleration.
 *
 * @param name Unique identifier for the callback function in CasADi
 * @param opts Additional CasADi configuration options
 */
DynamicsCallback::DynamicsCallback(std::string name, Dict opts) 
    : name_(name), opts_(opts) {

    // Initialize GPU integrator with sensitivity analysis capability
    // This enables both forward dynamics and Jacobian computation
    integrator_ = std::make_unique<DynamicsIntegrator>(true);
    
    // Register this callback with CasADi's callback system
    Callback::construct(name_, opts_);
}

/**
 * Get human-readable name for input argument
 *
 * Provides symbolic names for debugging and code generation.
 *
 * @param i Input index (0-2)
 * @return Descriptive string for the input
 */
std::string DynamicsCallback::get_name_in(casadi_int i) {
    switch(i) {
        case 0: return "X";    // Initial states [q0,q1,q2,q3,wx,wy,wz] × n_stp
        case 1: return "U";    // Control torques [tau_x,tau_y,tau_z] × n_stp  
        case 2: return "dt";   // Integration time steps × n_stp
        default: return "unknown";
    }
}

/**
 * Get human-readable name for output argument
 *
 * @param i Output index (0 only)
 * @return Descriptive string for the output
 */
std::string DynamicsCallback::get_name_out(casadi_int i) {
    switch(i) {
        case 0: return "X_next";  // Final states after integration
        default: return "unknown";
    }
}

/**
 * Define sparsity pattern for input matrices
 *
 * All inputs are dense matrices with dimensions determined by the number
 * of spacecraft systems and state/control dimensions.
 *
 * @param i Input index
 * @return Dense sparsity pattern with appropriate dimensions
 */
Sparsity DynamicsCallback::get_sparsity_in(casadi_int i) {
    switch(i) {
        case 0: // initial_states: 7 states × n_stp systems
            return Sparsity::dense(n_states, n_stp);
        case 1: // torque_params: 3 controls × n_stp systems
            return Sparsity::dense(n_controls, n_stp);
        case 2: // delta_t: n_stp time steps × 1
            return Sparsity::dense(n_stp, 1);
        default:
            casadi_error("Invalid input index");
    }
}

/**
 * Define sparsity pattern for output matrices
 *
 * @param i Output index
 * @return Dense sparsity pattern for final states
 */
Sparsity DynamicsCallback::get_sparsity_out(casadi_int i) {
    switch(i) {
        case 0: // final_states: 7 states × n_stp systems
            return Sparsity::dense(n_states, n_stp);
        default:
            casadi_error("Invalid output index");
    }
}

/**
 * Convert CasADi DM matrix to integrator-compatible array format
 *
 * CasADi uses column-major storage while the integrator expects a specific
 * memory layout. This function performs the necessary transformation.
 *
 * Memory layout transformation:
 * CasADi: [col0, col1, col2, ...] (column-major)
 * → Integrator: [sys0_states, sys1_states, ...] (system-major)
 *
 * @param dm Input CasADi matrix
 * @param output Pre-allocated output array (caller-managed)
 */
void DynamicsCallback::convertDMtoArray(const DM& dm, double* output) const {
    // Convert column-major to system-major layout
    int idx = 0;
    for (casadi_int i = 0; i < dm.size2(); ++i) {        // For each system
        for (casadi_int j = 0; j < dm.size1(); ++j) {    // For each state/control
            output[idx++] = dm(j, i).scalar();
        }
    }
}

/**
 * Convert integrator array format back to CasADi DM matrix
 *
 * @param data Input array in integrator format
 * @param rows Number of rows in output matrix
 * @param cols Number of columns in output matrix
 * @return CasADi DM matrix in column-major format
 */
DM DynamicsCallback::convertArrayToDM(const double* data, int rows, int cols) const {
    DM result = DM::zeros(rows, cols);
    
    // Transform from system-major to column-major layout
    int idx = 0;
    for (int i = 0; i < cols; ++i) {        // For each system
        for (int j = 0; j < rows; ++j) {    // For each state
            result(j, i) = data[idx++];
        }
    }
    
    return result;
}

/**
 * Main evaluation function: Perform spacecraft dynamics integration
 *
 * This is the core function called by CasADi during optimization. It:
 * 1. Validates input dimensions
 * 2. Converts CasADi matrices to integrator format
 * 3. Calls GPU-accelerated integration
 * 4. Converts results back to CasADi format
 *
 * @param arg Vector of input DM matrices [states, controls, time]
 * @return Vector containing single DM matrix with final states
 * @throws casadi_error on dimension mismatch or integration failure
 */
std::vector<DM> DynamicsCallback::eval(const std::vector<DM>& arg) const {
    // Validate number of input arguments
    if (arg.size() != 3) {
        casadi_error("Expected 3 inputs: [initial_states, torque_params, delta_t]");
    }
    
    // Extract input matrices with descriptive names
    DM initial_states_dm = arg[0];  // n_states × n_stp
    DM torque_params_dm  = arg[1];  // n_controls × n_stp
    DM delta_t_dm        = arg[2];  // n_stp × 1
    
    // Validate input dimensions to prevent runtime errors
    if (initial_states_dm.size1() != n_states || initial_states_dm.size2() != n_stp) {
        casadi_error("Initial states dimension mismatch: expected " + 
                    std::to_string(n_states) + "×" + std::to_string(n_stp) +
                    ", got " + std::to_string(initial_states_dm.size1()) + "×" + 
                    std::to_string(initial_states_dm.size2()));
    }
    
    if (torque_params_dm.size1() != n_controls || torque_params_dm.size2() != n_stp) {
        casadi_error("Torque params dimension mismatch: expected " + 
                    std::to_string(n_controls) + "×" + std::to_string(n_stp) +
                    ", got " + std::to_string(torque_params_dm.size1()) + "×" + 
                    std::to_string(torque_params_dm.size2()));
    }
    
    if (delta_t_dm.size1() != n_stp || delta_t_dm.size2() != 1) {
        casadi_error("Delta time dimension mismatch: expected " + 
                    std::to_string(n_stp) + "×1" +
                    ", got " + std::to_string(delta_t_dm.size1()) + "×" + 
                    std::to_string(delta_t_dm.size2()));
    }
    
    // Allocate arrays for integrator input (caller-managed memory)
    double* initial_states = new double[n_states * n_stp];
    double* torque_params = new double[n_controls * n_stp];
    double* delta_t = new double[n_stp];
    
    try {
        // Convert CasADi matrices to integrator-compatible arrays
        convertDMtoArray(initial_states_dm, initial_states);
        convertDMtoArray(torque_params_dm, torque_params);
        convertDMtoArray(delta_t_dm, delta_t);

        // Perform GPU-accelerated integration (sensitivity disabled for forward evaluation)
        int ret_code = integrator_->solve(initial_states, torque_params, delta_t, false);
        
        if (ret_code != 0) {
            casadi_error("Integration failed with error code " + std::to_string(ret_code) + 
                        ". Check initial conditions and integration parameters.");
        }
        
        // Allocate array for solution (caller-managed memory)
        double* final_states = new double[n_states * n_stp];
        
        try {
            // Retrieve solution from GPU memory
            ret_code = integrator_->getSolution(final_states);
            
            if (ret_code != 0) {
                casadi_error("Failed to retrieve solution with error code " + std::to_string(ret_code));
            }
            
            // Convert back to CasADi matrix format: n_states × n_stp
            DM result = convertArrayToDM(final_states, n_states, n_stp);
            
            // Cleanup output array
            delete[] final_states;
            
            // Cleanup input arrays
            delete[] initial_states;
            delete[] torque_params;
            delete[] delta_t;
            
            return {result};
            
        } catch (...) {
            delete[] final_states;
            throw;
        }
        
    } catch (...) {
        // Cleanup input arrays on exception
        delete[] initial_states;
        delete[] torque_params;
        delete[] delta_t;
        throw;
    }
}

/**
 * Create analytical Jacobian callback function
 *
 * Constructs a specialized callback that computes sparse Jacobian matrices
 * using forward sensitivity analysis. The Jacobian callback shares the same
 * integrator instance to ensure consistency.
 *
 * @param name Unique name for the Jacobian function
 * @param inames Input argument names (for code generation)
 * @param onames Output argument names (for code generation)
 * @param opts CasADi options specific to Jacobian computation
 * @return CasADi Function object that computes analytical Jacobians
 */
Function DynamicsCallback::get_jacobian(const std::string& name, 
                        const std::vector<std::string>& inames,
                        const std::vector<std::string>& onames, 
                        const Dict& opts) const {
    
    // Create shared Jacobian callback to maintain proper lifetime
    jac_callback_ = std::make_shared<JacobianCallback>(this, integrator_.get(), name, opts);
    return *jac_callback_;
}

//==============================================================================
// JACOBIAN CALLBACK IMPLEMENTATION
//==============================================================================

/**
 * Constructor: Initialize Jacobian callback with sparse pattern setup
 *
 * Sets up pre-computed sparsity patterns for efficient Jacobian computation.
 * The patterns are determined by the physical coupling structure of spacecraft
 * dynamics equations.
 *
 * @param parent Pointer to parent dynamics callback
 * @param integrator Pointer to GPU integrator (shared with parent)
 * @param name Function name for CasADi registration
 * @param opts CasADi configuration options
 */
JacobianCallback::JacobianCallback(const DynamicsCallback* parent, 
                                  DynamicsIntegrator* integrator, 
                                  const std::string& name, 
                                  const Dict& opts)
    : parent_(parent), integrator_(integrator) {
    
    // Initialize sparse Jacobian patterns from integrator
    setupSparsity();
    
    // Register with CasADi callback system
    Callback::construct(name, opts);
}

/**
 * Define sparsity patterns for Jacobian function inputs
 *
 * Jacobian function takes the same inputs as the dynamics function plus
 * the final states (required by CasADi's automatic differentiation system).
 *
 * @param i Input index (0-3)
 * @return Dense sparsity pattern for each input
 */
Sparsity JacobianCallback::get_sparsity_in(casadi_int i) {
    switch(i) {
        case 0: // initial_states
            return Sparsity::dense(n_states, n_stp);
        case 1: // torque_params
            return Sparsity::dense(n_controls, n_stp);
        case 2: // delta_t
            return Sparsity::dense(n_stp, 1);
        case 3: // final_states (required by CasADi for Jacobian computation)
            return Sparsity::dense(n_states, n_stp);
        default:
            casadi_error("Invalid input index");
    }
}

/**
 * Initialize sparse Jacobian patterns from GPU integrator
 *
 * Retrieves pre-computed sparsity structures that reflect the physical
 * coupling in spacecraft dynamics:
 * - Block-diagonal structure (systems are independent)
 * - Quaternion-angular velocity coupling within each system
 * - Direct control influence on angular accelerations
 */
void JacobianCallback::setupSparsity() {
    // Get sparsity sizes first
    int y0_nnz = integrator_->getSparsitySizeY0();
    int u_nnz = integrator_->getSparsitySizeU();
    int dt_nnz = integrator_->getSparsitySizeDt();
    
    if (y0_nnz < 0 || u_nnz < 0 || dt_nnz < 0) {
        casadi_error("Failed to get sparsity sizes from integrator");
    }
    
    // Allocate arrays for sparsity patterns
    casadi_int* y0_row_indices = new casadi_int[y0_nnz];
    casadi_int* y0_col_pointers = new casadi_int[n_stp * n_states + 1];
    
    casadi_int* u_row_indices = new casadi_int[u_nnz];
    casadi_int* u_col_pointers = new casadi_int[n_stp * n_controls + 1];
    
    casadi_int* dt_row_indices = new casadi_int[dt_nnz];
    casadi_int* dt_col_pointers = new casadi_int[n_stp + 1];
    
    try {
        
        int ret_y0 = integrator_->getSparsityY0(y0_row_indices, y0_col_pointers);
        int ret_u = integrator_->getSparsityU(u_row_indices, u_col_pointers);
        int ret_dt = integrator_->getSparsityDt(dt_row_indices, dt_col_pointers);
        
        if (ret_y0 != 0 || ret_u != 0 || ret_dt != 0) {
            casadi_error("Failed to retrieve sparsity patterns from integrator");
        }
        
        // Convert to std::vector for CasADi sparsity constructor
        std::vector<casadi_int> y0_col_ptr(y0_col_pointers, y0_col_pointers + n_stp * n_states + 1);
        std::vector<casadi_int> y0_row_idx(y0_row_indices, y0_row_indices + y0_nnz);
        
        std::vector<casadi_int> u_col_ptr(u_col_pointers, u_col_pointers + n_stp * n_controls + 1);
        std::vector<casadi_int> u_row_idx(u_row_indices, u_row_indices + u_nnz);
        
        std::vector<casadi_int> dt_col_ptr(dt_col_pointers, dt_col_pointers + n_stp + 1);
        std::vector<casadi_int> dt_row_idx(dt_row_indices, dt_row_indices + dt_nnz);
        
        // Construct CasADi sparsity objects using Compressed Column Storage (CCS) format
        Jac_y0 = Sparsity(n_stp * n_states, n_stp * n_states, y0_col_ptr, y0_row_idx);
        Jac_u = Sparsity(n_stp * n_states, n_stp * n_controls, u_col_ptr, u_row_idx);
        Jac_dt = Sparsity(n_stp * n_states, n_stp, dt_col_ptr, dt_row_idx);
        
        // Cleanup allocated arrays
        delete[] y0_row_indices;
        delete[] y0_col_pointers;
        delete[] u_row_indices;
        delete[] u_col_pointers;
        delete[] dt_row_indices;
        delete[] dt_col_pointers;
        
    } catch (...) {
        // Cleanup on exception
        delete[] y0_row_indices;
        delete[] y0_col_pointers;
        delete[] u_row_indices;
        delete[] u_col_pointers;
        delete[] dt_row_indices;
        delete[] dt_col_pointers;
        throw;
    }
}

/**
 * Define sparsity patterns for Jacobian function outputs
 *
 * Returns the pre-computed sparse patterns for each Jacobian matrix.
 *
 * @param i Output index (0-2)
 * @return Sparse pattern for the corresponding Jacobian matrix
 */
Sparsity JacobianCallback::get_sparsity_out(casadi_int i) {
    switch(i) {
        case 0: // ∂X_next/∂X: (n_stp×n_states) × (n_stp×n_states)
            return Jac_y0;
        case 1: // ∂X_next/∂U: (n_stp×n_states) × (n_stp×n_controls)
            return Jac_u;
        case 2: // ∂X_next/∂dt: (n_stp×n_states) × n_stp
            return Jac_dt;
        default:
            casadi_error("Invalid output index");
    }
}

/**
 * Compute analytical Jacobian matrices using forward sensitivity analysis
 *
 * This function performs the core Jacobian computation by:
 * 1. Running integration with sensitivity analysis enabled
 * 2. Extracting sensitivity data in sparse CCS format
 * 3. Constructing CasADi sparse matrices with proper sparsity patterns
 *
 * The computed Jacobians are exact (not finite-difference approximations)
 * and maintain the sparse structure for computational efficiency.
 *
 * @param arg Input arguments [states, controls, time, final_states]
 * @return Vector of three sparse Jacobian matrices
 * @throws casadi_error on input validation or integration failure
 */
std::vector<DM> JacobianCallback::eval(const std::vector<DM>& arg) const {
    // Validate number of input arguments
    if (arg.size() != 4) {
        casadi_error("Expected 4 inputs for Jacobian computation");
    }
    
    // Extract inputs (final_states not used but required by CasADi interface)
    DM initial_states_dm = arg[0];  // n_states × n_stp
    DM torque_params_dm  = arg[1];  // n_controls × n_stp
    DM delta_t_dm        = arg[2];  // n_stp × 1
    // arg[3] is final_states - not used in computation
    
    // Validate input dimensions (same checks as parent callback)
    if (initial_states_dm.size1() != n_states || initial_states_dm.size2() != n_stp) {
        casadi_error("Initial states dimension mismatch in Jacobian computation");
    }
    
    if (torque_params_dm.size1() != n_controls || torque_params_dm.size2() != n_stp) {
        casadi_error("Torque params dimension mismatch in Jacobian computation");
    }
    
    if (delta_t_dm.size1() != n_stp || delta_t_dm.size2() != 1) {
        casadi_error("Delta time dimension mismatch in Jacobian computation");
    }
    
    // Allocate arrays for integrator input (caller-managed memory)
    double* initial_states = new double[n_states * n_stp];
    double* torque_params = new double[n_controls * n_stp];
    double* delta_t = new double[n_stp];
    
    try {
        // Convert CasADi matrices to integrator format
        parent_->convertDMtoArray(initial_states_dm, initial_states);
        parent_->convertDMtoArray(torque_params_dm, torque_params);
        parent_->convertDMtoArray(delta_t_dm, delta_t);
        
        // Perform integration with sensitivity analysis enabled
        int ret_code = integrator_->solve(initial_states, torque_params, delta_t, true);
        
        if (ret_code != 0) {
            casadi_error("Integration with sensitivity failed with code " + std::to_string(ret_code));
        }

        // Get sensitivity sizes
        int y0_nnz = integrator_->getSparsitySizeY0();
        int u_nnz = integrator_->getSparsitySizeU();
        int dt_nnz = integrator_->getSparsitySizeDt();
        
        if (y0_nnz < 0 || u_nnz < 0 || dt_nnz < 0) {
            casadi_error("Failed to get sensitivity sizes");
        }
        
        // Allocate arrays for sensitivity values
        double* y0_values = new double[y0_nnz];
        double* u_values = new double[u_nnz];
        double* dt_values = new double[dt_nnz];
        
        try {
            // Extract sensitivity values in sparse CCS format
            ret_code = integrator_->getSensitivitiesY0(y0_values);
            if (ret_code != 0) {
                casadi_error("Failed to get Y0 sensitivities");
            }
            
            ret_code = integrator_->getSensitivitiesU(u_values);
            if (ret_code != 0) {
                casadi_error("Failed to get U sensitivities");
            }
            
            ret_code = integrator_->getSensitivitiesDt(dt_values);
            if (ret_code != 0) {
                casadi_error("Failed to get dt sensitivities");
            }
            
            // Convert arrays to std::vector for CasADi matrix construction
            std::vector<double> y0_vec(y0_values, y0_values + y0_nnz);
            std::vector<double> u_vec(u_values, u_values + u_nnz);
            std::vector<double> dt_vec(dt_values, dt_values + dt_nnz);
            
            // Construct sparse CasADi matrices using pre-computed sparsity patterns
            // CasADi automatically maps values to correct positions using sparsity structure
            DM dy_dy0 = DM(Jac_y0, y0_vec);   // State-to-state Jacobian
            DM dy_du = DM(Jac_u, u_vec);      // Control-to-state Jacobian
            DM dy_ddt = DM(Jac_dt, dt_vec);   // Time-to-state Jacobian
            
            // Cleanup sensitivity arrays
            delete[] y0_values;
            delete[] u_values;
            delete[] dt_values;
            
            // Cleanup input arrays
            delete[] initial_states;
            delete[] torque_params;
            delete[] delta_t;
            
            // Return Jacobian matrices in order expected by CasADi
            std::vector<DM> jacobian;
            jacobian.push_back(dy_dy0);   // ∂X_next/∂X
            jacobian.push_back(dy_du);    // ∂X_next/∂U
            jacobian.push_back(dy_ddt);   // ∂X_next/∂dt

            return jacobian;
            
        } catch (...) {
            delete[] y0_values;
            delete[] u_values;
            delete[] dt_values;
            throw;
        }
        
    } catch (...) {
        // Cleanup input arrays on exception
        delete[] initial_states;
        delete[] torque_params;
        delete[] delta_t;
        throw;
    }
}