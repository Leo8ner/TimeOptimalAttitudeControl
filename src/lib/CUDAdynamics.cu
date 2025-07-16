#include <cuda_dynamics.h>

using namespace std;

/**
 * Device arrays for constant parameters during integration
 * d_torque_params: Array of torque parameters for each spacecraft system
 * d_inertia_constants: Precomputed inertia values and frequently used constants
 */
__device__ sunrealtype* d_torque_params;
__device__ __constant__ sunrealtype d_inertia_constants[12];

/**
 * GPU kernel for computing right-hand side of spacecraft dynamics equations
 * 
 * Computes derivatives for quaternion attitude representation and angular velocities
 * using Euler's equations for rigid body rotation.
 * 
 * Memory layout: Each system has n_states consecutive elements
 * State vector: [q0, q1, q2, q3, wx, wy, wz] per system
 * 
 * @param n_total Total number of states across all systems
 * @param y Input state vector (device memory)
 * @param ydot Output derivative vector (device memory)
 */
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot) {
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys < 0 || sys >= n_stp) return;

    // Load precomputed constants from constant memory (automatically cached)
    const sunrealtype Ix = d_inertia_constants[0];
    const sunrealtype Iy = d_inertia_constants[1]; 
    const sunrealtype Iz = d_inertia_constants[2];
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];

    int base_idx = sys * n_states;
    if (base_idx + n_states > n_total) return;

    // Load all state variables for this system (coalesced memory access)
    sunrealtype state_vars[n_states];
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        state_vars[i] = y[base_idx + i];
    }
    
    // Extract individual state components
    sunrealtype q0 = state_vars[0], q1 = state_vars[1];
    sunrealtype q2 = state_vars[2], q3 = state_vars[3];
    sunrealtype wx = state_vars[4], wy = state_vars[5], wz = state_vars[6];
    
    // Get torque parameters for this system
    sunrealtype tau_x = d_torque_params[sys * n_controls + 0];
    sunrealtype tau_y = d_torque_params[sys * n_controls + 1];
    sunrealtype tau_z = d_torque_params[sys * n_controls + 2];

    // Compute derivatives
    sunrealtype derivs[n_states];
    
    // Quaternion kinematic equations: q_dot = 0.5 * Q(q) * omega
    derivs[0] = half * (-q1*wx - q2*wy - q3*wz);  // q0_dot
    derivs[1] = half * ( q0*wx - q3*wy + q2*wz);  // q1_dot
    derivs[2] = half * ( q3*wx + q0*wy - q1*wz);  // q2_dot
    derivs[3] = half * (-q2*wx + q1*wy + q0*wz);  // q3_dot
    
    // Euler's equations: I*omega_dot = tau - omega x (I*omega)
    sunrealtype Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;
    
    derivs[4] = Ix_inv * (tau_x - (wy * Iw_z - wz * Iw_y));  // wx_dot
    derivs[5] = Iy_inv * (tau_y - (wz * Iw_x - wx * Iw_z));  // wy_dot
    derivs[6] = Iz_inv * (tau_z - (wx * Iw_y - wy * Iw_x));  // wz_dot
    
    // Store results (coalesced memory access)
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ydot[base_idx + i] = derivs[i];
    }
}

/**
 * GPU kernel for computing sparse Jacobian matrix blocks
 * 
 * Computes analytical Jacobian of dynamics RHS for each spacecraft system.
 * Uses block-diagonal CSR sparse format with NNZ_PER_BLOCK entries per system.
 * 
 * Sparsity pattern (per 7x7 block):
 * - Rows 0-3 (quaternion): All columns (quaternion-angular velocity coupling)
 * - Rows 4-6 (angular velocity): Only columns 4-6 (gyroscopic coupling)
 * 
 * @param n_blocks Number of spacecraft systems
 * @param block_data Sparse matrix data array (device memory)
 * @param y Current state vector (device memory)
 */
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, sunrealtype* y) {
    int block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    
    sunrealtype* block_jac = block_data + block_id * NNZ_PER_BLOCK;
    int base_state_idx = block_id * n_states;
    
    // Load state variables
    sunrealtype q0 = y[base_state_idx + 0], q1 = y[base_state_idx + 1];
    sunrealtype q2 = y[base_state_idx + 2], q3 = y[base_state_idx + 3];
    sunrealtype wx = y[base_state_idx + 4], wy = y[base_state_idx + 5], wz = y[base_state_idx + 6];
    
    // Load constants
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];
    const sunrealtype minus_half = d_inertia_constants[10];
    const sunrealtype zero = d_inertia_constants[11];
    
    // Fill entries according to CSR structure (37 entries total)
    int idx = 0;
    
    // Quaternion rows (0-3): 7 entries each = 28 total
    // Row 0: all 7 columns
    block_jac[idx++] = zero;                 // col 0
    block_jac[idx++] = minus_half * wx;      // col 1
    block_jac[idx++] = minus_half * wy;      // col 2
    block_jac[idx++] = minus_half * wz;      // col 3
    block_jac[idx++] = minus_half * q1;      // col 4
    block_jac[idx++] = minus_half * q2;      // col 5
    block_jac[idx++] = minus_half * q3;      // col 6
    
    // Row 1: all 7 columns
    block_jac[idx++] = half * wx;            // col 0
    block_jac[idx++] = zero;                 // col 1
    block_jac[idx++] = half * wz;            // col 2
    block_jac[idx++] = minus_half * wy;      // col 3
    block_jac[idx++] = half * q0;            // col 4
    block_jac[idx++] = minus_half * q3;      // col 5
    block_jac[idx++] = half * q2;            // col 6
    
    // Row 2: all 7 columns
    block_jac[idx++] = half * wy;            // col 0
    block_jac[idx++] = minus_half * wz;      // col 1
    block_jac[idx++] = zero;                 // col 2
    block_jac[idx++] = half * wx;            // col 3
    block_jac[idx++] = half * q3;            // col 4
    block_jac[idx++] = half * q0;            // col 5
    block_jac[idx++] = minus_half * q1;      // col 6
    
    // Row 3: all 7 columns
    block_jac[idx++] = half * wz;            // col 0
    block_jac[idx++] = half * wy;            // col 1
    block_jac[idx++] = minus_half * wx;      // col 2
    block_jac[idx++] = zero;                 // col 3
    block_jac[idx++] = minus_half * q2;      // col 4
    block_jac[idx++] = half * q1;            // col 5
    block_jac[idx++] = half * q0;            // col 6
    
    // Angular velocity rows (4-6): 3 entries each = 9 total
    // Row 4: only columns 4,5,6
    block_jac[idx++] = zero;                                // col 4
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wz;       // col 5
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wy;       // col 6
    
    // Row 5: only columns 4,5,6
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wz;       // col 4
    block_jac[idx++] = zero;                                // col 5
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wx;       // col 6
    
    // Row 6: only columns 4,5,6
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wy;       // col 4
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wx;       // col 5
    block_jac[idx++] = zero;                                // col 6
    
}

/**
 * GPU kernel for computing sensitivity RHS for spacecraft dynamics
 * 
 * Computes the sensitivity of the dynamics RHS with respect to initial conditions
 * and torque parameters. This is used for sensitivity analysis in CVODES.
 * 
 * Memory layout: Each system has n_states consecutive elements
 * Sensitivity vector: [∂q0/∂y, ∂q1/∂y, ..., ∂wz/∂y] per system
 * 
 * @param Ns Total number of sensitivity parameters (n_stp * (n_states + n_controls))
 * @param y Current state vector (device memory)
 * @param yS_data_array Array of sensitivity vectors (device memory)
 * @param ySdot_data_array Array of sensitivity derivatives (device memory)
 */
__global__ void sensitivityRHS(int Ns, sunrealtype* y, 
                                     sunrealtype** yS_data_array, 
                                     sunrealtype** ySdot_data_array) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= Ns) return;
    
    // Decode parameter info
    int sys = param_idx / (n_states + n_controls);
    int param_type = param_idx % (n_states + n_controls);
    
    if (sys >= n_stp) return; // Out of bounds check

    // Access full sensitivity vectors (size n_total = 350)
    sunrealtype* yS_data = yS_data_array[param_idx];
    sunrealtype* ySdot_data = ySdot_data_array[param_idx];
    
    // Load system state from global state vector
    int global_base = sys * n_states;
    sunrealtype q0 = y[global_base + 0], q1 = y[global_base + 1];
    sunrealtype q2 = y[global_base + 2], q3 = y[global_base + 3];
    sunrealtype wx = y[global_base + 4], wy = y[global_base + 5], wz = y[global_base + 6];
    
    // Load sensitivity state for this system (from full vector)
    sunrealtype s_q0 = yS_data[global_base + 0], s_q1 = yS_data[global_base + 1];
    sunrealtype s_q2 = yS_data[global_base + 2], s_q3 = yS_data[global_base + 3];
    sunrealtype s_wx = yS_data[global_base + 4], s_wy = yS_data[global_base + 5], s_wz = yS_data[global_base + 6];
    
    // Load constants
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];

    // Compute Jacobian-vector product: J * s (only for this system)
    sunrealtype Js[n_states];
    
    // Quaternion sensitivity equations: ∂/∂y(q_dot) * s
    Js[0] = half * (-wx*s_q1 - wy*s_q2 - wz*s_q3 - q1*s_wx - q2*s_wy - q3*s_wz);
    Js[1] = half * (wx*s_q0 + wz*s_q2 - wy*s_q3 + q0*s_wx - q3*s_wy + q2*s_wz);
    Js[2] = half * (wy*s_q0 - wz*s_q1 + wx*s_q3 + q3*s_wx + q0*s_wy - q1*s_wz);
    Js[3] = half * (wz*s_q0 + wy*s_q1 - wx*s_q2 - q2*s_wx + q1*s_wy + q0*s_wz);
    
    // Angular velocity sensitivity equations
    Js[4] = - Ix_inv * Iz_minus_Iy * wz * s_wy - Ix_inv * Iz_minus_Iy * wy * s_wz;
    Js[5] = -Iy_inv * Ix_minus_Iz * wz * s_wx - Iy_inv * Ix_minus_Iz * wx * s_wz;
    Js[6] = -Iz_inv * Iy_minus_Ix * wy * s_wx - Iz_inv * Iy_minus_Ix * wx * s_wy;
    
    // Add direct parameter dependencies: ∂f/∂p
    if (param_type >= n_states) {
        // Only torque parameters have direct RHS dependencies
        int torque_idx = param_type - n_states;
        if (torque_idx == 0) {
            Js[4] += Ix_inv;  // ∂(wx_dot)/∂(tau_x) = 1/Ix
        } else if (torque_idx == 1) {
            Js[5] += Iy_inv;  // ∂(wy_dot)/∂(tau_y) = 1/Iy
        } else if (torque_idx == 2) {
            Js[6] += Iz_inv;  // ∂(wz_dot)/∂(tau_z) = 1/Iz
        }
    }
    
    // Store results only for this system (global indexing)
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ySdot_data[global_base + i] = Js[i];
    }
}


/**
 * Constructor: Initialize CUDA-based dynamics integrator
 * 
 * Sets up GPU memory, SUNDIALS context, sparse matrices, and linear solvers
 * for batch integration of spacecraft attitude dynamics.
 */
DynamicsIntegrator::DynamicsIntegrator(bool enable_sensitivity, bool verb) : 
    n_total(N_TOTAL_STATES), setup_time(0), solve_time(0), verbose(verb), 
    d_yS_ptrs(nullptr), d_ySdot_ptrs(nullptr),yS(nullptr), Ns(0), 
    sensitivity_enabled(enable_sensitivity),
    sens_was_setup(false) {
    
    // Create CUDA events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    try {
        // Initialize constant memory with precomputed inertia values
        sunrealtype h_constants[12] = {
            i_x, i_y, i_z, 0.5,                    // [0-3]: Inertias and 0.5
            1.0/i_x, 1.0/i_y, 1.0/i_z,             // [4-6]: Inverse inertias
            (i_z - i_y), (i_x - i_z), (i_y - i_x), // [7-9]: Inertia differences (Euler equations)
            -0.5, 0.0                              // [10-11]: Commonly used constants
        };
        
        cudaError_t cuda_err = cudaMemcpyToSymbol(d_inertia_constants, h_constants, 
                                    12 * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error copying to constant memory: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        // Initialize SUNDIALS context
        if (SUNContext_Create(NULL, &sunctx) != 0) {
            throw runtime_error("Error creating SUNDIALS context");
        }
        
        // Create CUDA library handles
        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            throw runtime_error("Error creating cuSPARSE handle: " + 
                                    to_string(cusparse_status));
        }
        
        cusolverStatus_t cusolver_status = cusolverSpCreate(&cusolver_handle);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            throw runtime_error("Error creating cuSolver handle: " + 
                                    to_string(cusolver_status));
        }
        
        // Create main state vector on GPU
        y = N_VNew_Cuda(n_total, sunctx);
        if (!y) {
            throw runtime_error("Error creating CUDA vector");
        }
        
        // Allocate pinned host memory for efficient transfers
        cuda_err = cudaMallocHost(&h_y_pinned, n_total * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating states pinned memory: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        // Allocate pinned host memory for efficient transfers
        cuda_err = cudaMallocHost(&h_tau_pinned, n_stp * n_controls * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating controls pinned memory: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        // Create CVODES integrator instance
        cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
        if (!cvode_mem) {
            throw runtime_error("Error creating CVODES memory");
        }

        // Initialize CVODES for this integration
        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != 0) {
            throw runtime_error("Error initializing CVODES: " + to_string(retval));
        }
        
        // Create block-diagonal sparse Jacobian matrix
        nnz = n_stp * NNZ_PER_BLOCK;
        Jac = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, 
                                            n_states, NNZ_PER_BLOCK, 
                                            cusparse_handle, sunctx);
        if (!Jac) {
            throw runtime_error("Error creating cuSPARSE matrix");
        }
        
        setupJacobianStructure();
        
        // Create batch QR linear solver for block-diagonal systems
        LS = SUNLinSol_cuSolverSp_batchQR(y, Jac, cusolver_handle, sunctx);
        if (!LS) {
            throw runtime_error("Error creating cuSolverSp_batchQR linear solver");
        }
        
        // Allocate device memory for torque parameters
        cuda_err = cudaMalloc(&d_torque_params_ptr, n_stp * n_controls * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating device memory for torque params: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }
        
        // Store device pointer in device constant memory
        cuda_err = cudaMemcpyToSymbol(d_torque_params, &d_torque_params_ptr, sizeof(sunrealtype*));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error copying torque params pointer to symbol: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }
        
        retval = CVodeSetUserData(cvode_mem, this);
        if (retval != 0) {
            throw runtime_error("Error setting CVODES user data: " + to_string(retval));
        }
        
        
        // Set integration tolerances
        retval = CVodeSStolerances(cvode_mem, DEFAULT_RTOL, DEFAULT_ATOL);
        if (retval != 0) {
            throw runtime_error("Error setting CVODES tolerances: " + to_string(retval));
        }
        
        // Attach linear solver
        retval = CVodeSetLinearSolver(cvode_mem, LS, Jac);
        if (retval != 0) {
            throw runtime_error("Error setting CVODES linear solver: " + to_string(retval));
        }
        
        // Set Jacobian function
        retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
        if (retval != 0) {
            throw runtime_error("Error setting Jacobian function: " + to_string(retval));
        }
        
        // Set maximum number of internal steps
        retval = CVodeSetMaxNumSteps(cvode_mem, MAX_CVODE_STEPS);
        if (retval != 0) {
            throw runtime_error("Error setting maximum CVODES steps: " + to_string(retval));
        }

        // Setup sensitivity analysis if requested
        if (enable_sensitivity) {
            retval = setupSensitivityAnalysis();
            if (retval != 0) {
                throw runtime_error("Error setting up sensitivity analysis: " + to_string(retval));
            }
        }
        
    } catch (const runtime_error& e) {
        cleanup(); // Ensure cleanup is called before re-throwing
        throw;     // Re-throw the exception
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setup_time, start, stop);
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
;
}

/**
 * Destructor: Clean up resources allocated by the integrator
 * 
 * Frees all GPU memory, SUNDIALS context, and other resources used during integration.
*/
DynamicsIntegrator::~DynamicsIntegrator() {
    cleanup();
}

/**
 * Cleanup: Free all allocated resources
 * 
 * This method is called in the destructor and on error to ensure all GPU and SUNDIALS
 * resources are properly released.
 */
void DynamicsIntegrator::cleanup() {
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (Jac) SUNMatDestroy(Jac);
    if (LS) SUNLinSolFree(LS);
    if (y) N_VDestroy(y);
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
    if (d_torque_params_ptr) cudaFree(d_torque_params_ptr);
    if (h_y_pinned) cudaFreeHost(h_y_pinned);
    if (h_tau_pinned) cudaFreeHost(h_tau_pinned);
    if (sunctx) SUNContext_Free(&sunctx);
    if (yS) N_VDestroyVectorArray(yS, Ns);
    if (d_yS_ptrs) {
        cudaFree(d_yS_ptrs);
        d_yS_ptrs = nullptr;
    }
    if (d_ySdot_ptrs) {
        cudaFree(d_ySdot_ptrs);
        d_ySdot_ptrs = nullptr;
    }
}

/**
 * Setup the sparse Jacobian structure for the dynamics equations
 * 
 * Builds the CSR structure for the Jacobian matrix based on the sparsity pattern
 * of the dynamics equations. The Jacobian is block-diagonal with a fixed sparsity
 * pattern that does not change during integration.
 */
void DynamicsIntegrator::setupJacobianStructure() {
    vector<sunindextype> h_rowptrs(n_states + 1);
    vector<sunindextype> h_colvals(NNZ_PER_BLOCK);
    
    int nnz_count = 0;
    
    // Quaternion rows (0-3): All 7 columns each
    for (int row = 0; row < 4; row++) {
        h_rowptrs[row] = nnz_count;
        for (int j = 0; j < n_states; j++) {
            h_colvals[nnz_count++] = j;  // Columns 0,1,2,3,4,5,6
        }
    }
    
    // Angular velocity rows (4-6): Only columns 4,5,6 each
    for (int row = 4; row < n_states; row++) {
        h_rowptrs[row] = nnz_count;
        for (int j = 4; j < n_states; j++) {
            h_colvals[nnz_count++] = j;  // Columns 4,5,6 only
        }
    }
    
    h_rowptrs[n_states] = nnz_count;
    
    // Verify correct count
    assert(nnz_count == NNZ_PER_BLOCK);  // Should be exactly 37
    
    // Transfer to device (rest unchanged)
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);
    
    CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs.data(),
                         (n_states + 1) * sizeof(sunindextype), 
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_colvals, h_colvals.data(),
                         NNZ_PER_BLOCK * sizeof(sunindextype),
                         cudaMemcpyHostToDevice));
    
    SUNMatrix_cuSparse_SetFixedPattern(Jac, SUNTRUE);
}

/**
 * Initialize forward sensitivity analysis
 * 
 * Sets up sensitivity vectors and CVODES sensitivity module for computing
 * derivatives with respect to initial conditions and torque parameters.
 * 
 * @return 0 on success, negative on error
 */
int DynamicsIntegrator::setupSensitivityAnalysis() {
    // Total parameters: (n_states initial conditions + n_controls torques) per system
    Ns = n_stp * (n_states + n_controls);
    
    // Create sensitivity vector array
    yS = N_VCloneVectorArray(Ns, y);
    if (!yS) {
        cerr << "Error creating sensitivity vectors" << endl;
        return -1;
    }
    
    // Initialize sensitivity vectors for current initial conditions
    initializeSensitivityVectors();
        
    // Initialize CVODES sensitivity module
    SUNDIALS_CHECK(
        CVodeSensInit(cvode_mem, Ns, CV_STAGGERED, sensitivityRHSFunction, yS),
        "Error initializing sensitivity analysis"
    );
    
    // Set sensitivity tolerances (same absolute tolerance for all parameters)
    vector<sunrealtype> abstol_S(Ns, SENSITIVITY_ATOL);
    SUNDIALS_CHECK(
        CVodeSensSStolerances(cvode_mem, SENSITIVITY_RTOL, abstol_S.data()),
        "Error setting sensitivity tolerances"
    );
    
    // Enable sensitivity error control
    SUNDIALS_CHECK(
        CVodeSetSensErrCon(cvode_mem, SUNTRUE),
        "Error enabling sensitivity error control"
    );

    // Allocate device arrays for sensitivity vector pointers
    CUDA_CHECK(cudaMalloc(&d_yS_ptrs, Ns * sizeof(sunrealtype*)));
    CUDA_CHECK(cudaMalloc(&d_ySdot_ptrs, Ns * sizeof(sunrealtype*)));
    initializeSensitivityPointers();

    sens_was_setup = true;
    sensitivity_enabled = true;
    return 0;
}

/**
 * Initialize sensitivity vectors for identity matrix structure
 * 
 * Sets ∂y/∂p at t=0:
 * - ∂(initial_state_i)/∂(initial_condition_j) = δ_ij (Kronecker delta)
 * - ∂(initial_state)/∂(torque_param) = 0
 */
void DynamicsIntegrator::initializeSensitivityVectors() {
    sunrealtype one = 1.0;
    int sys, state_idx, param_type;
    // Zero all sensitivity vectors using CUDA memset
    for (int is = 0; is < Ns; is++) {
        sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS[is]);
        CUDA_CHECK(cudaMemset(yS_data, 0, n_total * sizeof(sunrealtype)));
        param_type = is % (n_states + n_controls);
        if (param_type < n_states) {
            // Initial condition sensitivity: set identity
            sys = is / (n_states + n_controls);
            state_idx = sys * n_states + param_type;
            CUDA_CHECK(cudaMemcpy(&yS_data[state_idx], &one, sizeof(sunrealtype), cudaMemcpyHostToDevice));  // ∂y/∂y = 1
        }

    }
    
    // Torque parameters start with zero sensitivities (already done by memset)
}

/**
 * Initialize sensitivity pointers for GPU kernels
 * 
 * This method sets up device pointers to the sensitivity vectors, allowing
 * efficient access in the sensitivity RHS kernel.
 */
void DynamicsIntegrator::initializeSensitivityPointers() {
    // This method sets up POINTERS for GPU kernels
    // Called ONCE after yS vectors are created
    vector<sunrealtype*> h_yS_ptrs(Ns);
    for (int i = 0; i < Ns; i++) {
        h_yS_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(yS[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_yS_ptrs, h_yS_ptrs.data(), 
                         Ns * sizeof(sunrealtype*), cudaMemcpyHostToDevice));
}

/**
 * CVODES-compatible RHS function wrapper
 */
int DynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
    // Launch RHS kernel with optimal thread configuration
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dynamicsRHS, 0, 0);
    int gridSize = (n_stp + blockSize - 1) / blockSize;
    
    dynamicsRHS<<<gridSize, blockSize>>>(N_VGetLength(y), y_data, ydot_data);
    
    CUDA_CHECK_KERNEL();
    
    return 0;
}

/**
 * CVODES-compatible Jacobian function wrapper
 */
int DynamicsIntegrator::jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                           SUNMatrix Jac, void* user_data, 
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    
    int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        
    // Launch Jacobian kernel (one thread block per spacecraft system)
    dim3 blockSize(32);
    dim3 gridSize(n_blocks);
    
    sparseJacobian<<<gridSize, blockSize>>>(n_blocks, data, y_data);
        
    CUDA_CHECK_KERNEL();
        
    return 0;
}
/**
 * CVODES-compatible wrapper for sensitivity RHS function
 */
int DynamicsIntegrator::sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                                             N_Vector* yS, N_Vector* ySdot, void* user_data,
                                             N_Vector tmp1, N_Vector tmp2) {
    
    DynamicsIntegrator* integrator = static_cast<DynamicsIntegrator*>(user_data);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);

    // Extract device pointers from current N_Vector arrays
    vector<sunrealtype*> h_yS_ptrs(Ns), h_ySdot_ptrs(Ns);
    for (int i = 0; i < Ns; i++) {
        h_yS_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(yS[i]);
        h_ySdot_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(ySdot[i]);
        
        // Zero each output vector efficiently using CUDA memset
        CUDA_CHECK(cudaMemset(h_ySdot_ptrs[i], 0, integrator->n_total * sizeof(sunrealtype)));
    }
    
    // Copy current pointers to pre-allocated device arrays
    CUDA_CHECK(cudaMemcpy(integrator->d_yS_ptrs, h_yS_ptrs.data(), 
                         Ns * sizeof(sunrealtype*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(integrator->d_ySdot_ptrs, h_ySdot_ptrs.data(), 
                         Ns * sizeof(sunrealtype*), cudaMemcpyHostToDevice));

    // Single batched kernel launch
    dim3 blockSize(128, 1);
    dim3 gridSize((Ns + blockSize.x - 1) / blockSize.x);
    // Launch kernel with pointer arrays
    sensitivityRHS<<<gridSize, blockSize>>>(Ns, y_data, integrator->d_yS_ptrs, integrator->d_ySdot_ptrs);

    CUDA_CHECK_KERNEL();
    
    return 0;  
}

/**
 * Main integration function
 * 
 * @param initial_states Initial conditions for all spacecraft systems
 * @param torque_params Torque inputs for all spacecraft systems
 * @param delta_t Integration time span
 * @param enable_sensitivity Whether to compute parameter sensitivities
 * @return 0 on success, negative on error
 */
int DynamicsIntegrator::solve(const vector<vector<sunrealtype>>& initial_states, 
                             const vector<vector<sunrealtype>>& torque_params,
                             const sunrealtype& delta_t, bool enable_sensitivity) {
    
    // Create CUDA events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
    // Transfer initial conditions to GPU
    setInitialConditions(initial_states, torque_params);

    // Set sensitivity analysis if requested
    if (enable_sensitivity && !sensitivity_enabled) {
        if (sens_was_setup) {
            // If sensitivity was previously set up, just reinitialize
            initializeSensitivityVectors();
            SUNDIALS_CHECK(CVodeSensReInit(cvode_mem, CV_STAGGERED, yS), 
            "Error reinitializing sensitivity analysis");
        } else {
            SUNDIALS_CHECK(setupSensitivityAnalysis(), "Error setting up sensitivity analysis");
        }
        sensitivity_enabled = true;
    } else if (!enable_sensitivity && sensitivity_enabled) {
        // Disable sensitivity analysis if it was previously enabled
        SUNDIALS_CHECK(CVodeSensToggleOff(cvode_mem), "Error disabling  sensitivity analysis");
        sensitivity_enabled = false;
    } else if (enable_sensitivity && sensitivity_enabled) {
        // If already enabled, just reinitialize
        initializeSensitivityVectors();
        SUNDIALS_CHECK(CVodeSensReInit(cvode_mem, CV_STAGGERED, yS), "Error reinitializing sensitivity analysis");
    }

    // Reset CVODE state
    SUNDIALS_CHECK(CVodeReInit(cvode_mem, 0.0, y), "Error reinitializing CVODE");

    // Perform integration from t=0 to t=delta_t
    sunrealtype t = 0.0;
    SUNDIALS_CHECK(CVode(cvode_mem, delta_t, y, &t, CV_NORMAL), "CVode integration error");

    CUDA_CHECK(cudaDeviceSynchronize());

    if (sensitivity_enabled) {
        // Compute sensitivity derivatives after integration
        SUNDIALS_CHECK(CVodeGetSens(cvode_mem, &t, yS), "Error getting sensitivity derivatives");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&solve_time, start, stop);

    if (verbose) {
        printSolutionStats();
    }
    return 0;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Copy initial conditions and torque parameters to GPU memory
 * 
 * @param initial_states Initial quaternion and angular velocity for each system
 * @param torque_params Applied torques for each system
 */
void DynamicsIntegrator::setInitialConditions(const vector<vector<sunrealtype>>& initial_states, 
                                                const vector<vector<sunrealtype>>& torque_params) {
    vector<sunrealtype> y0;
    vector<sunrealtype> torque_params_flat;
    y0.reserve(n_total);
    torque_params_flat.reserve(n_stp * n_controls);
    // Pack state data in memory layout: [sys0_states, sys1_states, ...]
    for (int i = 0; i < n_stp; i++) {
        const auto& state = initial_states[i];
        const auto& torque = torque_params[i];
        for (int j = 0; j < n_states; j++) {
            y0.push_back(state[j]);
        }
        for (int j = 0; j < n_controls; j++) {
            torque_params_flat.push_back(torque[j]);
        }
    }
    
    // Use pinned memory for faster host-device transfers
    memcpy(h_y_pinned, y0.data(), n_total * sizeof(sunrealtype));
    memcpy(h_tau_pinned, torque_params_flat.data(), n_stp * n_controls * sizeof(sunrealtype));

    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_total * sizeof(sunrealtype), 
                              cudaMemcpyHostToDevice));

    // Copy torque parameters to device
    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, h_tau_pinned, 
                             n_stp * n_controls * sizeof(sunrealtype), cudaMemcpyHostToDevice));
}

/**
 * Retrieve final state solution from GPU memory
 * 
 * @return Vector of final states for all spacecraft systems
 */
vector<vector<sunrealtype>> DynamicsIntegrator::getSolution() const {

    vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    
    // Copy solution from GPU to host
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), 
                         cudaMemcpyDeviceToHost));
    
    // Unpack into vector<sunrealtype> structure
    vector<vector<sunrealtype>> solutions(n_stp);
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        solutions[i].reserve(n_states);
        for (int j = 0; j < n_states; j++) {
            solutions[i].push_back(y_host[base_idx + j]);
        }
    }
    return solutions;
}

/**
 * Compute quaternion norms to check integration accuracy
 * 
 * @return Vector of quaternion norms (should be close to 1.0)
 */
vector<sunrealtype> DynamicsIntegrator::getQuaternionNorms() const {

    vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), 
                         cudaMemcpyDeviceToHost));
    
    vector<sunrealtype> norms(n_stp);
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        norms[i] = sqrt(y_host[base_idx+0]*y_host[base_idx+0] + 
                       y_host[base_idx+1]*y_host[base_idx+1] + 
                       y_host[base_idx+2]*y_host[base_idx+2] + 
                       y_host[base_idx+3]*y_host[base_idx+3]);
    }
    return norms;
}

/**
 * Retrieve sensitivity analysis results
 * @return Tuple containing:
 * - Vector of sensitivity data (CSR format)
 * - Vector of column indices for CSR   
 * - Vector of row pointers for CSR
 * - Total number of states (n_total)
 * - Number of sensitivity parameters (Ns)
 */
tuple<vector<sunrealtype>, vector<int>, vector<int>, int, int> 
DynamicsIntegrator::getSensitivities() const {
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if (!sensitivity_enabled) {
        cerr << "Sensitivity analysis not enabled" << endl;
        return {};
    }
    
    // Copy sensitivity data from GPU - only extract non-zero entries
    vector<vector<sunrealtype>> sensitivity_data(Ns);
    for (int p = 0; p < Ns; p++) {
        int param_sys = p / (n_states + n_controls);
        int base_idx = param_sys * n_states;
        
        sensitivity_data[p].resize(n_states);  // Only 7 entries per parameter
        sunrealtype* d_yS = N_VGetDeviceArrayPointer_Cuda(yS[p]);
        
        // Copy only the 7 non-zero entries from this parameter's system
        CUDA_CHECK(cudaMemcpy(sensitivity_data[p].data(), 
                             &d_yS[base_idx], 
                             n_states * sizeof(sunrealtype), 
                             cudaMemcpyDeviceToHost));
    }
    
    // Build block-diagonal CSR structure
    vector<sunrealtype> data;
    vector<int> indices;
    vector<int> indptr;
    
    int total_nnz = Ns * n_states;  // 500 * 7 = 3,500 entries
    data.reserve(total_nnz);
    indices.reserve(total_nnz);
    indptr.reserve(n_total + 1);
    
    // Build CSR row by row (global state indexing)
    for (int global_row = 0; global_row < n_total; global_row++) {
        indptr.push_back(data.size());
        
        int row_sys = global_row / n_states;
        int local_state = global_row % n_states;
        
        // Add entries only for this system's parameters
        int param_start = row_sys * (n_states + n_controls);
        int param_end = param_start + (n_states + n_controls);
        
        for (int param_idx = param_start; param_idx < param_end; param_idx++) {
            // Extract from compressed sensitivity data
            data.push_back(sensitivity_data[param_idx][local_state]);
            indices.push_back(param_idx);
        }
    }
    indptr.push_back(data.size());
    
    return make_tuple(move(data), move(indices), move(indptr), n_total, Ns);
}

/**
 * Print detailed integration statistics and solution quality metrics
 */
void DynamicsIntegrator::printSolutionStats() {
    // CVODES statistics
    long int nsteps, nfevals, nlinsetups;
    CVodeGetNumSteps(cvode_mem, &nsteps);
    CVodeGetNumRhsEvals(cvode_mem, &nfevals);
    CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
    
    long int njevals, nliters;
    CVodeGetNumJacEvals(cvode_mem, &njevals);
    CVodeGetNumLinIters(cvode_mem, &nliters);
    
    cout << "Batch integration statistics:" << endl;
    cout << "  Setup time: " << setup_time << " ms" << endl;
    cout << "  Solve time: " << solve_time << " ms" << endl;
    cout << "  Total time: " << (setup_time + solve_time) << " ms" << endl;
    cout << "  Integration steps: " << nsteps << endl;
    cout << "  RHS evaluations: " << nfevals << endl;
    cout << "  Jacobian evaluations: " << njevals << endl;
    cout << "  Linear solver setups: " << nlinsetups << endl;
    cout << "  Linear iterations: " << nliters << endl;
    
    // Solution quality assessment
    auto quaternion_norms = getQuaternionNorms();
    double avg_norm = accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / n_stp;
    double max_deviation = 0.0;
    for (auto norm : quaternion_norms) {
        max_deviation = max(max_deviation, abs(static_cast<double>(norm) - 1.0));
    }
    
    cout << "Solution quality:" << endl;
    cout << "  Average quaternion norm: " << setprecision(6) << avg_norm << endl;
    cout << "  Maximum norm deviation: " << setprecision(2) << (max_deviation * 100.0) << "%" << endl;
}




