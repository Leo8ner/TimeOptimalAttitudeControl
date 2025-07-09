#include <cuda_dynamics.h>

/**
 * Device arrays for constant parameters during integration
 * d_torque_params: Array of torque parameters for each spacecraft system
 * d_inertia_constants: Precomputed inertia values and frequently used constants
 */
__device__ TorqueParams* d_torque_params;
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
    if (sys >= n_stp) return;

    // Load precomputed constants from constant memory (automatically cached)
    const sunrealtype Ix = d_inertia_constants[0];
    const sunrealtype Iy = d_inertia_constants[1]; 
    const sunrealtype Iz = d_inertia_constants[2];
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];

    int base_idx = sys * n_states;
    
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
    TorqueParams params = d_torque_params[sys];
    
    // Compute derivatives
    sunrealtype derivs[n_states];
    
    // Quaternion kinematic equations: q_dot = 0.5 * Q(q) * omega
    derivs[0] = half * (-q1*wx - q2*wy - q3*wz);  // q0_dot
    derivs[1] = half * ( q0*wx - q3*wy + q2*wz);  // q1_dot
    derivs[2] = half * ( q3*wx + q0*wy - q1*wz);  // q2_dot
    derivs[3] = half * (-q2*wx + q1*wy + q0*wz);  // q3_dot
    
    // Euler's equations: I*omega_dot = tau - omega x (I*omega)
    sunrealtype Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;
    
    derivs[4] = Ix_inv * (params.tau_x - (wy * Iw_z - wz * Iw_y));  // wx_dot
    derivs[5] = Iy_inv * (params.tau_y - (wz * Iw_x - wx * Iw_z));  // wy_dot
    derivs[6] = Iz_inv * (params.tau_z - (wx * Iw_y - wy * Iw_x));  // wz_dot
    
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
 * GPU kernel for computing sensitivity equation right-hand side
 * 
 * Implements forward sensitivity analysis: d/dt(∂y/∂p) = ∂f/∂y * (∂y/∂p) + ∂f/∂p
 * 
 * Parameter encoding (linear indexing):
 * - Parameters 0 to n_states-1: Initial conditions for system 0
 * - Parameters n_states to 2*n_states+2: Initial conditions + torques for system 0
 * - And so on for each system
 * 
 * @param n_total Total number of states
 * @param Ns Total number of sensitivity parameters
 * @param y Current state vector
 * @param yS_array Sensitivity vector for all parameters
 * @param ySdot_array Output sensitivity derivative
 */
 __global__ void sensitivityRHS(int n_total, int Ns, sunrealtype* y, 
                                      N_Vector* yS_array, N_Vector* ySdot_array) {
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    int param_idx = blockIdx.y;
    
    if (sys >= n_stp || param_idx >= Ns) return;
    
    // Get sensitivity data pointers
    sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS_array[param_idx]);
    sunrealtype* ySdot_data = N_VGetDeviceArrayPointer_Cuda(ySdot_array[param_idx]);
    
    int base_idx = sys * n_states;
    
    // Load current state for this system
    sunrealtype q0 = y[base_idx + 0], q1 = y[base_idx + 1];
    sunrealtype q2 = y[base_idx + 2], q3 = y[base_idx + 3];
    sunrealtype wx = y[base_idx + 4], wy = y[base_idx + 5], wz = y[base_idx + 6];
    
    // Load current sensitivity state for this system and parameter
    sunrealtype s_q0 = yS_data[base_idx + 0], s_q1 = yS_data[base_idx + 1];
    sunrealtype s_q2 = yS_data[base_idx + 2], s_q3 = yS_data[base_idx + 3];
    sunrealtype s_wx = yS_data[base_idx + 4], s_wy = yS_data[base_idx + 5], s_wz = yS_data[base_idx + 6];
    
    // Load constants
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];
    
    // Decode which system and parameter type this refers to
    int param_sys = param_idx / (n_states + 3);  // Which spacecraft system
    int param_type = param_idx % (n_states + 3); // Parameter type within system
    
    // Compute Jacobian-vector product: J * s
    sunrealtype Js[n_states];
    
    // Quaternion sensitivity equations: ∂/∂y(q_dot) * s
    Js[0] = half * (-wx * s_q1 - wy * s_q2 - wz * s_q3 - q1 * s_wx - q2 * s_wy - q3 * s_wz);
    Js[1] = half * (wx * s_q0 + q0 * s_wx - wy * s_q3 - q3 * s_wy + wz * s_q2 + q2 * s_wz);
    Js[2] = half * (wx * s_q3 + q3 * s_wx + wy * s_q0 + q0 * s_wy - wz * s_q1 - q1 * s_wz);
    Js[3] = half * (-wx * s_q2 - q2 * s_wx + wy * s_q1 + q1 * s_wy + wz * s_q0 + q0 * s_wz);
    
    // Angular velocity sensitivity equations: ∂/∂y(ω_dot) * s (gyroscopic terms)
    Js[4] = Ix_inv * (Iz_minus_Iy * wz * s_wy + Iz_minus_Iy * wy * s_wz);
    Js[5] = Iy_inv * (Ix_minus_Iz * wz * s_wx + Ix_minus_Iz * wx * s_wz);
    Js[6] = Iz_inv * (Iy_minus_Ix * wy * s_wx + Iy_minus_Ix * wx * s_wy);
    
    // Add direct parameter dependencies: ∂f/∂p
    if (param_sys == sys && param_type >= n_states) {
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
    // Note: Initial condition parameters have no direct RHS contribution
    
    // Store sensitivity derivatives
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ySdot_data[base_idx + i] = Js[i];
    }
}

/**
 * Constructor: Initialize CUDA-based dynamics integrator
 * 
 * Sets up GPU memory, SUNDIALS context, sparse matrices, and linear solvers
 * for batch integration of spacecraft attitude dynamics.
 */
DynamicsIntegrator::DynamicsIntegrator(bool verb) : 
    n_total(N_TOTAL_STATES), setup_time(0), solve_time(0), verbose(verb),
    yS(nullptr), Ns(0), sensitivity_enabled(false), d_jacobian_workspace(nullptr) {
    
    timer.start();
    
    // Initialize constant memory with precomputed inertia values
    sunrealtype h_constants[12] = {
        i_x, i_y, i_z, 0.5,                    // [0-3]: Inertias and 0.5
        1.0/i_x, 1.0/i_y, 1.0/i_z,             // [4-6]: Inverse inertias
        (i_z - i_y), (i_x - i_z), (i_y - i_x), // [7-9]: Inertia differences (Euler equations)
        -0.5, 0.0                              // [10-11]: Commonly used constants
    };
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_inertia_constants, h_constants, 
                                12 * sizeof(sunrealtype)));

    // Initialize SUNDIALS context
    if (SUNContext_Create(NULL, &sunctx) != 0) {
        std::cerr << "Error creating SUNDIALS context" << std::endl;
        std::exit(1);
    }
    
    // Create CUDA library handles
    cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "Error creating cuSPARSE handle: " << cusparse_status << std::endl;
        exit(1);
    }
    
    cusolverStatus_t cusolver_status = cusolverSpCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error creating cuSolver handle: " << cusolver_status << std::endl;
        exit(1);
    }
    
    // Create main state vector on GPU
    y = N_VNew_Cuda(n_total, sunctx);
    if (!y) {
        std::cerr << "Error creating CUDA vector" << std::endl;
        exit(1);
    }
    
    // Allocate pinned host memory for efficient transfers
    CUDA_CHECK(cudaMallocHost(&h_y_pinned, n_total * sizeof(sunrealtype)));
    
    // Create block-diagonal sparse Jacobian matrix
    nnz = n_stp * NNZ_PER_BLOCK;
    Jac = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, 
                                      n_states, NNZ_PER_BLOCK, 
                                      cusparse_handle, sunctx);
    if (!Jac) {
        std::cerr << "Error creating cuSPARSE matrix" << std::endl;
        exit(1);
    }
    
    setupJacobianStructure();
    
    // Create batch QR linear solver for block-diagonal systems
    LS = SUNLinSol_cuSolverSp_batchQR(y, Jac, cusolver_handle, sunctx);
    if (!LS) {
        std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
        exit(1);
    }
    
    // Allocate device memory for torque parameters
    CUDA_CHECK(cudaMalloc(&d_torque_params_ptr, n_stp * sizeof(TorqueParams)));
    
    // Store device pointer in device constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_torque_params, &d_torque_params_ptr, sizeof(TorqueParams*)));

    // Create CVODES integrator instance
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (!cvode_mem) {
        std::cerr << "Error creating CVODES memory" << std::endl;
        exit(1);
    }
    
    setup_time = timer.getElapsedMs();
}

/**
 * Destructor: Clean up resources allocated by the integrator
 * 
 * Frees all GPU memory, SUNDIALS context, and other resources used during integration.
*/
DynamicsIntegrator::~DynamicsIntegrator() {
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (Jac) SUNMatDestroy(Jac);
    if (LS) SUNLinSolFree(LS);
    if (y) N_VDestroy(y);
    
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
    if (d_torque_params_ptr) cudaFree(d_torque_params_ptr);
    if (h_y_pinned) cudaFreeHost(h_y_pinned);
    if (sunctx) SUNContext_Free(&sunctx);
    if (d_jacobian_workspace) cudaFree(d_jacobian_workspace);

    // Clean up sensitivity vectors
    if (yS) {
        N_VDestroyVectorArray(yS, Ns);
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
    std::vector<sunindextype> h_rowptrs(n_states + 1);
    std::vector<sunindextype> h_colvals(NNZ_PER_BLOCK);
    
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
    // Total parameters: (n_states initial conditions + 3 torques) per system
    Ns = n_stp * (n_states + 3);
    
    // Create sensitivity vector array
    yS = N_VCloneVectorArray(Ns, y);
    if (!yS) {
        std::cerr << "Error creating sensitivity vectors" << std::endl;
        return -1;
    }
    
    // Initialize sensitivity vectors for current initial conditions
    initializeSensitivityVectors();
    
    // Allocate workspace for sensitivity computations
    CUDA_CHECK(cudaMalloc(&d_jacobian_workspace, Ns * n_total * sizeof(sunrealtype)));
    
    // Initialize CVODES sensitivity module
    SUNDIALS_CHECK(
        CVodeSensInit(cvode_mem, Ns, CV_SIMULTANEOUS, sensitivityRHSFunction, yS),
        "Error initializing sensitivity analysis"
    );
    
    // Set sensitivity tolerances (same absolute tolerance for all parameters)
    std::vector<sunrealtype> abstol_S(Ns, 1e-8);
    SUNDIALS_CHECK(
        CVodeSensSStolerances(cvode_mem, 1e-6, abstol_S.data()),
        "Error setting sensitivity tolerances"
    );
    
    // Enable sensitivity error control
    SUNDIALS_CHECK(
        CVodeSetSensErrCon(cvode_mem, SUNTRUE),
        "Error enabling sensitivity error control"
    );
    
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
    // Zero all sensitivity vectors using CUDA memset (faster than N_VConst)
    for (int is = 0; is < Ns; is++) {
        sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS[is]);
        CUDA_CHECK(cudaMemset(yS_data, 0, n_total * sizeof(sunrealtype)));
    }
    
    // Set identity elements with single-value transfers
    for (int sys = 0; sys < n_stp; sys++) {
        for (int state = 0; state < n_states; state++) {
            int param_idx = sys * (n_states + 3) + state;
            int global_state_idx = sys * n_states + state;
            
            if (param_idx < Ns) {
                sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
                sunrealtype one = 1.0;
                
                // Single element transfer (8 bytes instead of n_total * 8 bytes)
                CUDA_CHECK(cudaMemcpy(&yS_data[global_state_idx], &one, 
                                     sizeof(sunrealtype), cudaMemcpyHostToDevice));
            }
        }
    }
}

/**
 * CVODES-compatible RHS function wrapper
 */
int DynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
    // Launch RHS kernel with optimal thread configuration
    int blockSize = 128;
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
    
    // Single batched kernel launch instead of Ns separate launches
    dim3 blockSize(128, 1);
    dim3 gridSize((n_stp + blockSize.x - 1) / blockSize.x, Ns);
    
    sensitivityRHS<<<gridSize, blockSize>>>(integrator->n_total, Ns, y_data, yS, ySdot);
    CUDA_CHECK_KERNEL();
    
    return 0;  // Remove cudaDeviceSynchronize()
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
int DynamicsIntegrator::solve(const std::vector<StateParams>& initial_states, 
                             const std::vector<TorqueParams>& torque_params,
                             const sunrealtype& delta_t, bool enable_sensitivity) {
    
    // Validate input dimensions
    if (!validateInputs(initial_states, torque_params)) {
        return -1;
    }
    
    timer.start();
    
    // Transfer initial conditions to GPU
    setInitialConditions(initial_states, torque_params);
    
    // Initialize CVODES for this integration
    SUNDIALS_CHECK(
        CVodeInit(cvode_mem, rhsFunction, 0.0, y),
        "Error reinitializing CVODES"
    );
    
    CVodeSetUserData(cvode_mem, this);
    
    // Setup sensitivity analysis if requested
    if (enable_sensitivity && !sensitivity_enabled) {
        int retval = setupSensitivityAnalysis();
        if (retval != 0) {
            return retval;
        }
    } else if (enable_sensitivity && sensitivity_enabled) {
        // Reinitialize sensitivity vectors for new initial conditions
        initializeSensitivityVectors();
        SUNDIALS_CHECK(
            CVodeSensReInit(cvode_mem, CV_SIMULTANEOUS, yS),
            "Error reinitializing sensitivity analysis"
        );
    }
    
    // Set integration tolerances
    SUNDIALS_CHECK(
        CVodeSStolerances(cvode_mem, 1e-6, 1e-8),
        "Error setting tolerances"
    );
    
    // Attach linear solver
    SUNDIALS_CHECK(
        CVodeSetLinearSolver(cvode_mem, LS, Jac),
        "Error setting linear solver"
    );
    
    // Set Jacobian function
    SUNDIALS_CHECK(
        CVodeSetJacFn(cvode_mem, jacobianFunction),
        "Error setting Jacobian function"
    );
    
    // Set maximum number of internal steps
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    
    // Perform integration from t=0 to t=delta_t
    sunrealtype t = 0.0;
    int retval = CVode(cvode_mem, delta_t, y, &t, CV_NORMAL);
    
    if (retval < 0) {
        std::cerr << "CVode integration error: " << retval << std::endl;
        return retval;
    }

    // Synchronize device to ensure all operations complete
    CUDA_CHECK(cudaDeviceSynchronize());

    solve_time = timer.getElapsedMs();

    if (verbose) {
        printSolutionStats();
    }
    return 0;
}

/**
 * Copy initial conditions and torque parameters to GPU memory
 * 
 * @param initial_states Initial quaternion and angular velocity for each system
 * @param torque_params Applied torques for each system
 */
void DynamicsIntegrator::setInitialConditions(const std::vector<StateParams>& initial_states, 
                                                const std::vector<TorqueParams>& torque_params) {
    std::vector<sunrealtype> y0(n_total);
    
    // Pack state data in memory layout: [sys0_states, sys1_states, ...]
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        const auto& state = initial_states[i];
        
        y0[base_idx + 0] = state.q0; // Quaternion scalar part
        y0[base_idx + 1] = state.q1; // Quaternion vector part
        y0[base_idx + 2] = state.q2;
        y0[base_idx + 3] = state.q3;
        y0[base_idx + 4] = state.wx; // Angular velocity
        y0[base_idx + 5] = state.wy;
        y0[base_idx + 6] = state.wz;
    }
    
    // Use pinned memory for faster host-device transfers
    memcpy(h_y_pinned, y0.data(), n_total * sizeof(sunrealtype));
    
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_total * sizeof(sunrealtype), 
                              cudaMemcpyHostToDevice));

    // Copy torque parameters to device
    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, torque_params.data(), 
                             n_stp * sizeof(TorqueParams), cudaMemcpyHostToDevice));
}

/**
 * Validate input vectors have correct dimensions and physical validity
 */
bool DynamicsIntegrator::validateInputs(const std::vector<StateParams>& initial_states, 
                                               const std::vector<TorqueParams>& torque_params) {
    if (initial_states.size() != n_stp) {
        std::cerr << "Error: initial_states size (" << initial_states.size() 
                  << ") does not match n_stp (" << n_stp << ")" << std::endl;
        return false;
    }
    
    if (torque_params.size() != n_stp) {
        std::cerr << "Error: torque_params size (" << torque_params.size() 
                  << ") does not match n_stp (" << n_stp << ")" << std::endl;
        return false;
    }
    
    // Check quaternion normalization for each system
    for (int i = 0; i < n_stp; i++) {
        sunrealtype quat_norm_sq = initial_states[i].q0*initial_states[i].q0 + 
                                  initial_states[i].q1*initial_states[i].q1 + 
                                  initial_states[i].q2*initial_states[i].q2 + 
                                  initial_states[i].q3*initial_states[i].q3;
        if (abs(quat_norm_sq - 1.0) > 1e-6) {
            std::cerr << "Warning: initial_states[" << i << "] quaternion not normalized (norm^2 = " 
                      << quat_norm_sq << ")" << std::endl;
        }
    }
    
    return true;
}

/**
 * Retrieve final state solution from GPU memory
 * 
 * @return Vector of final states for all spacecraft systems
 */
std::vector<StateParams> DynamicsIntegrator::getSolution() {
    std::vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    
    // Copy solution from GPU to host
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), 
                         cudaMemcpyDeviceToHost));
    
    // Unpack into StateParams structure
    std::vector<StateParams> solutions(n_stp);
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        solutions[i].q0 = y_host[base_idx + 0];
        solutions[i].q1 = y_host[base_idx + 1];
        solutions[i].q2 = y_host[base_idx + 2];
        solutions[i].q3 = y_host[base_idx + 3];
        solutions[i].wx = y_host[base_idx + 4]; 
        solutions[i].wy = y_host[base_idx + 5];
        solutions[i].wz = y_host[base_idx + 6];
    }
    return solutions;
}

/**
 * Compute quaternion norms to check integration accuracy
 * 
 * @return Vector of quaternion norms (should be close to 1.0)
 */
std::vector<sunrealtype> DynamicsIntegrator::getQuaternionNorms() {
    std::vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), 
                         cudaMemcpyDeviceToHost));
    
    std::vector<sunrealtype> norms(n_stp);
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        norms[i] = 1.0/rsqrt(y_host[base_idx+0]*y_host[base_idx+0] + 
                       y_host[base_idx+1]*y_host[base_idx+1] + 
                       y_host[base_idx+2]*y_host[base_idx+2] + 
                       y_host[base_idx+3]*y_host[base_idx+3]);
    }
    return norms;
}

/**
 * Retrieve sensitivity analysis results
 * 
 * @return 3D array: sensitivities[param_idx][system_idx] = StateParams sensitivity
 */
std::vector<std::vector<StateParams>> DynamicsIntegrator::getSensitivities() {
    if (!sensitivity_enabled) {
        std::cerr << "Sensitivity analysis not enabled" << std::endl;
        return {};
    }
    
    std::vector<std::vector<StateParams>> sensitivities(Ns);
    
    // Copy each sensitivity vector from GPU and unpack
    for (int is = 0; is < Ns; is++) {
        std::vector<sunrealtype> yS_host(n_total);
        sunrealtype* d_yS = N_VGetDeviceArrayPointer_Cuda(yS[is]);
        CUDA_CHECK(cudaMemcpy(yS_host.data(), d_yS, n_total * sizeof(sunrealtype), 
                             cudaMemcpyDeviceToHost));
        
        sensitivities[is].resize(n_stp);
        for (int sys = 0; sys < n_stp; sys++) {
            int base_idx = sys * n_states;
            sensitivities[is][sys].q0 = yS_host[base_idx + 0];
            sensitivities[is][sys].q1 = yS_host[base_idx + 1];
            sensitivities[is][sys].q2 = yS_host[base_idx + 2];
            sensitivities[is][sys].q3 = yS_host[base_idx + 3];
            sensitivities[is][sys].wx = yS_host[base_idx + 4]; 
            sensitivities[is][sys].wy = yS_host[base_idx + 5];
            sensitivities[is][sys].wz = yS_host[base_idx + 6];
        }
    }
    return sensitivities;
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
    
    std::cout << "Batch integration statistics:" << std::endl;
    std::cout << "  Setup time: " << setup_time << " ms" << std::endl;
    std::cout << "  Solve time: " << solve_time << " ms" << std::endl;
    std::cout << "  Total time: " << (setup_time + solve_time) << " ms" << std::endl;
    std::cout << "  Integration steps: " << nsteps << std::endl;
    std::cout << "  RHS evaluations: " << nfevals << std::endl;
    std::cout << "  Jacobian evaluations: " << njevals << std::endl;
    std::cout << "  Linear solver setups: " << nlinsetups << std::endl;
    std::cout << "  Linear iterations: " << nliters << std::endl;
    
    // Solution quality assessment
    auto quaternion_norms = getQuaternionNorms();
    double avg_norm = std::accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / n_stp;
    double max_deviation = 0.0;
    for (auto norm : quaternion_norms) {
        max_deviation = std::max(max_deviation, std::abs(static_cast<double>(norm) - 1.0));
    }
    
    std::cout << "Solution quality:" << std::endl;
    std::cout << "  Average quaternion norm: " << std::setprecision(6) << avg_norm << std::endl;
    std::cout << "  Maximum norm deviation: " << std::setprecision(2) << (max_deviation * 100.0) << "%" << std::endl;
}