#include <cuda_dynamics.h>

// Device arrays for step parameters (constant during integration)
__device__ TorqueParams* d_torque_params;
__device__ __constant__ sunrealtype d_inertia_constants[12];

// Optimized RHS kernel with better memory access patterns
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot,
                           int systems_per_block = 4) {
        
    int sys = blockIdx.x * blockDim.x + threadIdx.x;

    if (sys >= n_stp) return;

    
    // Direct access to constant memory (cached automatically)
    const sunrealtype Ix = d_inertia_constants[0];
    const sunrealtype Iy = d_inertia_constants[1]; 
    const sunrealtype Iz = d_inertia_constants[2];
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];

    int base_idx = sys * n_states;
    
    // Coalesced memory loads
    sunrealtype state_vars[n_states];
    
    // Load all states for this system at once
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        state_vars[i] = y[base_idx + i];
    }
    
    // Extract state variables
    sunrealtype q0 = state_vars[0], q1 = state_vars[1];
    sunrealtype q2 = state_vars[2], q3 = state_vars[3];
    sunrealtype wx = state_vars[4], wy = state_vars[5], wz = state_vars[6];
    
    // Get step parameters from constant memory or texture memory
    TorqueParams params = d_torque_params[sys];
    
    // Compute derivatives with optimized math functions
    sunrealtype derivs[n_states];
    
    // Quaternion derivatives (vectorized)
    derivs[0] = half * (-q1*wx - q2*wy - q3*wz);
    derivs[1] = half * ( q0*wx - q3*wy + q2*wz);
    derivs[2] = half * ( q3*wx + q0*wy - q1*wz);
    derivs[3] = half * (-q2*wx + q1*wy + q0*wz);
    
    // Angular velocity derivatives with FMA operations
    sunrealtype Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;
    
    derivs[4] = Ix_inv * (params.tau_x - (wy * Iw_z - wz * Iw_y));
    derivs[5] = Iy_inv * (params.tau_y - (wz * Iw_x - wx * Iw_z));
    derivs[6] = Iz_inv * (params.tau_z - (wx * Iw_y - wy * Iw_x));
    
    // Coalesced memory stores
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ydot[base_idx + i] = derivs[i];
    }
    
}

// Sparse Jacobian kernel
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, sunrealtype* y) {
    int block_id = blockIdx.x;
    
    if (block_id >= n_blocks) return;
    
    // Get pointer to this block's data
    sunrealtype* block_jac = block_data + block_id * NNZ_PER_BLOCK;
    
    // Calculate state index for this block
    int base_state_idx = block_id * n_states;
    
    // Extract state variables for this block
    sunrealtype q0 = y[base_state_idx + 0];
    sunrealtype q1 = y[base_state_idx + 1];
    sunrealtype q2 = y[base_state_idx + 2];
    sunrealtype q3 = y[base_state_idx + 3];
    sunrealtype wx = y[base_state_idx + 4];
    sunrealtype wy = y[base_state_idx + 5];
    sunrealtype wz = y[base_state_idx + 6];
    
    // Use constant memory here too
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[8];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[9];
    const sunrealtype minus_half = d_inertia_constants[10];
    const sunrealtype zero = d_inertia_constants[11];
    
    // Fill sparse Jacobian entries in CSR order
    int idx = 0;
    
    // Row 0: q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] = zero;        // J[0,0] diagonal
    block_jac[idx++] = minus_half * wx;  // J[0,1]
    block_jac[idx++] = minus_half * wy;  // J[0,2]
    block_jac[idx++] = minus_half * wz;  // J[0,3]
    block_jac[idx++] = minus_half * q1;  // J[0,4]
    block_jac[idx++] = minus_half * q2;  // J[0,5]
    block_jac[idx++] = minus_half * q3;  // J[0,6]
    
    // Row 1: q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] = half * wx;        // J[1,0]
    block_jac[idx++] = zero;             // J[1,1] diagonal
    block_jac[idx++] = half * wz;        // J[1,2]
    block_jac[idx++] = minus_half * wy;  // J[1,3]
    block_jac[idx++] = half * q0;        // J[1,4]
    block_jac[idx++] = minus_half * q3;  // J[1,5]
    block_jac[idx++] = half * q2;        // J[1,6]
    
    // Row 2: q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] = half * wy;  // J[2,0]
    block_jac[idx++] = minus_half * wz;  // J[2,1]
    block_jac[idx++] = zero;        // J[2,2] diagonal
    block_jac[idx++] = half * wx;  // J[2,3]
    block_jac[idx++] = half * q3;  // J[2,4]
    block_jac[idx++] = half * q0;  // J[2,5]
    block_jac[idx++] = minus_half * q1;  // J[2,6]
    
    // Row 3: q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] = half * wz;  // J[3,0]
    block_jac[idx++] = half * wy;  // J[3,1]
    block_jac[idx++] = minus_half * wx;  // J[3,2]
    block_jac[idx++] = zero;        // J[3,3] diagonal
    block_jac[idx++] = minus_half * q2;  // J[3,4]
    block_jac[idx++] = half * q1;  // J[3,5]
    block_jac[idx++] = half * q0;  // J[3,6]
    
    // Row 4: wx_dot
    // Columns: 4,5,6
    block_jac[idx++] = zero;                       // J[4,4] diagonal
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wz;  // J[4,5]
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wy;  // J[4,6]
    
    // Row 5: wy_dot
    // Columns: 4,5,6
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wz;  // J[5,4]
    block_jac[idx++] = zero;                       // J[5,5] diagonal
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wx;  // J[5,6]
    
    // Row 6: wz_dot
    // Columns: 4,5,6
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wy;  // J[6,4]
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wx;  // J[6,5]
    block_jac[idx++] = zero;                       // J[6,6] diagonal
}

// GPU kernel for sensitivity RHS computation
__global__ void sensitivityRHS(int n_total, int Ns, 
                                     sunrealtype* y, sunrealtype* yS_data, 
                                     sunrealtype* ySdot_data, int param_idx) {
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys >= n_stp) return;
    
    int base_idx = sys * n_states;
    
    // Load state variables
    sunrealtype q0 = y[base_idx + 0], q1 = y[base_idx + 1];
    sunrealtype q2 = y[base_idx + 2], q3 = y[base_idx + 3];
    sunrealtype wx = y[base_idx + 4], wy = y[base_idx + 5], wz = y[base_idx + 6];
    
    // Load sensitivity variables for this parameter
    sunrealtype s_q0 = yS_data[base_idx + 0], s_q1 = yS_data[base_idx + 1];
    sunrealtype s_q2 = yS_data[base_idx + 2], s_q3 = yS_data[base_idx + 3];
    sunrealtype s_wx = yS_data[base_idx + 4], s_wy = yS_data[base_idx + 5], s_wz = yS_data[base_idx + 6];
    
    // Get constants
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype minus_half = d_inertia_constants[10];
    const sunrealtype zero = d_inertia_constants[11];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];
    
    // Determine parameter type and system
    int param_sys = param_idx / (n_states + 3);  // Which system this parameter belongs to
    int param_type = param_idx % (n_states + 3); // Type: 0-6 = initial conditions, 7-9 = torques
    
    // Compute Jacobian * sensitivity (J * s)
    sunrealtype Js[n_states];
    
    // Row 0: dq0_dot/dy * s
    Js[0] = zero * s_q0 + minus_half * wx * s_q1 + minus_half * wy * s_q2 + minus_half * wz * s_q3 +
            minus_half * q1 * s_wx + minus_half * q2 * s_wy + minus_half * q3 * s_wz;
    
    // Row 1: dq1_dot/dy * s  
    Js[1] = half * wx * s_q0 + zero * s_q1 + half * wz * s_q2 + minus_half * wy * s_q3 +
            half * q0 * s_wx + minus_half * q3 * s_wy + half * q2 * s_wz;
    
    // Row 2: dq2_dot/dy * s
    Js[2] = half * wy * s_q0 + minus_half * wz * s_q1 + zero * s_q2 + half * wx * s_q3 +
            half * q3 * s_wx + half * q0 * s_wy + minus_half * q1 * s_wz;
    
    // Row 3: dq3_dot/dy * s
    Js[3] = half * wz * s_q0 + half * wy * s_q1 + minus_half * wx * s_q2 + zero * s_q3 +
            minus_half * q2 * s_wx + half * q1 * s_wy + half * q0 * s_wz;
    
    // Row 4: dwx_dot/dy * s
    Js[4] = zero * s_wx + Ix_inv * Iz_minus_Iy * wz * s_wy + Ix_inv * Iz_minus_Iy * wy * s_wz;
    
    // Row 5: dwy_dot/dy * s  
    Js[5] = Iy_inv * Ix_minus_Iz * wz * s_wx + zero * s_wy + Iy_inv * Ix_minus_Iz * wx * s_wz;
    
    // Row 6: dwz_dot/dy * s
    Js[6] = Iz_inv * Iy_minus_Ix * wy * s_wx + Iz_inv * Iy_minus_Ix * wx * s_wy + zero * s_wz;
    
    // Add parameter derivatives (df/dp)
    if (param_sys == sys) {
        if (param_type < n_states) {
            // Initial condition parameter - no direct dependency in RHS
            // Js already contains the correct values
        } else {
            // Torque parameter
            int torque_idx = param_type - n_states;  // 0=tau_x, 1=tau_y, 2=tau_z
            if (torque_idx == 0) {
                Js[4] += Ix_inv;  // df4/dtau_x = 1/Ix
            } else if (torque_idx == 1) {
                Js[5] += Iy_inv;  // df5/dtau_y = 1/Iy
            } else if (torque_idx == 2) {
                Js[6] += Iz_inv;  // df6/dtau_z = 1/Iz
            }
        }
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ySdot_data[base_idx + i] = Js[i];
    }
}


// Constructor - performs one-time setup
DynamicsIntegrator::DynamicsIntegrator(bool verb) : 
    n_total(N_TOTAL_STATES), setup_time(0), solve_time(0), verbose(verb),
    yS(nullptr), Ns(0), sensitivity_enabled(false),
    d_param_indices(nullptr), d_jacobian_workspace(nullptr) {
    timer.start();
    
    // Initialize constant memory with precomputed values
    sunrealtype h_constants[12] = {
        i_x, i_y, i_z, 0.5,                    // [0-3]
        1.0/i_x, 1.0/i_y, 1.0/i_z,             // [4-7]
        (i_z - i_y), (i_x - i_z), (i_y - i_x), // [8-10] Euler equation terms
        -0.5, 0.0                              // [11-15] Other frequently used
    };
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_inertia_constants, h_constants, 
                                12 * sizeof(sunrealtype)));

    // Initialize SUNDIALS context
    int retval = SUNContext_Create(NULL, &sunctx);
    if (retval != 0) {
        std::cerr << "Error creating SUNDIALS context" << std::endl;
        exit(1);
    }
    
    // Create CUDA handles
    cusparseCreate(&cusparse_handle);
    cusolverSpCreate(&cusolver_handle);
    
    // Initialize vectors - just a single N_Vector
    y = N_VNew_Cuda(n_total, sunctx);
    if (!y) {
        std::cerr << "Error creating CUDA vector" << std::endl;
        exit(1);
    }
    
    // Use pinned memory only for initial conditions (most critical transfer)
    CUDA_CHECK(cudaMallocHost(&h_y_pinned, n_total * sizeof(sunrealtype)));
    
    // Create block-diagonal sparse matrix with ACTUAL sparsity
    nnz = n_stp * NNZ_PER_BLOCK;
    
    Jac = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, 
                                      n_states, NNZ_PER_BLOCK, 
                                      cusparse_handle, sunctx);
    if (!Jac) {
        std::cerr << "Error creating cuSPARSE matrix" << std::endl;
        exit(1);
    }
    
    setupJacobianStructure();
    
    // BatchQR solver works with sparse blocks
    LS = SUNLinSol_cuSolverSp_batchQR(y, Jac, cusolver_handle, sunctx);
    if (!LS) {
        std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
        exit(1);
    }
    
    // Allocate device memory for step parameters
    CUDA_CHECK(cudaMalloc(&d_torque_params_ptr, n_stp * sizeof(TorqueParams)));
    
    // Copy device pointer to device constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_torque_params, &d_torque_params_ptr, sizeof(TorqueParams*)));

    // Allocate Hessian matrix structure
    //setupHessianStructure();
    
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (!cvode_mem) {
        std::cerr << "Error creating CVODES memory" << std::endl;
        exit(1);
    }
    
    setup_time = timer.getElapsedMs();
}

// Destructor - cleanup
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
    if (d_param_indices) cudaFree(d_param_indices);
    if (d_jacobian_workspace) cudaFree(d_jacobian_workspace);

    // Clean up sensitivity vectors
    if (yS) {
        N_VDestroyVectorArray(yS, Ns);
    }
}

// // Enable/disable sensitivity analysis
// void DynamicsIntegrator::enableSensitivityAnalysis(bool enable) {
//     sensitivity_enabled = enable;
//     if (enable && !yS) {
//         setupSensitivityAnalysis();
//     }
// }
void DynamicsIntegrator::setupJacobianStructure() {
    std::vector<sunindextype> h_rowptrs(n_states + 1);
    std::vector<sunindextype> h_colvals(NNZ_PER_BLOCK);
    
    // Build CSR structure (columns must be in ascending order)
    int nnz_count = 0;
    
    // Row 0: columns 0,1,2,3,4,5,6
    h_rowptrs[0] = nnz_count;
    for (int j = 0; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 1: columns 0,1,2,3,4,5,6
    h_rowptrs[1] = nnz_count;
    for (int j = 0; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 2: columns 0,1,2,3,4,5,6
    h_rowptrs[2] = nnz_count;
    for (int j = 0; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 3: columns 0,1,2,3,4,5,6
    h_rowptrs[3] = nnz_count;
    for (int j = 0; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 4: columns 4,5,6
    h_rowptrs[4] = nnz_count;
    for (int j = 4; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 5: columns 4,5,6
    h_rowptrs[5] = nnz_count;
    for (int j = 4; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Row 6: columns 4,5,6
    h_rowptrs[6] = nnz_count;
    for (int j = 4; j < n_states; j++) h_colvals[nnz_count++] = j;
    
    // Final row pointer
    h_rowptrs[n_states] = nnz_count;
    
    // Copy to device
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);
    
    CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs.data(),
                             (n_states + 1) * sizeof(sunindextype), 
                             cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_colvals, h_colvals.data(),
                             NNZ_PER_BLOCK * sizeof(sunindextype),
                             cudaMemcpyHostToDevice));
    
    
    // Set fixed pattern since our sparsity structure doesn't change
    SUNMatrix_cuSparse_SetFixedPattern(Jac, SUNTRUE);
}

// // Setup sensitivity analysis
// int DynamicsIntegrator::setupSensitivityAnalysis() {
//     // Number of parameters: n_states initial conditions + 3 torque params per system
//     Ns = n_stp * (n_states + 3);
    
//     // Create parameter mapping
//     param_map.resize(Ns);
//     for (int sys = 0; sys < n_stp; sys++) {
//         for (int i = 0; i < n_states + 3; i++) {
//             param_map[sys * (n_states + 3) + i] = sys * 1000 + i; // Encode sys and param type
//         }
//     }
    
//     // Create sensitivity vectors on GPU
//     yS = N_VCloneVectorArray(Ns, y);
//     if (!yS) {
//         std::cerr << "Error creating sensitivity vectors" << std::endl;
//         return -1;
//     }
    
//     // Allocate workspace for Jacobian computation
//     CUDA_CHECK(cudaMalloc(&d_jacobian_workspace, Ns * n_total * sizeof(sunrealtype)));
    
//     // Initialize CVODES sensitivity analysis
//     int retval = CVodeSensInit(cvode_mem, Ns, CV_SIMULTANEOUS, 
//                               sensitivityRHS, yS);
//     if (retval != CV_SUCCESS) {
//         std::cerr << "Error initializing sensitivity analysis: " << retval << std::endl;
//         return retval;
//     }
    
//     // Set sensitivity tolerances
//     retval = CVodeSensSStolerances(cvode_mem, 1e-6, 1e-8);
//     if (retval != CV_SUCCESS) {
//         std::cerr << "Error setting sensitivity tolerances: " << retval << std::endl;
//         return retval;
//     }
    
//     // Enable sensitivity error control
//     retval = CVodeSetSensErrCon(cvode_mem, SUNTRUE);
//     if (retval != CV_SUCCESS) {
//         std::cerr << "Error enabling sensitivity error control: " << retval << std::endl;
//         return retval;
//     }
    
//     return 0;
// }



void DynamicsIntegrator::setInitialConditions(const std::vector<StateParams>& initial_states, 
                                                const std::vector<TorqueParams>& torque_params) {
    std::vector<sunrealtype> y0(n_total);
    
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        const auto& state = initial_states[i];
        
        y0[base_idx + 0] = state.q0; // q0
        y0[base_idx + 1] = state.q1; // q1
        y0[base_idx + 2] = state.q2; // q2
        y0[base_idx + 3] = state.q3; // q3
        y0[base_idx + 4] = state.wx; // wx
        y0[base_idx + 5] = state.wy; // wy
        y0[base_idx + 6] = state.wz; // wz
    }
    
    // Use pinned memory for faster transfers
    memcpy(h_y_pinned, y0.data(), n_total * sizeof(sunrealtype));
    
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_total * sizeof(sunrealtype), 
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, torque_params.data(), 
                             n_stp * sizeof(TorqueParams), cudaMemcpyHostToDevice));
}


int DynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    int n_total = N_VGetLength(y);
    
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
    // Find optimal configuration
    int blockSize = 128;  // Based on occupancy optimization
    int gridSize = (n_stp + 3) / 4;  // Systems per block = 4
    
    // Launch optimized kernel with stream
    dynamicsRHS<<<gridSize, blockSize, 0, 0>>>(n_total, y_data, ydot_data, 4);
    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RHS kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    return 0;
}

int DynamicsIntegrator::jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                           SUNMatrix Jac, void* user_data, 
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    
    // Get block information
    int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
    
    // Get pointer to the data array (contains all blocks)
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        
    // Launch kernel with one block per spacecraft system, using stream
    dim3 blockSize(32);
    dim3 gridSize(n_blocks);
    
    sparseJacobian<<<gridSize, blockSize, 0, 0>>>(n_blocks, data, y_data);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Jacobian kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }    
    return 0;
}

// // Static sensitivity RHS function
// int DynamicsIntegrator::sensitivityRHS(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
//                                       N_Vector* yS, N_Vector* ySdot, void* user_data,
//                                       N_Vector tmp1, N_Vector tmp2) {
    
//     DynamicsIntegrator* integrator = static_cast<DynamicsIntegrator*>(user_data);
    
//     sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    
//     // Launch kernel for each parameter
//     int blockSize = 128;
//     int gridSize = (n_stp + blockSize - 1) / blockSize;
    
//     for (int is = 0; is < Ns; is++) {
//         sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS[is]);
//         sunrealtype* ySdot_data = N_VGetDeviceArrayPointer_Cuda(ySdot[is]);
        
//         sensitivityRHS<<<gridSize, blockSize>>>(
//             integrator->n_total, Ns, y_data, yS_data, ySdot_data, is);
        
//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             std::cerr << "Sensitivity RHS kernel error for parameter " << is 
//                       << ": " << cudaGetErrorString(err) << std::endl;
//             return -1;
//         }
//     }
    
//     return 0;
// }

int DynamicsIntegrator::solve(const std::vector<StateParams>& initial_states, 
                                       const std::vector<TorqueParams>& torque_params,
                                       const sunrealtype& delta_t) {
    
    // Validate inputs
    if (!validateInputs(initial_states, torque_params)) {
        return -1.0;
    }
    
    timer.start();
    
    // Set up for this solve
    setInitialConditions(initial_states, torque_params);
    
    // Initialize CVODES for this solve
    int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error reinitializing CVODES: " << retval << std::endl;
        return -1;
    }
    
    CVodeSetUserData(cvode_mem, this);
    
    // Relaxed tolerances for batch processing
    retval = CVodeSStolerances(cvode_mem, 1e-6, 1e-8);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting tolerances: " << retval << std::endl;
        return -1;
    }
    
    retval = CVodeSetLinearSolver(cvode_mem, LS, Jac);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting linear solver: " << retval << std::endl;
        return -1;
    }
    
    retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting Jacobian function: " << retval << std::endl;
        return -1;
    }
    
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    
    // Solve
    sunrealtype t = 0.0;
    retval = CVode(cvode_mem, delta_t, y, &t, CV_NORMAL);
    
    if (retval < 0) {
        std::cerr << "CVode error: " << retval << std::endl;
        return -1;
    }
    
    solve_time = timer.getElapsedMs();

    if (verbose) {
        printSolutionStats();
    }
    return 0;
}

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
    
    for (int i = 0; i < n_stp; i++) {
        
        // Check quaternion normalization
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

std::vector<StateParams> DynamicsIntegrator::getSolution() {
    std::vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
    
    std::vector<StateParams> solutions(n_stp);
    int base_idx;
    for (int i = 0; i < n_stp; i++) {
        base_idx = i * n_states;
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

std::vector<sunrealtype> DynamicsIntegrator::getQuaternionNorms() {
    std::vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
    
    std::vector<sunrealtype> norms(n_stp);
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        norms[i] = sqrt(y_host[base_idx+0]*y_host[base_idx+0] + 
                       y_host[base_idx+1]*y_host[base_idx+1] + 
                       y_host[base_idx+2]*y_host[base_idx+2] + 
                       y_host[base_idx+3]*y_host[base_idx+3]);
    }
    return norms;
}


// std::tuple<std::vector<sunrealtype>, std::vector<sunindextype>, std::vector<sunindextype>> 
// DynamicsIntegrator::getJacobian() {    
    
//     // Compute fresh Jacobian at current state
//     N_Vector tmp1 = N_VClone(y);
//     N_Vector tmp2 = N_VClone(y);  
//     N_Vector tmp3 = N_VClone(y);
    
//     int retval = jacobianFunction(0.0, y, nullptr, Jac, this, tmp1, tmp2, tmp3);
    
//     N_VDestroy(tmp1);
//     N_VDestroy(tmp2);
//     N_VDestroy(tmp3);
    
//     if (retval != 0) {
//         std::cerr << "Error computing Jacobian" << std::endl;
//         return {};
//     }
    
//     // Get pointers to GPU data
//     sunrealtype* d_data = SUNMatrix_cuSparse_Data(Jac);
//     sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
//     sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);
        
//     // Allocate host memory
//     std::vector<sunrealtype> values(TOTAL_NNZ);
//     std::vector<sunindextype> row_ptrs(N_TOTAL_STATES + 1);
//     std::vector<sunindextype> col_vals(TOTAL_NNZ);
    
//     // Single batch of GPU->CPU transfers
//     cudaMemcpy(values.data(), d_data, TOTAL_NNZ * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
//     cudaMemcpy(row_ptrs.data(), d_rowptrs, (N_TOTAL_STATES + 1) * sizeof(sunindextype), cudaMemcpyDeviceToHost);
//     cudaMemcpy(col_vals.data(), d_colvals, TOTAL_NNZ * sizeof(sunindextype), cudaMemcpyDeviceToHost);
    
//     return std::make_tuple(std::move(values), std::move(row_ptrs), std::move(col_vals));
// }

void DynamicsIntegrator::printSolutionStats() {
    long int nsteps, nfevals, nlinsetups;
    CVodeGetNumSteps(cvode_mem, &nsteps);
    CVodeGetNumRhsEvals(cvode_mem, &nfevals);
    CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
    
    long int njevals, nliters;
    CVodeGetNumJacEvals(cvode_mem, &njevals);
    CVodeGetNumLinIters(cvode_mem, &nliters);
    
    std::cout << "Batch integration stats:" << std::endl;
    std::cout << "  Setup time: " << setup_time << " ms" << std::endl;
    std::cout << "  Solve time: " << solve_time << " ms" << std::endl;
    std::cout << "  Total time: " << (setup_time + solve_time) << " ms" << std::endl;
    std::cout << "  Steps: " << nsteps << std::endl;
    std::cout << "  RHS evaluations: " << nfevals << std::endl;
    std::cout << "  Jacobian evaluations: " << njevals << std::endl;
    std::cout << "  Linear solver setups: " << nlinsetups << std::endl;
    std::cout << "  Linear iterations: " << nliters << std::endl;
    
    auto quaternion_norms = getQuaternionNorms();
    double avg_norm = std::accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / n_stp;
    double max_deviation = 0.0;
    for (auto norm : quaternion_norms) {
        max_deviation = std::max(max_deviation, std::abs(static_cast<double>(norm) - 1.0));
    }
    
    std::cout << "  Average quaternion norm: " << std::setprecision(6) << avg_norm << std::endl;
    std::cout << "  Maximum norm deviation: " << std::setprecision(2) << (max_deviation * 100.0) << "%" << std::endl;
}