#include <cuda_dynamics.h>

// Device arrays for step parameters (constant during integration)
__device__ TorqueParams* d_torque_params;

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

__global__ void sparseHessian(int n_systems, sunrealtype* hess_data, sunrealtype* y, sunindextype* row_ptrs) {
    __shared__ sunrealtype shared_states[32][7];  // 32 steps per block
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    int equation_in_sys = threadIdx.y;  // Which equation within the system (0-6)
    
    if (sys >= n_systems) return;
    
    int global_equation = sys * n_states + equation_in_sys;
    int base_state_idx = sys * n_states;

    // Use constant memory
    const sunrealtype half = d_inertia_constants[3];
    const sunrealtype minus_half = d_inertia_constants[10];
    const sunrealtype zero = d_inertia_constants[11];
    const sunrealtype Ix_inv = d_inertia_constants[4];
    const sunrealtype Iy_inv = d_inertia_constants[5];
    const sunrealtype Iz_inv = d_inertia_constants[6];
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];
    const sunrealtype Iy_minus_Ix = d_inertia_constants[8];
    const sunrealtype Ix_minus_Iz = d_inertia_constants[9];
    
    // Each equation gets a row in the Hessian matrix
    // Row size is n_total × n_total, but most entries are zero
    int hess_row_start = row_ptrs[global_equation];
    int hess_row_end = row_ptrs[global_equation + 1];
    int num_entries = hess_row_end - hess_row_start;
    
    sunrealtype* hess_row = hess_data + hess_row_start;
    
    // Initialize to zero
    for (int i = 0; i < num_entries; i++) {
        hess_row[i] = zero;
    }   

    // Extract state variables for this system
    sunrealtype state_vars[n_states];

    if (sys < n_systems && threadIdx.y == 0) {  // Only one thread per spacecraft loads
        int base_state_idx = sys * n_states;
        #pragma unroll
        for (int i = 0; i < n_states; i++) {
            shared_states[threadIdx.x][i] = y[base_state_idx + i];
        }
    }
    
    __syncthreads();  // Wait for all loads

    sunrealtype q0 = shared_states[threadIdx.x][0];
    sunrealtype q1 = shared_states[threadIdx.x][1];
    sunrealtype q2 = shared_states[threadIdx.x][2];
    sunrealtype q3 = shared_states[threadIdx.x][3];
    sunrealtype wx = shared_states[threadIdx.x][4];
    sunrealtype wy = shared_states[threadIdx.x][5];
    sunrealtype wz = shared_states[threadIdx.x][6];

    int idx = 0;
    
    // Fill Hessian entries based on which equation this is
    switch(equation_in_sys) {
        case 0: // q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
            // ∂²q0_dot/∂q1∂wx = ∂²q0_dot/∂wx∂q1 = -0.5
            hess_row[idx++] = minus_half;  // H[q1,wx]
            hess_row[idx++] = minus_half;  // H[q2,wy]
            hess_row[idx++] = minus_half;  // H[q3,wz]
            hess_row[idx++] = minus_half;  // H[wx,q1] (symmetric)
            hess_row[idx++] = minus_half;  // H[wy,q2] (symmetric)
            hess_row[idx++] = minus_half;  // H[wz,q3] (symmetric)
            break;
            
        case 1: // q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
            hess_row[idx++] = half;        // H[q0,wx]
            hess_row[idx++] = half;        // H[q2,wz]
            hess_row[idx++] = minus_half;  // H[q3,wy]
            hess_row[idx++] = half;        // H[wx,q0] (symmetric)
            hess_row[idx++] = minus_half;  // H[wy,q3] (symmetric)
            hess_row[idx++] = half;        // H[wz,q2] (symmetric)
            break;
            
        case 2: // q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
            hess_row[idx++] = half;        // H[q0,wy]
            hess_row[idx++] = minus_half;  // H[q1,wz]
            hess_row[idx++] = half;        // H[q3,wx]
            hess_row[idx++] = half;        // H[wx,q3] (symmetric)
            hess_row[idx++] = half;        // H[wy,q0] (symmetric)
            hess_row[idx++] = minus_half;  // H[wz,q1] (symmetric)
            break;
            
        case 3: // q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
            hess_row[idx++] = half;        // H[q0,wz]
            hess_row[idx++] = half;        // H[q1,wy]
            hess_row[idx++] = minus_half;  // H[q2,wx]
            hess_row[idx++] = minus_half;  // H[wx,q2] (symmetric)
            hess_row[idx++] = half;        // H[wy,q1] (symmetric)
            hess_row[idx++] = half;        // H[wz,q0] (symmetric)
            break;
            
        case 4: // wx_dot = Ix_inv * (tau_x - wy*wz*(Iz - Iy))
            hess_row[idx++] = -Ix_inv * Iz_minus_Iy;  // H[wy,wz]
            hess_row[idx++] = -Ix_inv * Iz_minus_Iy;  // H[wz,wy] (symmetric)
            break;
            
        case 5: // wy_dot = Iy_inv * (tau_y - wz*wx*(Ix - Iz))
            hess_row[idx++] = -Iy_inv * Ix_minus_Iz;  // H[wx,wz]
            hess_row[idx++] = -Iy_inv * Ix_minus_Iz;  // H[wz,wx] (symmetric)
            break;
            
        case 6: // wz_dot = Iz_inv * (tau_z - wx*wy*(Iy - Ix))
            hess_row[idx++] = -Iz_inv * Iy_minus_Ix;  // H[wx,wy]
            hess_row[idx++] = -Iz_inv * Iy_minus_Ix;  // H[wy,wx] (symmetric)
            break;
    }
}

// Constructor - performs one-time setup
DynamicsIntegrator::DynamicsIntegrator(bool verb) : n_total(N_TOTAL_STATES), setup_time(0), solve_time(0), verbose(verb) {
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
    setupHessianStructure();
    
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
    if (Hes) SUNMatDestroy(Hes);
}

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

// Setup full system Hessian structure  
void DynamicsIntegrator::setupHessianStructure() {
    // Create Hessian matrix: n_total rows × (n_total × n_total) columns
    // But we only store non-zero entries for each equation
    int hess_rows = n_total;
    int hess_cols = n_total * n_total;  // Flattened Hessian tensor
    int total_hess_nnz = TOTAL_HESSIAN_NNZ;
    
    Hes = SUNMatrix_cuSparse_NewCSR(hess_rows, hess_cols, total_hess_nnz, cusparse_handle, sunctx);
    if (!Hes) {
        std::cerr << "Error creating full Hessian matrix" << std::endl;
        exit(1);
    }
    
    // Build sparsity pattern for the full Hessian matrix
    std::vector<sunindextype> h_rowptrs(hess_rows + 1);
    std::vector<sunindextype> h_colvals(total_hess_nnz);
    
    int nnz_count = 0;
    
    // For each equation (row in the Hessian matrix)
    for (int eq = 0; eq < n_total; eq++) {
        h_rowptrs[eq] = nnz_count;
        
        int sys = eq / n_states;  // Which system this equation belongs to
        int eq_in_sys = eq % n_states;  // Which equation within the system
        
        int base_col_offset = sys * n_states;  // Base column offset for this system
        
        // Add column indices for non-zero Hessian entries
        // Only entries within the same system are non-zero (no cross-coupling)
        switch(eq_in_sys) {
            case 0: // q0_dot equation
                // Flattened indices for mixed partial derivatives
                h_colvals[nnz_count++] = (base_col_offset + 1) * n_total + (base_col_offset + 4); // H[q1,wx]
                h_colvals[nnz_count++] = (base_col_offset + 2) * n_total + (base_col_offset + 5); // H[q2,wy]
                h_colvals[nnz_count++] = (base_col_offset + 3) * n_total + (base_col_offset + 6); // H[q3,wz]
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 1); // H[wx,q1]
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 2); // H[wy,q2]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 3); // H[wz,q3]
                break;
                
            case 1: // q1_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 0) * n_total + (base_col_offset + 4); // H[q0,wx]
                h_colvals[nnz_count++] = (base_col_offset + 2) * n_total + (base_col_offset + 6); // H[q2,wz]
                h_colvals[nnz_count++] = (base_col_offset + 3) * n_total + (base_col_offset + 5); // H[q3,wy]
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 0); // H[wx,q0]
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 3); // H[wy,q3]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 2); // H[wz,q2]
                break;
                
            case 2: // q2_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 0) * n_total + (base_col_offset + 5); // H[q0,wy]
                h_colvals[nnz_count++] = (base_col_offset + 1) * n_total + (base_col_offset + 6); // H[q1,wz]
                h_colvals[nnz_count++] = (base_col_offset + 3) * n_total + (base_col_offset + 4); // H[q3,wx]
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 3); // H[wx,q3]
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 0); // H[wy,q0]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 1); // H[wz,q1]
                break;
                
            case 3: // q3_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 0) * n_total + (base_col_offset + 6); // H[q0,wz]
                h_colvals[nnz_count++] = (base_col_offset + 1) * n_total + (base_col_offset + 5); // H[q1,wy]
                h_colvals[nnz_count++] = (base_col_offset + 2) * n_total + (base_col_offset + 4); // H[q2,wx]
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 2); // H[wx,q2]
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 1); // H[wy,q1]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 0); // H[wz,q0]
                break;
                
            case 4: // wx_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 6); // H[wy,wz]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 5); // H[wz,wy]
                break;
                
            case 5: // wy_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 6); // H[wx,wz]
                h_colvals[nnz_count++] = (base_col_offset + 6) * n_total + (base_col_offset + 4); // H[wz,wx]
                break;
                
            case 6: // wz_dot equation
                h_colvals[nnz_count++] = (base_col_offset + 4) * n_total + (base_col_offset + 5); // H[wx,wy]
                h_colvals[nnz_count++] = (base_col_offset + 5) * n_total + (base_col_offset + 4); // H[wy,wx]
                break;
        }
    }
    
    h_rowptrs[n_total] = nnz_count;
    
    // Verify we have the right number of non-zeros
    if (nnz_count != total_hess_nnz) {
        std::cerr << "Error: Expected " << total_hess_nnz << " Hessian non-zeros, got " << nnz_count << std::endl;
        exit(1);
    }
    
    // Copy to device
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Hes);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Hes);
    
    CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs.data(),
                         (n_total + 1) * sizeof(sunindextype), 
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_colvals, h_colvals.data(),
                         total_hess_nnz * sizeof(sunindextype),
                         cudaMemcpyHostToDevice));
    
    // Set fixed pattern
    SUNMatrix_cuSparse_SetFixedPattern(Hes, SUNTRUE);
}


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


std::tuple<std::vector<sunrealtype>, std::vector<sunindextype>, std::vector<sunindextype>> 
DynamicsIntegrator::getJacobian() {    
    
    // Compute fresh Jacobian at current state
    N_Vector tmp1 = N_VClone(y);
    N_Vector tmp2 = N_VClone(y);  
    N_Vector tmp3 = N_VClone(y);
    
    int retval = jacobianFunction(0.0, y, nullptr, Jac, this, tmp1, tmp2, tmp3);
    
    N_VDestroy(tmp1);
    N_VDestroy(tmp2);
    N_VDestroy(tmp3);
    
    if (retval != 0) {
        std::cerr << "Error computing Jacobian" << std::endl;
        return {};
    }
    
    // Get pointers to GPU data
    sunrealtype* d_data = SUNMatrix_cuSparse_Data(Jac);
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);
        
    // Allocate host memory
    std::vector<sunrealtype> values(TOTAL_NNZ);
    std::vector<sunindextype> row_ptrs(N_TOTAL_STATES + 1);
    std::vector<sunindextype> col_vals(TOTAL_NNZ);
    
    // Single batch of GPU->CPU transfers
    cudaMemcpy(values.data(), d_data, TOTAL_NNZ * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_ptrs.data(), d_rowptrs, (N_TOTAL_STATES + 1) * sizeof(sunindextype), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_vals.data(), d_colvals, TOTAL_NNZ * sizeof(sunindextype), cudaMemcpyDeviceToHost);
    
    return std::make_tuple(std::move(values), std::move(row_ptrs), std::move(col_vals));
}

std::tuple<std::vector<sunrealtype>, std::vector<sunindextype>, std::vector<sunindextype>> 
DynamicsIntegrator::getHessian() {
    
    // Get device state data
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* d_hess_data = SUNMatrix_cuSparse_Data(Hes);
    
    // Launch kernel: one thread per system, one block dimension per equation
    dim3 blockSize(32, 7);  // 32 systems, 7 equations
    dim3 gridSize((n_stp + blockSize.x - 1) / blockSize.x, 1);
    
    sparseHessian<<<gridSize, blockSize>>>(n_stp, d_hess_data, d_y);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Full Hessian kernel error: " << cudaGetErrorString(err) << std::endl;
        return {};
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get pointers to GPU data
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Hes);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Hes);
        
    // Allocate host memory and copy from device
    std::vector<sunrealtype> values(TOTAL_HESSIAN_NNZ);
    std::vector<sunindextype> row_ptrs(n_total + 1);
    std::vector<sunindextype> col_vals(TOTAL_HESSIAN_NNZ);
    
    CUDA_CHECK(cudaMemcpy(values.data(), d_hess_data, TOTAL_HESSIAN_NNZ * sizeof(sunrealtype), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(row_ptrs.data(), d_rowptrs, (n_total + 1) * sizeof(sunindextype), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(col_vals.data(), d_colvals, TOTAL_HESSIAN_NNZ * sizeof(sunindextype), 
                         cudaMemcpyDeviceToHost));
    
    return std::make_tuple(std::move(values), std::move(row_ptrs), std::move(col_vals));
}


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