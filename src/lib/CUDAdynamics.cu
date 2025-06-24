#include <cuda_dynamics.h>

// Device arrays for step parameters (constant during integration)
__device__ StepParams* d_step_params;

// Optimized RHS kernel with better memory access patterns
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot,
                           int systems_per_block = 4) {
    
    // Use shared memory for better memory bandwidth
    __shared__ sunrealtype s_constants[8];  // Store frequently used constants
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int system_id = tid / n_states;
    
    // Initialize shared constants once per block
    if (threadIdx.x < 8) {
        s_constants[0] = 1.0 / i_x;  // Ix_inv
        s_constants[1] = 1.0 / i_y;  // Iy_inv  
        s_constants[2] = 1.0 / i_z;  // Iz_inv
        s_constants[3] = 0.5;        // Half for quaternion derivatives
    }
    __syncthreads();
    
    if (system_id >= n_stp || tid >= n_total) return;
    
    // Process multiple systems per thread block for better cache utilization
    for (int sys = system_id; sys < min(system_id + systems_per_block, n_stp); 
         sys += gridDim.x * blockDim.x / n_states) {
        
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
        StepParams params = d_step_params[sys];
        
        // Use shared memory constants
        const sunrealtype half = s_constants[3];
        const sunrealtype Ix_inv = s_constants[0];
        const sunrealtype Iy_inv = s_constants[1]; 
        const sunrealtype Iz_inv = s_constants[2];
        
        // Compute derivatives with optimized math functions
        sunrealtype derivs[n_states];
        
        // Quaternion derivatives (vectorized)
        derivs[0] = half * (-q1*wx - q2*wy - q3*wz);
        derivs[1] = half * ( q0*wx - q3*wy + q2*wz);
        derivs[2] = half * ( q3*wx + q0*wy - q1*wz);
        derivs[3] = half * (-q2*wx + q1*wy + q0*wz);
        
        // Angular velocity derivatives with FMA operations
        sunrealtype Iw_x = i_x * wx, Iw_y = i_y * wy, Iw_z = i_z * wz;
        
        derivs[4] = Ix_inv * (params.tau_x - (wy * Iw_z - wz * Iw_y));
        derivs[5] = Iy_inv * (params.tau_y - (wz * Iw_x - wx * Iw_z));
        derivs[6] = Iz_inv * (params.tau_z - (wx * Iw_y - wy * Iw_x));
        
        // Coalesced memory stores
        #pragma unroll
        for (int i = 0; i < n_states; i++) {
            ydot[base_idx + i] = derivs[i];
        }
    }
}

// Sparse Jacobian kernel
__global__ void sparseBatchJacobian(int n_blocks, sunrealtype* block_data, 
                                    sunrealtype* y) {
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
    
    const sunrealtype Ix = i_x, Iy = i_y, Iz = i_z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Fill sparse Jacobian entries in CSR order
    int idx = 0;
    
    // Row 0: q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] = 0.0;        // J[0,0] diagonal
    block_jac[idx++] = -0.5 * wx;  // J[0,1]
    block_jac[idx++] = -0.5 * wy;  // J[0,2]
    block_jac[idx++] = -0.5 * wz;  // J[0,3]
    block_jac[idx++] = -0.5 * q1;  // J[0,4]
    block_jac[idx++] = -0.5 * q2;  // J[0,5]
    block_jac[idx++] = -0.5 * q3;  // J[0,6]
    
    // Row 1: q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] =  0.5 * wx;  // J[1,0]
    block_jac[idx++] = 0.0;        // J[1,1] diagonal
    block_jac[idx++] =  0.5 * wz;  // J[1,2]
    block_jac[idx++] = -0.5 * wy;  // J[1,3]
    block_jac[idx++] =  0.5 * q0;  // J[1,4]
    block_jac[idx++] = -0.5 * q3;  // J[1,5]
    block_jac[idx++] =  0.5 * q2;  // J[1,6]
    
    // Row 2: q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] =  0.5 * wy;  // J[2,0]
    block_jac[idx++] = -0.5 * wz;  // J[2,1]
    block_jac[idx++] = 0.0;        // J[2,2] diagonal
    block_jac[idx++] =  0.5 * wx;  // J[2,3]
    block_jac[idx++] =  0.5 * q3;  // J[2,4]
    block_jac[idx++] =  0.5 * q0;  // J[2,5]
    block_jac[idx++] = -0.5 * q1;  // J[2,6]
    
    // Row 3: q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
    // Columns: 0,1,2,3,4,5,6
    block_jac[idx++] =  0.5 * wz;  // J[3,0]
    block_jac[idx++] =  0.5 * wy;  // J[3,1]
    block_jac[idx++] = -0.5 * wx;  // J[3,2]
    block_jac[idx++] = 0.0;        // J[3,3] diagonal
    block_jac[idx++] = -0.5 * q2;  // J[3,4]
    block_jac[idx++] =  0.5 * q1;  // J[3,5]
    block_jac[idx++] =  0.5 * q0;  // J[3,6]
    
    // Row 4: wx_dot
    // Columns: 4,5,6
    block_jac[idx++] = 0.0;                       // J[4,4] diagonal
    block_jac[idx++] = -Ix_inv * (Iz - Iy) * wz;  // J[4,5]
    block_jac[idx++] = -Ix_inv * (Iz - Iy) * wy;  // J[4,6]
    
    // Row 5: wy_dot
    // Columns: 4,5,6
    block_jac[idx++] = -Iy_inv * (Ix - Iz) * wz;  // J[5,4]
    block_jac[idx++] = 0.0;                       // J[5,5] diagonal
    block_jac[idx++] = -Iy_inv * (Ix - Iz) * wx;  // J[5,6]
    
    // Row 6: wz_dot
    // Columns: 4,5,6
    block_jac[idx++] = -Iz_inv * (Iy - Ix) * wy;  // J[6,4]
    block_jac[idx++] = -Iz_inv * (Iy - Ix) * wx;  // J[6,5]
    block_jac[idx++] = 0.0;                       // J[6,6] diagonal
}

// Constructor - performs one-time setup
OptimizedDynamicsIntegrator::OptimizedDynamicsIntegrator(bool verb) : n_total(N_TOTAL_STATES), setup_time(0), solve_time(0), verbose(verb) {
    timer.start();
    
    // Initialize SUNDIALS context
    int retval = SUNContext_Create(NULL, &sunctx);
    if (retval != 0) {
        std::cerr << "Error creating SUNDIALS context" << std::endl;
        exit(1);
    }
    
    // Create CUDA handles
    cusparseCreate(&cusparse_handle);
    cusolverSpCreate(&cusolver_handle);
    
    // Initialize minimal CUDA streams
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
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
    
    A = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, 
                                      n_states, NNZ_PER_BLOCK, 
                                      cusparse_handle, sunctx);
    if (!A) {
        std::cerr << "Error creating cuSPARSE matrix" << std::endl;
        exit(1);
    }
    
    setupSparseJacobianStructure();
    
    // BatchQR solver works with sparse blocks
    LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
    if (!LS) {
        std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
        exit(1);
    }
    
    // Allocate device memory for step parameters
    CUDA_CHECK(cudaMalloc(&d_step_params_ptr, n_stp * sizeof(StepParams)));
    
    // Copy device pointer to device constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_step_params, &d_step_params_ptr, sizeof(StepParams*)));
    
    initializeCVODES();
    
    setup_time = timer.getElapsedMs();
}

// Destructor - cleanup
OptimizedDynamicsIntegrator::~OptimizedDynamicsIntegrator() {
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (A) SUNMatDestroy(A);
    if (LS) SUNLinSolFree(LS);
    if (y) N_VDestroy(y);
    
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
    if (d_step_params_ptr) cudaFree(d_step_params_ptr);
    if (h_y_pinned) cudaFreeHost(h_y_pinned);
    if (sunctx) SUNContext_Free(&sunctx);
}

void OptimizedDynamicsIntegrator::setupSparseJacobianStructure() {
    std::vector<sunindextype> h_rowptrs(n_states + 1);
    std::vector<sunindextype> h_colvals(NNZ_PER_BLOCK);
    
    // Build CSR structure (columns must be in ascending order)
    int nnz_count = 0;
    
    // Row 0: columns 0,1,2,3,4,5,6
    h_rowptrs[0] = nnz_count;
    for (int j = 0; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 1: columns 0,1,2,3,4,5,6
    h_rowptrs[1] = nnz_count;
    for (int j = 0; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 2: columns 0,1,2,3,4,5,6
    h_rowptrs[2] = nnz_count;
    for (int j = 0; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 3: columns 0,1,2,3,4,5,6
    h_rowptrs[3] = nnz_count;
    for (int j = 0; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 4: columns 4,5,6
    h_rowptrs[4] = nnz_count;
    for (int j = 4; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 5: columns 4,5,6
    h_rowptrs[5] = nnz_count;
    for (int j = 4; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Row 6: columns 4,5,6
    h_rowptrs[6] = nnz_count;
    for (int j = 4; j < 7; j++) h_colvals[nnz_count++] = j;
    
    // Final row pointer
    h_rowptrs[7] = nnz_count;
    
    // Copy to device using stream for async transfer
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(A);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(A);
    
    CUDA_CHECK(cudaMemcpyAsync(d_rowptrs, h_rowptrs.data(),
                             (n_states + 1) * sizeof(sunindextype), 
                             cudaMemcpyHostToDevice, streams[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_colvals, h_colvals.data(),
                             NNZ_PER_BLOCK * sizeof(sunindextype),
                             cudaMemcpyHostToDevice, streams[0]));
    
    // Synchronize to ensure transfer is complete
    cudaStreamSynchronize(streams[0]);
    
    // Set fixed pattern since our sparsity structure doesn't change
    SUNMatrix_cuSparse_SetFixedPattern(A, SUNTRUE);
}

void OptimizedDynamicsIntegrator::initializeCVODES() {
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (!cvode_mem) {
        std::cerr << "Error creating CVODES memory" << std::endl;
        exit(1);
    }
}

void OptimizedDynamicsIntegrator::setInitialConditions(const std::vector<std::vector<sunrealtype>>& initial_states) {
    std::vector<sunrealtype> y0(n_total);
    
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        const auto& state = initial_states[i];
        
        y0[base_idx + 0] = state[0]; // q0
        y0[base_idx + 1] = state[1]; // q1
        y0[base_idx + 2] = state[2]; // q2
        y0[base_idx + 3] = state[3]; // q3
        y0[base_idx + 4] = state[4]; // wx
        y0[base_idx + 5] = state[5]; // wy
        y0[base_idx + 6] = state[6]; // wz
    }
    
    // Use pinned memory for faster transfers
    memcpy(h_y_pinned, y0.data(), n_total * sizeof(sunrealtype));
    
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpyAsync(d_y, h_y_pinned, n_total * sizeof(sunrealtype), 
                              cudaMemcpyHostToDevice, streams[0]));
    cudaStreamSynchronize(streams[0]);
}

void OptimizedDynamicsIntegrator::setStepParams(const std::vector<StepParams>& step_params) {
    // Copy to device using streams for async transfer
    CUDA_CHECK(cudaMemcpyAsync(d_step_params_ptr, step_params.data(), 
                             n_stp * sizeof(StepParams), cudaMemcpyHostToDevice,
                             streams[0]));
    cudaStreamSynchronize(streams[0]);
}

int OptimizedDynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    OptimizedDynamicsIntegrator* solver = static_cast<OptimizedDynamicsIntegrator*>(user_data);
    int n_total = N_VGetLength(y);
    
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
    
    // Get current stream
    cudaStream_t current_stream = solver->streams[solver->current_stream];
    
    // Find optimal configuration
    int blockSize = 128;  // Based on occupancy optimization
    int gridSize = (n_stp + 3) / 4;  // Systems per block = 4
    
    // Launch optimized kernel with stream
    dynamicsRHS<<<gridSize, blockSize, 0, current_stream>>>(n_total, y_data, ydot_data, 4);
    
    // Increment stream counter
    solver->current_stream = (solver->current_stream + 1) % N_STREAMS;
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RHS kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    return 0;
}

int OptimizedDynamicsIntegrator::jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                           SUNMatrix Jac, void* user_data, 
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    OptimizedDynamicsIntegrator* solver = static_cast<OptimizedDynamicsIntegrator*>(user_data);
    
    // Get block information
    int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
    
    // Get pointer to the data array (contains all blocks)
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    
    // Get current stream
    cudaStream_t current_stream = solver->streams[solver->current_stream];
    
    // Launch kernel with one block per spacecraft system, using stream
    dim3 blockSize(32);
    dim3 gridSize(n_blocks);
    
    sparseBatchJacobian<<<gridSize, blockSize, 0, current_stream>>>(n_blocks, data, y_data);
    
    // Increment stream counter
    solver->current_stream = (solver->current_stream + 1) % N_STREAMS;
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Jacobian kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    return 0;
}

double OptimizedDynamicsIntegrator::solve(const std::vector<std::vector<sunrealtype>>& initial_states, 
                                       const std::vector<StepParams>& step_params,
                                       sunrealtype delta_t) {
    
    // Validate inputs
    if (!validateInputs(initial_states, step_params)) {
        return -1.0;
    }
    
    timer.start();
    
    // Set up for this solve
    setInitialConditions(initial_states);
    setStepParams(step_params);
    
    // Initialize CVODES for this solve
    int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error reinitializing CVODES: " << retval << std::endl;
        return -1.0;
    }
    
    CVodeSetUserData(cvode_mem, this);
    
    // Relaxed tolerances for batch processing
    retval = CVodeSStolerances(cvode_mem, 1e-6, 1e-8);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting tolerances: " << retval << std::endl;
        return -1.0;
    }
    
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting linear solver: " << retval << std::endl;
        return -1.0;
    }
    
    retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
    if (retval != CV_SUCCESS) {
        std::cerr << "Error setting Jacobian function: " << retval << std::endl;
        return -1.0;
    }
    
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    
    // Solve
    sunrealtype t = 0.0;
    retval = CVode(cvode_mem, delta_t, y, &t, CV_NORMAL);
    
    if (retval < 0) {
        std::cerr << "CVode error: " << retval << std::endl;
        return -1.0;
    }
    
    // Synchronize all streams
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    solve_time = timer.getElapsedMs();

    if (verbose) {
        printSolutionStats();
    }
    return solve_time;
}

bool OptimizedDynamicsIntegrator::validateInputs(const std::vector<std::vector<sunrealtype>>& initial_states, 
                                               const std::vector<StepParams>& step_params) {
    if (initial_states.size() != n_stp) {
        std::cerr << "Error: initial_states size (" << initial_states.size() 
                  << ") does not match n_stp (" << n_stp << ")" << std::endl;
        return false;
    }
    
    if (step_params.size() != n_stp) {
        std::cerr << "Error: step_params size (" << step_params.size() 
                  << ") does not match n_stp (" << n_stp << ")" << std::endl;
        return false;
    }
    
    for (int i = 0; i < n_stp; i++) {
        if (initial_states[i].size() != n_states) {
            std::cerr << "Error: initial_states[" << i << "] size (" << initial_states[i].size() 
                      << ") does not match n_states (" << n_states << ")" << std::endl;
            return false;
        }
        
        // Check quaternion normalization
        sunrealtype quat_norm_sq = initial_states[i][0]*initial_states[i][0] + 
                                  initial_states[i][1]*initial_states[i][1] + 
                                  initial_states[i][2]*initial_states[i][2] + 
                                  initial_states[i][3]*initial_states[i][3];
        if (abs(quat_norm_sq - 1.0) > 1e-6) {
            std::cerr << "Warning: initial_states[" << i << "] quaternion not normalized (norm^2 = " 
                      << quat_norm_sq << ")" << std::endl;
        }
    }
    
    return true;
}

std::vector<std::vector<sunrealtype>> OptimizedDynamicsIntegrator::getAllSolutions() {
    std::vector<sunrealtype> y_host(n_total);
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
    
    std::vector<std::vector<sunrealtype>> solutions(n_stp);
    for (int i = 0; i < n_stp; i++) {
        solutions[i].resize(n_states);
        int base_idx = i * n_states;
        for (int j = 0; j < n_states; j++) {
            solutions[i][j] = y_host[base_idx + j];
        }
    }
    return solutions;
}

std::vector<sunrealtype> OptimizedDynamicsIntegrator::getQuaternionNorms() {
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

void OptimizedDynamicsIntegrator::printSolutionStats() {
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