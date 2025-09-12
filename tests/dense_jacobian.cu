#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>

// SUNDIALS headers
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>

// Define constants for batch processing
#define N_STATES_PER_SYSTEM 7
#define N_STEPS 100
#define N_TOTAL_STATES (N_STATES_PER_SYSTEM * N_STEPS)
#define DELTA_T 0.01  // Time step for integration

// Default physical parameters
#define I_X 0.5                   
#define I_Y 1.2
#define I_Z 0.8

// Sparsity constants
#define NNZ_PER_STEP 30  // 4*6 + 3*2 = 24 + 6 = 30 nonzeros per STEP
#define TOTAL_NNZ (NNZ_PER_STEP * N_STEPS)

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Timing utility class
class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMs() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// Step parameters structure
struct StepParams {
    // Control inputs
    sunrealtype tau_x, tau_y, tau_z;
    // Initial conditions [q0, q1, q2, q3, wx, wy, wz]
    sunrealtype q0, q1, q2, q3, wx, wy, wz;
};

// Device arrays for step parameters (constant during integration)
__device__ StepParams* d_step_params;

// Batch RHS function - processes all steps in parallel
__global__ void spacecraftAttitudeBatchRHS(int n_total, sunrealtype* y, sunrealtype* ydot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step_id = tid / N_STATES_PER_SYSTEM;
    int state_idx = tid % N_STATES_PER_SYSTEM;
    
    // Safety checks
    if (step_id >= N_STEPS || tid >= n_total) return;
    
    // Only compute once per step (when processing the first state)
    if (state_idx != 0) return;
    
    // Get step parameters
    StepParams params = d_step_params[step_id];
    
    // Calculate base index for this step
    int base_idx = step_id * N_STATES_PER_SYSTEM;
    
    // Extract state variables for this step
    sunrealtype q0 = y[base_idx + 0];
    sunrealtype q1 = y[base_idx + 1];
    sunrealtype q2 = y[base_idx + 2];
    sunrealtype q3 = y[base_idx + 3];
    sunrealtype wx = y[base_idx + 4];
    sunrealtype wy = y[base_idx + 5];
    sunrealtype wz = y[base_idx + 6];
    
    // Use step-specific parameters
    const sunrealtype tau_x = params.tau_x;
    const sunrealtype tau_y = params.tau_y;
    const sunrealtype tau_z = params.tau_z;
    
    const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Quaternion dynamics
    ydot[base_idx + 0] = 0.5 * (-q1*wx - q2*wy - q3*wz);
    ydot[base_idx + 1] = 0.5 * ( q0*wx - q3*wy + q2*wz);
    ydot[base_idx + 2] = 0.5 * ( q3*wx + q0*wy - q1*wz);
    ydot[base_idx + 3] = 0.5 * (-q2*wx + q1*wy + q0*wz);
    
    // Angular velocity dynamics
    sunrealtype Iw_x = Ix * wx;
    sunrealtype Iw_y = Iy * wy;
    sunrealtype Iw_z = Iz * wz;
    
    sunrealtype cross_x = wy * Iw_z - wz * Iw_y;
    sunrealtype cross_y = wz * Iw_x - wx * Iw_z;
    sunrealtype cross_z = wx * Iw_y - wy * Iw_x;
    
    ydot[base_idx + 4] = Ix_inv * (tau_x - cross_x);
    ydot[base_idx + 5] = Iy_inv * (tau_y - cross_y);
    ydot[base_idx + 6] = Iz_inv * (tau_z - cross_z);
}

// Batch Jacobian function - processes all steps in parallel
__global__ void denseBatchJacobian(int n_blocks, sunrealtype* block_data, 
                                   sunrealtype* y) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if (block_id >= n_blocks) return;
    
    // Each block has N_STATES_PER_SYSTEM * N_STATES_PER_SYSTEM entries
    int block_size = N_STATES_PER_SYSTEM * N_STATES_PER_SYSTEM;
    sunrealtype* block_jac = block_data + block_id * block_size;
    
    // Calculate state index for this block
    int base_state_idx = block_id * N_STATES_PER_SYSTEM;
    
    // Extract state variables for this block
    sunrealtype q0 = y[base_state_idx + 0];
    sunrealtype q1 = y[base_state_idx + 1];
    sunrealtype q2 = y[base_state_idx + 2];
    sunrealtype q3 = y[base_state_idx + 3];
    sunrealtype wx = y[base_state_idx + 4];
    sunrealtype wy = y[base_state_idx + 5];
    sunrealtype wz = y[base_state_idx + 6];
    
    const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Initialize all entries to zero (cooperative among threads)
    for (int i = tid; i < block_size; i += blockDim.x) {
        block_jac[i] = 0.0;
    }
    
    __syncthreads();
    
    // Fill Jacobian entries - each thread handles one or more rows
    if (tid < N_STATES_PER_SYSTEM) {
        int row_offset = tid * N_STATES_PER_SYSTEM;
        
        switch(tid) {
            case 0: // q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
                block_jac[row_offset + 1] = -0.5 * wx;
                block_jac[row_offset + 2] = -0.5 * wy;
                block_jac[row_offset + 3] = -0.5 * wz;
                block_jac[row_offset + 4] = -0.5 * q1;
                block_jac[row_offset + 5] = -0.5 * q2;
                block_jac[row_offset + 6] = -0.5 * q3;
                break;
                
            case 1: // q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
                block_jac[row_offset + 0] =  0.5 * wx;
                block_jac[row_offset + 2] =  0.5 * wz;
                block_jac[row_offset + 3] = -0.5 * wy;
                block_jac[row_offset + 4] =  0.5 * q0;
                block_jac[row_offset + 5] = -0.5 * q3;
                block_jac[row_offset + 6] =  0.5 * q2;
                break;
                
            case 2: // q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
                block_jac[row_offset + 0] =  0.5 * wy;
                block_jac[row_offset + 1] = -0.5 * wz;
                block_jac[row_offset + 3] =  0.5 * wx;
                block_jac[row_offset + 4] =  0.5 * q3;
                block_jac[row_offset + 5] =  0.5 * q0;
                block_jac[row_offset + 6] = -0.5 * q1;
                break;
                
            case 3: // q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
                block_jac[row_offset + 0] =  0.5 * wz;
                block_jac[row_offset + 1] =  0.5 * wy;
                block_jac[row_offset + 2] = -0.5 * wx;
                block_jac[row_offset + 4] = -0.5 * q2;
                block_jac[row_offset + 5] =  0.5 * q1;
                block_jac[row_offset + 6] =  0.5 * q0;
                break;
                
            case 4: // wx_dot
                block_jac[row_offset + 5] = -Ix_inv * (Iz - Iy) * wz;
                block_jac[row_offset + 6] = -Ix_inv * (Iz - Iy) * wy;
                break;
                
            case 5: // wy_dot
                block_jac[row_offset + 4] = -Iy_inv * (Ix - Iz) * wz;
                block_jac[row_offset + 6] = -Iy_inv * (Ix - Iz) * wx;
                break;
                
            case 6: // wz_dot
                block_jac[row_offset + 4] = -Iz_inv * (Iy - Ix) * wy;
                block_jac[row_offset + 5] = -Iz_inv * (Iy - Ix) * wx;
                break;
        }
    }
}

class BatchSpacecraftSolver {
private:
    void* cvode_mem;
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector y;
    SUNContext sunctx;
    int n_total, nnz;
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
    std::vector<StepParams> h_step_params;
    StepParams* d_step_params_ptr;
    
    PrecisionTimer timer;
    double total_solve_time;
    double setup_time;
    
public:
    BatchSpacecraftSolver() : n_total(N_TOTAL_STATES), total_solve_time(0), setup_time(0) {
        timer.start();
        
        int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
        if (retval != 0) {
            std::cerr << "Error creating SUNDIALS context" << std::endl;
            exit(1);
        }
        
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        y = N_VNew_Cuda(n_total, sunctx);
        if (!y) {
            std::cerr << "Error creating CUDA vector" << std::endl;
            exit(1);
        }
        
        // Create block-diagonal sparse matrix structure
        // For BCSR: total NNZ = blocks * nnz_per_block
        int nnz_per_block = N_STATES_PER_SYSTEM * N_STATES_PER_SYSTEM;
        nnz = N_STEPS * nnz_per_block;  // Total NNZ for all blocks
        
        A = SUNMatrix_cuSparse_NewBlockCSR(N_STEPS, N_STATES_PER_SYSTEM, 
                                        N_STATES_PER_SYSTEM, nnz_per_block, 
                                        cusparse_handle, sunctx);
        if (!A) {
            std::cerr << "Error creating cuSPARSE matrix" << std::endl;
            exit(1);
        }
        
        printf("Block matrix created:\n");
        printf("  Number of blocks: %d\n", SUNMatrix_cuSparse_NumBlocks(A));
        printf("  Block dimensions: %d x %d\n", 
            SUNMatrix_cuSparse_BlockRows(A), SUNMatrix_cuSparse_BlockColumns(A));
        printf("  Non-zeros per block: %d\n", SUNMatrix_cuSparse_BlockNNZ(A));
        printf("  Total non-zeros: %d\n", nnz);
        
        setupJacobianDenseStructure();
        
        verifyBlockStructure();
        
        // BatchQR solver is ideal for this block-diagonal structure
        LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        if (!LS) {
            std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
            exit(1);
        }
        
        allocateStepParams();
        generateRandomSteps();
        initializeCVODES();
        
        setup_time = timer.getElapsedMs();
    }
    
    ~BatchSpacecraftSolver() {
        if (cvode_mem) CVodeFree(&cvode_mem);
        if (A) SUNMatDestroy(A);
        if (LS) SUNLinSolFree(LS);
        if (y) N_VDestroy(y);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
        if (d_step_params_ptr) cudaFree(d_step_params_ptr);
        if (sunctx) SUNContext_Free(&sunctx);
    }

    void setupJacobianDenseStructure() {
        // For BCSR format, we only need to specify the structure of ONE block
        // SUNDIALS will replicate this pattern for all blocks
        const int block_nnz = N_STATES_PER_SYSTEM * N_STATES_PER_SYSTEM;
        
        // Allocate host arrays for ONE block structure
        std::vector<sunindextype> h_rowptrs(N_STATES_PER_SYSTEM + 1);
        std::vector<sunindextype> h_colvals(block_nnz);
        
        // Dense block structure - each row has N_STATES_PER_SYSTEM entries
        for (int i = 0; i < N_STATES_PER_SYSTEM; i++) {
            h_rowptrs[i] = i * N_STATES_PER_SYSTEM;
            
            // Column indices for this row (0 to N_STATES_PER_SYSTEM-1)
            for (int j = 0; j < N_STATES_PER_SYSTEM; j++) {
                h_colvals[i * N_STATES_PER_SYSTEM + j] = j;
            }
        }
        
        // Set final row pointer
        h_rowptrs[N_STATES_PER_SYSTEM] = block_nnz;
        
        printf("Block structure: %d x %d with %d non-zeros\n", 
            N_STATES_PER_SYSTEM, N_STATES_PER_SYSTEM, block_nnz);
        
        // Copy to device - these pointers represent the FIRST block's structure
        sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(A);
        sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(A);
        
        CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs.data(),
                            (N_STATES_PER_SYSTEM + 1) * sizeof(sunindextype), 
                            cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_colvals, h_colvals.data(),
                            block_nnz * sizeof(sunindextype),
                            cudaMemcpyHostToDevice));
        
        // Set fixed pattern since our sparsity structure doesn't change
        SUNMatrix_cuSparse_SetFixedPattern(A, SUNTRUE);
    }
    
    void verifyBlockStructure() {
        std::cout << "\nVerifying Block Structure:" << std::endl;
        std::cout << "Matrix type: " << 
            (SUNMatrix_cuSparse_SparseType(A) == SUNMAT_CUSPARSE_BCSR ? "BCSR" : "CSR") 
            << std::endl;
        std::cout << "Number of blocks: " << SUNMatrix_cuSparse_NumBlocks(A) << std::endl;
        std::cout << "Block rows: " << SUNMatrix_cuSparse_BlockRows(A) << std::endl;
        std::cout << "Block columns: " << SUNMatrix_cuSparse_BlockColumns(A) << std::endl;
        std::cout << "Block NNZ: " << SUNMatrix_cuSparse_BlockNNZ(A) << std::endl;
        
        // You can also print the first block's structure
        std::vector<sunindextype> h_rowptrs(N_STATES_PER_SYSTEM + 1);
        std::vector<sunindextype> h_colvals(N_STATES_PER_SYSTEM * N_STATES_PER_SYSTEM);
        
        SUNMatrix_cuSparse_CopyFromDevice(A, nullptr, h_rowptrs.data(), h_colvals.data());
        
        std::cout << "\nFirst block structure:" << std::endl;
        std::cout << "Row pointers: ";
        for (int i = 0; i <= N_STATES_PER_SYSTEM; i++) {
            std::cout << h_rowptrs[i] << " ";
        }
        std::cout << std::endl;
    }
    
    void allocateStepParams() {
        h_step_params.resize(N_STEPS);
        
        CUDA_CHECK(cudaMalloc(&d_step_params_ptr, N_STEPS * sizeof(StepParams)));
        
        // Copy device pointer to device constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(d_step_params, &d_step_params_ptr, sizeof(StepParams*)));
    }
    
    void generateRandomSteps() {
        std::random_device rd;
        //std::mt19937 gen(rd());
        std::mt19937 gen(1);

        // Distributions for random parameters
        std::uniform_real_distribution<float> torque_dist(-1, 1);
        std::uniform_real_distribution<float> quat_dist(0, 1);
        std::uniform_real_distribution<float> omega_dist(-1, 1);
        
        for (int i = 0; i < N_STEPS; i++) {
            auto& params = h_step_params[i];
            
            // Random control inputs
            params.tau_x = torque_dist(gen);
            params.tau_y = torque_dist(gen);
            params.tau_z = torque_dist(gen);
                        
            // Random initial quaternion (normalized)
            params.q0 = 1.0 + quat_dist(gen);  // q0
            params.q1 = quat_dist(gen);        // q1
            params.q2 = quat_dist(gen);        // q2
            params.q3 = quat_dist(gen);        // q3
            
            // Normalize quaternion
            sunrealtype norm = sqrt(params.q0*params.q0 +
                                  params.q1*params.q1 +
                                  params.q2*params.q2 +
                                  params.q3*params.q3);
            params.q0 /= norm;
            params.q1 /= norm;
            params.q2 /= norm;
            params.q3 /= norm;
            
            // Random initial angular velocities
            params.wx = omega_dist(gen);  // wx
            params.wy = omega_dist(gen);  // wy
            params.wz = omega_dist(gen);  // wz
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_step_params_ptr, h_step_params.data(), 
                             N_STEPS * sizeof(StepParams), cudaMemcpyHostToDevice));
    }
    
    void initializeCVODES() {
        cvode_mem = CVodeCreate(CV_BDF, sunctx);
        if (!cvode_mem) {
            std::cerr << "Error creating CVODES memory" << std::endl;
            exit(1);
        }
        
        setInitialConditions();
        
        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error initializing CVODES: " << retval << std::endl;
            exit(1);
        }
        
        CVodeSetUserData(cvode_mem, this);
        
        // Relaxed tolerances for batch processing
        retval = CVodeSStolerances(cvode_mem, 1e-10, 1e-12);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting tolerances: " << retval << std::endl;
            exit(1);
        }
        
        retval = CVodeSetLinearSolver(cvode_mem, LS, A);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting linear solver: " << retval << std::endl;
            exit(1);
        }
        
        retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting Jacobian function: " << retval << std::endl;
            exit(1);
        }
        
        CVodeSetMaxNumSteps(cvode_mem, 100000);
    }
    
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
        int n_total = N_VGetLength(y);
        
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
        // Launch with enough threads to cover all steps
        dim3 blockSize(N_STATES_PER_SYSTEM);
        dim3 gridSize(N_STEPS);
        
        spacecraftAttitudeBatchRHS<<<gridSize, blockSize>>>(n_total, y_data, ydot_data);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Batch RHS kernel error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        cudaDeviceSynchronize();
        return 0;
    }
    
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                            SUNMatrix Jac, void* user_data, 
                            N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        // Get block information
        int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
        
        // Get pointer to the data array (contains all blocks)
        sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        
        // Launch kernel with one block per spacecraft system
        dim3 blockSize(32); // Can be tuned, but should be >= N_STATES_PER_SYSTEM
        dim3 gridSize(n_blocks);
        
        denseBatchJacobian<<<gridSize, blockSize>>>(n_blocks, data, y_data);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Jacobian kernel error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        cudaDeviceSynchronize();
        return 0;
    }
    
    double solveBatchSteps(bool verbose = false) {
        if (verbose) {
            std::cout << "Solving " << N_STEPS << " steps in parallel..." << std::endl;
            std::cout << "Total states: " << n_total << std::endl;
            std::cout << "States per system: " << N_STATES_PER_SYSTEM << std::endl;
        }
        
        timer.start();
        
        // Define the time step size
        sunrealtype t_final = DELTA_T;
        
        sunrealtype t = 0.0;
        int retval = CVode(cvode_mem, t_final, y, &t, CV_NORMAL);
        
        if (retval < 0) {
            std::cerr << "CVode error: " << retval << std::endl;
            return -1;
        }
        
        total_solve_time = timer.getElapsedMs();
        
        if (verbose) {
            printBatchSolutionStats();
        }
        
        return total_solve_time;
    }
    
    void printBatchSolutionStats() {
        long int nsteps, nfevals, nlinsetups;
        CVodeGetNumSteps(cvode_mem, &nsteps);
        CVodeGetNumRhsEvals(cvode_mem, &nfevals);
        CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
        
        long int njevals, nliters;
        CVodeGetNumJacEvals(cvode_mem, &njevals);
        CVodeGetNumLinIters(cvode_mem, &nliters);
        
        std::cout << "Batch integration stats:" << std::endl;
        std::cout << "  Steps: " << nsteps << std::endl;
        std::cout << "  RHS evaluations: " << nfevals << std::endl;
        std::cout << "  Jacobian evaluations: " << njevals << std::endl;
        std::cout << "  Linear solver setups: " << nlinsetups << std::endl;
        std::cout << "  Linear iterations: " << nliters << std::endl;
    }
    
    std::vector<sunrealtype> getQuaternionNorms() {
        std::vector<sunrealtype> y_host(n_total);
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
        
        std::vector<sunrealtype> norms(N_STEPS);
        for (int i = 0; i < N_STEPS; i++) {
            int base_idx = i * N_STATES_PER_SYSTEM;
            norms[i] = sqrt(y_host[base_idx+0]*y_host[base_idx+0] + 
                           y_host[base_idx+1]*y_host[base_idx+1] + 
                           y_host[base_idx+2]*y_host[base_idx+2] + 
                           y_host[base_idx+3]*y_host[base_idx+3]);
        }
        return norms;
    }
    
    std::vector<std::vector<sunrealtype>> getAllSolutions() {
        std::vector<sunrealtype> y_host(n_total);
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        cudaMemcpy(y_host.data(), d_y, n_total * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
        
        std::vector<std::vector<sunrealtype>> solutions(N_STEPS);
        for (int i = 0; i < N_STEPS; i++) {
            solutions[i].resize(N_STATES_PER_SYSTEM);
            int base_idx = i * N_STATES_PER_SYSTEM;
            for (int j = 0; j < N_STATES_PER_SYSTEM; j++) {
                solutions[i][j] = y_host[base_idx + j];
            }
        }
        return solutions;
    }
    
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return total_solve_time; }
    
private:
    void setInitialConditions() {
        std::vector<sunrealtype> y0(n_total);
        
        for (int i = 0; i < N_STEPS; i++) {
            int base_idx = i * N_STATES_PER_SYSTEM;
            const auto& params = h_step_params[i];
            
            y0[base_idx + 0] = params.q0; // q0
            y0[base_idx + 1] = params.q1; // q1
            y0[base_idx + 2] = params.q2; // q2
            y0[base_idx + 3] = params.q3; // q3
            y0[base_idx + 4] = params.wx; // wx
            y0[base_idx + 5] = params.wy; // wy
            y0[base_idx + 6] = params.wz; // wz
        }
        
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        CUDA_CHECK(cudaMemcpy(d_y, y0.data(), n_total * sizeof(sunrealtype), cudaMemcpyHostToDevice));
    }
};

void runBatchPerformanceTest() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BATCH SPACECRAFT TRAJECTORY SOLVER PERFORMANCE TEST" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "Number of steps: " << N_STEPS << std::endl;
    std::cout << "States per step: " << N_STATES_PER_SYSTEM << std::endl;
    std::cout << "Total system size: " << N_TOTAL_STATES << " states" << std::endl;
    std::cout << "sunrealtype size: " << sizeof(sunrealtype) << " bytes" << std::endl;
    
    std::cout << "\nRunning batch simulation..." << std::endl;
    
    BatchSpacecraftSolver solver;
    std::cout << "Batch solver initialized." << std::endl;
    double solve_time = solver.solveBatchSteps(true);
    
    double setup_time = solver.getSetupTime();
    auto quaternion_norms = solver.getQuaternionNorms();
    
    // Calculate norm statistics
    double avg_norm = std::accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / N_STEPS;
    double max_deviation = 0.0;
    for (auto norm : quaternion_norms) {
        max_deviation = std::max(max_deviation, std::abs(static_cast<double>(norm) - 1.0));
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BATCH PERFORMANCE RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "  Setup time: " << setup_time << " ms" << std::endl;
    std::cout << "  Solve time: " << solve_time << " ms" << std::endl;
    std::cout << "  Total time: " << (setup_time + solve_time) << " ms" << std::endl;
    std::cout << "  Time per step: " << solve_time / N_STEPS << " ms" << std::endl;
    
    std::cout << "\nSolution Quality:" << std::endl;
    std::cout << "  Average quaternion norm: " << std::setprecision(6) << avg_norm << std::endl;
    std::cout << "  Maximum norm deviation: " << std::setprecision(2) << (max_deviation * 100.0) << "%" << std::endl;
    
    // Show a few sample steps
    auto solutions = solver.getAllSolutions();
    std::cout << "\nSample final states (first 3 steps):" << std::endl;
    for (int i = 0; i < std::min(3, N_STEPS); i++) {
        std::cout << "  Step " << i << ": quat=[" << std::setprecision(4)
                  << solutions[i][0] << "," << solutions[i][1] << "," 
                  << solutions[i][2] << "," << solutions[i][3] << "], "
                  << "ω=[" << solutions[i][4] << "," << solutions[i][5] << "," 
                  << solutions[i][6] << "]" << std::endl;
    }
}

int main() {
    try {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
        std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max blocks per grid: " << prop.maxGridSize[0] << std::endl;
        
        runBatchPerformanceTest();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// NEXT STEPS:
// 1. EXPLORE FURTHER OPTIMIZATIONS FOR BATCH PROCESSING
// 2. CONSIDER USING CUDA STREAMS FOR ASYNCHRONOUS EXECUTION

//nvcc -o dense_jacobian dense_jacobian.cu     -I$SUNDIALS_DIR/include     -L$SUNDIALS_DIR/lib     -lsundials_cvodes     -lsundials_nvecserial     -lsundials_nveccuda     -lsundials_sunmatrixcusparse     -lsundials_sunlinsolcusolversp     -lcudart -lcusparse -lcusolver -lsundials_core