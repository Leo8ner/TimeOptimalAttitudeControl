// Batched spacecraft attitude dynamics integrator for trajectory processing
// Solves multiple ODE systems simultaneously on GPU

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
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <sunlinsol/sunlinsol_dense.h>

using namespace std;

// Constants
#define N_STATES 7
#define I_X 1.2                    
#define I_Y 1.0
#define I_Z 0.8
#define TAU_X 0.1
#define TAU_Y 0.05
#define TAU_Z 0.02

#define NNZ 49  // Number of non-zero entries in the sparse Jacobian per trajectory

// Error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Trajectory point structure
struct TrajectoryPoint {
    vector<sunrealtype> initial_state;  // 7 elements: [q0,q1,q2,q3,wx,wy,wz]
    sunrealtype dt;                          // Integration time step
    
    TrajectoryPoint(const vector<sunrealtype>& state, sunrealtype time_step) 
        : initial_state(state), dt(time_step) {}
};

// Timing utility
class PrecisionTimer {
private:
    chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double getElapsedMs() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// Batched RHS kernel - processes multiple trajectories in parallel
__global__ void batchedRHS_kernel(int n_batch, int n_states, sunrealtype* y_batch, sunrealtype* ydot_batch) {
    int trajectory_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (trajectory_id >= n_batch || thread_id != 0) return;
    
    // Calculate offset for this trajectory
    int offset = trajectory_id * n_states;
    
    // Extract state variables for this trajectory
    sunrealtype* y = y_batch + offset;
    sunrealtype* ydot = ydot_batch + offset;
    
    sunrealtype q0 = y[0], q1 = y[1], q2 = y[2], q3 = y[3];
    sunrealtype wx = y[4], wy = y[5], wz = y[6];
    
    // Control inputs (could be trajectory-dependent in future)
    const sunrealtype tau_x = TAU_X;
    const sunrealtype tau_y = TAU_Y;
    const sunrealtype tau_z = TAU_Z;
    
    // Moment of inertia values
    const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Quaternion dynamics
    ydot[0] = 0.5 * (-q1*wx - q2*wy - q3*wz);
    ydot[1] = 0.5 * ( q0*wx - q3*wy + q2*wz);
    ydot[2] = 0.5 * ( q3*wx + q0*wy - q1*wz);
    ydot[3] = 0.5 * (-q2*wx + q1*wy + q0*wz);
    
    // Angular velocity dynamics
    sunrealtype Iw_x = Ix * wx;
    sunrealtype Iw_y = Iy * wy;
    sunrealtype Iw_z = Iz * wz;
    
    sunrealtype cross_x = wy * Iw_z - wz * Iw_y;
    sunrealtype cross_y = wz * Iw_x - wx * Iw_z;
    sunrealtype cross_z = wx * Iw_y - wy * Iw_x;
    
    ydot[4] = Ix_inv * (tau_x - cross_x);
    ydot[5] = Iy_inv * (tau_y - cross_y);
    ydot[6] = Iz_inv * (tau_z - cross_z);
}

// Batched dense Jacobian kernel
__global__ void batchedDenseJacobian_kernel(int n_batch, int n_states, 
                                           sunrealtype* data_batch, sunrealtype* y_batch) {
    int trajectory_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (trajectory_id >= n_batch || thread_id != 0) return;
    
    // Calculate offsets
    int state_offset = trajectory_id * n_states;
    int jac_offset = trajectory_id * n_states * n_states;  // Full 7x7 matrix per trajectory
    
    sunrealtype* y = y_batch + state_offset;
    sunrealtype* data = data_batch + jac_offset;
    
    // Extract state variables
    sunrealtype q0 = y[0], q1 = y[1], q2 = y[2], q3 = y[3];
    sunrealtype wx = y[4], wy = y[5], wz = y[6];
    
    const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Initialize all entries to zero
    for (int i = 0; i < n_states * n_states; i++) {
        data[i] = 0.0;
    }
    
    // Fill dense Jacobian entries
    // Note: Using row-major indexing: data[row*n_states + col]
    
    // Row 0: q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
    data[0*n_states + 1] = -0.5 * wx;   // d/dq1
    data[0*n_states + 2] = -0.5 * wy;   // d/dq2
    data[0*n_states + 3] = -0.5 * wz;   // d/dq3
    data[0*n_states + 4] = -0.5 * q1;   // d/dwx
    data[0*n_states + 5] = -0.5 * q2;   // d/dwy
    data[0*n_states + 6] = -0.5 * q3;   // d/dwz
    
    // Row 1: q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
    data[1*n_states + 0] =  0.5 * wx;   // d/dq0
    data[1*n_states + 2] =  0.5 * wz;   // d/dq2
    data[1*n_states + 3] = -0.5 * wy;   // d/dq3
    data[1*n_states + 4] =  0.5 * q0;   // d/dwx
    data[1*n_states + 5] = -0.5 * q3;   // d/dwy
    data[1*n_states + 6] =  0.5 * q2;   // d/dwz
    
    // Row 2: q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
    data[2*n_states + 0] =  0.5 * wy;   // d/dq0
    data[2*n_states + 1] = -0.5 * wz;   // d/dq1
    data[2*n_states + 3] =  0.5 * wx;   // d/dq3
    data[2*n_states + 4] =  0.5 * q3;   // d/dwx
    data[2*n_states + 5] =  0.5 * q0;   // d/dwy
    data[2*n_states + 6] = -0.5 * q1;   // d/dwz
    
    // Row 3: q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
    data[3*n_states + 0] =  0.5 * wz;   // d/dq0
    data[3*n_states + 1] =  0.5 * wy;   // d/dq1
    data[3*n_states + 2] = -0.5 * wx;   // d/dq2
    data[3*n_states + 4] = -0.5 * q2;   // d/dwx
    data[3*n_states + 5] =  0.5 * q1;   // d/dwy
    data[3*n_states + 6] =  0.5 * q0;   // d/dwz
    
    // Row 4: wx_dot = Ix_inv * (tau_x - wy*wz*(Iz - Iy))
    data[4*n_states + 5] = -Ix_inv * (Iz - Iy) * wz;  // d/dwy
    data[4*n_states + 6] = -Ix_inv * (Iz - Iy) * wy;  // d/dwz
    
    // Row 5: wy_dot = Iy_inv * (tau_y - wx*wz*(Ix - Iz))
    data[5*n_states + 4] = -Iy_inv * (Ix - Iz) * wz;  // d/dwx
    data[5*n_states + 6] = -Iy_inv * (Ix - Iz) * wx;  // d/dwz
    
    // Row 6: wz_dot = Iz_inv * (tau_z - wx*wy*(Iy - Ix))
    data[6*n_states + 4] = -Iz_inv * (Iy - Ix) * wy;  // d/dwx
    data[6*n_states + 5] = -Iz_inv * (Iy - Ix) * wx;  // d/dwy
}

// Batched sparse Jacobian kernel
__global__ void batchedSparseJacobian_kernel(int n_batch, int n_states, int nnz_per_matrix, 
                                             sunrealtype* data_batch, sunrealtype* y_batch) {
    int trajectory_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (trajectory_id >= n_batch || thread_id != 0) return;
    
    // Calculate offsets
    int state_offset = trajectory_id * n_states;
    int jac_offset = trajectory_id * nnz_per_matrix;
    
    sunrealtype* y = y_batch + state_offset;
    sunrealtype* data = data_batch + jac_offset;
    
    // Extract state variables
    sunrealtype q0 = y[0], q1 = y[1], q2 = y[2], q3 = y[3];
    sunrealtype wx = y[4], wy = y[5], wz = y[6];
    
    const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
    const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
    
    // Fill sparse Jacobian entries (same pattern as before, but for this trajectory)
    // Row 0: q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
    data[0] = -0.5 * wx;   // d/dq1
    data[1] = -0.5 * wy;   // d/dq2
    data[2] = -0.5 * wz;   // d/dq3
    data[3] = -0.5 * q1;   // d/dwx
    data[4] = -0.5 * q2;   // d/dwy
    data[5] = -0.5 * q3;   // d/dwz
    
    // Row 1: q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
    data[6] =  0.5 * wx;   // d/dq0
    data[7] =  0.5 * wz;   // d/dq2
    data[8] = -0.5 * wy;   // d/dq3
    data[9] =  0.5 * q0;   // d/dwx
    data[10] = -0.5 * q3;  // d/dwy
    data[11] =  0.5 * q2;  // d/dwz
    
    // Row 2: q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
    data[12] =  0.5 * wy;  // d/dq0
    data[13] = -0.5 * wz;  // d/dq1
    data[14] =  0.5 * wx;  // d/dq3
    data[15] =  0.5 * q3;  // d/dwx
    data[16] =  0.5 * q0;  // d/dwy
    data[17] = -0.5 * q1;  // d/dwz
    
    // Row 3: q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
    data[18] =  0.5 * wz;  // d/dq0
    data[19] =  0.5 * wy;  // d/dq1
    data[20] = -0.5 * wx;  // d/dq2
    data[21] = -0.5 * q2;  // d/dwx
    data[22] =  0.5 * q1;  // d/dwy
    data[23] =  0.5 * q0;  // d/dwz
    
    // Row 4: wx_dot = Ix_inv * (tau_x - wy*wz*(Iz - Iy))
    data[24] = -Ix_inv * (Iz - Iy) * wz;  // d/dwy
    data[25] = -Ix_inv * (Iz - Iy) * wy;  // d/dwz
    
    // Row 5: wy_dot = Iy_inv * (tau_y - wx*wz*(Ix - Iz))
    data[26] = -Iy_inv * (Ix - Iz) * wz;  // d/dwx
    data[27] = -Iy_inv * (Ix - Iz) * wx;  // d/dwz
    
    // Row 6: wz_dot = Iz_inv * (tau_z - wx*wy*(Iy - Ix))
    data[28] = -Iz_inv * (Iy - Ix) * wy;  // d/dwx
    data[29] = -Iz_inv * (Iy - Ix) * wx;  // d/dwy
}

class BatchedSpacecraftIntegrator {
private:
    int n_batch;
    int n_states;
    int nnz_per_matrix;
    bool sparse_jacobian;
    
    // SUNDIALS objects for batched solving
    void* cvode_mem;
    SUNMatrix A_batch;
    SUNLinearSolver LS_batch;
    N_Vector y_batch;
    SUNContext sunctx;
    
    // CUDA handles
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
    // Timing
    PrecisionTimer timer;
    double setup_time;
    double solve_time;
    
    vector<TrajectoryPoint> trajectory_points;
    vector<vector<sunrealtype>> results;
    
public:
    BatchedSpacecraftIntegrator(int batch_size, bool use_sparse_jacobian = false) 
        : n_batch(batch_size), n_states(N_STATES), sparse_jacobian(use_sparse_jacobian), 
          setup_time(0), solve_time(0) {

        if (sparse_jacobian) {
            nnz_per_matrix = NNZ; // 7x7 matrix with 30 non-zero entries
        } else {
            nnz_per_matrix = n_states * n_states; // Full dense matrix
        }
        
        cout << "Initializing batched integrator for " << n_batch << " trajectories..." << endl;
        
        timer.start();
        
        // Initialize SUNDIALS context
        int retval = SUNContext_Create(NULL, &sunctx);
        if (retval != 0) {
            cerr << "Error creating SUNDIALS context" << endl;
            exit(1);
        }
        
        // Initialize CUDA handles
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        // Create batched vector (n_batch * n_states total size)
        y_batch = N_VNew_Cuda(n_batch * n_states, sunctx);
        if (!y_batch) {
            cerr << "Error creating batched CUDA vector" << endl;
            exit(1);
        }        
        setupBatchedLinearSolver();
        
        setup_time = timer.getElapsedMs();
        cout << "Batch setup completed in " << setup_time << " ms" << endl;
    }
    
    ~BatchedSpacecraftIntegrator() {
        if (cvode_mem) CVodeFree(&cvode_mem);
        if (A_batch) SUNMatDestroy(A_batch);
        if (LS_batch) SUNLinSolFree(LS_batch);
        if (y_batch) N_VDestroy(y_batch);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
        if (sunctx) SUNContext_Free(&sunctx);
    }
    
    void setupBatchedLinearSolver() {

        cout << "Using BatchQR solver for " << n_batch << " matrices" << endl;
        
        // Use the proper block CSR constructor for block diagonal matrices
        // This is exactly what BatchQR was designed for!
        A_batch = SUNMatrix_cuSparse_NewBlockCSR(n_batch,        // number of blocks
                                                    n_states,       // rows per block 
                                                    n_states,       // cols per block
                                                    nnz_per_matrix, // NNZ per block
                                                    cusparse_handle, 
                                                    sunctx);
        
        if (!A_batch) {
            cerr << "Error creating batched Block CSR matrix" << endl;
            exit(1);
        }
        
        // Initialize sparse pattern - much simpler now!
        initializeBatchedSparsePattern();
        
        // Create BatchQR linear solver
        LS_batch = SUNLinSol_cuSolverSp_batchQR(y_batch, A_batch, cusolver_handle, sunctx);
        
        if (!LS_batch) {
            cerr << "Error creating BatchQR linear solver" << endl;
            exit(1);
        }
        
        cout << "BatchQR solver created successfully" << endl;
        cout << "Block diagonal matrix: " << n_batch << " blocks of " 
                    << n_states << "x" << n_states << " (" << nnz_per_matrix << " NNZ each)" << endl;
        cout << "Total system size: " << (n_batch * n_states) << " equations" << endl;
    }
    
    void initializeBatchedSparsePattern() {
        // For Block CSR format, we only need to set up the pattern for ONE block
        // SUNDIALS will automatically replicate it across all blocks
        
        sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(A_batch);
        sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(A_batch);
        
        // Sparsity pattern for one 7x7 spacecraft dynamics block
        vector<sunindextype> block_rowptrs = {0, 6, 12, 18, 24, 26, 28, 30};
        vector<sunindextype> block_colvals = {
            // Row 0: columns 1,2,3,4,5,6
            1, 2, 3, 4, 5, 6,
            // Row 1: columns 0,2,3,4,5,6
            0, 2, 3, 4, 5, 6,
            // Row 2: columns 0,1,3,4,5,6
            0, 1, 3, 4, 5, 6,
            // Row 3: columns 0,1,2,4,5,6
            0, 1, 2, 4, 5, 6,
            // Row 4: columns 5,6
            5, 6,
            // Row 5: columns 4,6
            4, 6,
            // Row 6: columns 4,5
            4, 5
        };
        
        // Copy pattern to device (only need one block's worth)
        CUDA_CHECK(cudaMemcpy(d_rowptrs, block_rowptrs.data(), 
                              (n_states + 1) * sizeof(sunindextype), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colvals, block_colvals.data(), 
                              nnz_per_matrix * sizeof(sunindextype), cudaMemcpyHostToDevice));
        
        cout << "Initialized block CSR sparse pattern for " << n_batch 
                  << " identical " << n_states << "x" << n_states << " blocks" << endl;
        cout << "Sparsity per block: " << nnz_per_matrix << "/" << (n_states * n_states) 
                  << " = " << fixed << setprecision(1) 
                  << (100.0 * nnz_per_matrix) / (n_states * n_states) << "%" << endl;
    }
    
    void addTrajectoryPoint(const vector<sunrealtype>& initial_state, sunrealtype dt) {
        if (trajectory_points.size() >= n_batch) {
            cerr << "Cannot add more trajectory points: batch is full" << endl;
            return;
        }
        
        if (initial_state.size() != n_states) {
            cerr << "Initial state must have " << n_states << " elements" << endl;
            return;
        }
        
        trajectory_points.emplace_back(initial_state, dt);
    }

    void printBatchedResults() {
        cout << "\n" << string(80, '=') << endl;
        cout << "BATCHED TRAJECTORY INTEGRATION RESULTS" << endl;
        cout << string(80, '=') << endl;
        
        cout << "Setup time: " << fixed << setprecision(2) << setup_time << " ms" << endl;
        cout << "Solve time: " << solve_time << " ms" << endl;
        cout << "Total time: " << (setup_time + solve_time) << " ms" << endl;
        cout << "Average time per trajectory: " << (solve_time / n_batch) << " ms" << endl;
        
        cout << "\nFirst 5 trajectory results:" << endl;
        for (int i = 0; i < min(5, n_batch); i++) {
            cout << "Trajectory " << i << ": ";
            cout << "q=[" << setprecision(4);
            for (int j = 0; j < 4; j++) {
                cout << results[i][j];
                if (j < 3) cout << ",";
            }
            cout << "] w=[";
            for (int j = 4; j < 7; j++) {
                cout << results[i][j];
                if (j < 6) cout << ",";
            }
            cout << "]" << endl;
        }
        
        // Calculate quaternion norm statistics
        vector<double> norms;
        for (int i = 0; i < n_batch; i++) {
            double norm = sqrt(results[i][0]*results[i][0] + results[i][1]*results[i][1] + 
                              results[i][2]*results[i][2] + results[i][3]*results[i][3]);
            norms.push_back(norm);
        }
        
        double mean_norm = accumulate(norms.begin(), norms.end(), 0.0) / norms.size();
        double max_error = 0.0;
        for (double norm : norms) {
            max_error = max(max_error, abs(norm - 1.0));
        }
        
        cout << "\nQuaternion norm statistics:" << endl;
        cout << "Mean norm: " << setprecision(6) << mean_norm << endl;
        cout << "Max error: " << setprecision(2) << (max_error * 100.0) << "%" << endl;
    }
    
    void solveBatchedTrajectories(bool verbose = false) {
        if (trajectory_points.size() != n_batch) {
            cerr << "Must have exactly " << n_batch << " trajectory points" << endl;
            return;
        }
        
        cout << "\nSolving " << n_batch << " spacecraft trajectories simultaneously..." << endl;
        
        // Copy initial conditions to batched vector
        setBatchedInitialConditions();
        
        timer.start();
        
        // Solve based on solver type
        results.resize(n_batch);
        
        // Initialize batched CVODES
        setupBatchedCVODES();
        solveBatchedSimultaneous(verbose);
        
        solve_time = timer.getElapsedMs();
        
        if (verbose) {
            printBatchedResults();
        }
    }

    static int batchedRhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
        BatchedSpacecraftIntegrator* integrator = static_cast<BatchedSpacecraftIntegrator*>(user_data);
        
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
        // Launch kernel with one block per trajectory
        dim3 blockSize(1);
        dim3 gridSize(integrator->n_batch);
        
        batchedRHS_kernel<<<gridSize, blockSize>>>(integrator->n_batch, integrator->n_states, 
                                                   y_data, ydot_data);
        
        cudaDeviceSynchronize();
        return 0;
    }

    static int batchedJacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                                      SUNMatrix Jac, void* user_data, 
                                      N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        BatchedSpacecraftIntegrator* integrator = static_cast<BatchedSpacecraftIntegrator*>(user_data);
        
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        sunrealtype* jac_data = SUNMatrix_cuSparse_Data(Jac);
        
        // Launch kernel with one block per trajectory
        dim3 blockSize(1);
        dim3 gridSize(integrator->n_batch);

        if (integrator->sparse_jacobian) {
            batchedSparseJacobian_kernel<<<gridSize, blockSize>>>(integrator->n_batch, 
                                                                  integrator->n_states,
                                                                  integrator->nnz_per_matrix,
                                                                  jac_data, y_data);
        } else {
            batchedDenseJacobian_kernel<<<gridSize, blockSize>>>(integrator->n_batch, 
                                                                 integrator->n_states,
                                                                 jac_data, y_data);
        }
        
        cudaDeviceSynchronize();
        return 0;
    }
    
    void setupBatchedCVODES() {
        // Note: For true batched solving, we'd need CVODES to support batched operations
        // For now, we'll demonstrate the approach with the BatchQR solver setup
        
        cvode_mem = CVodeCreate(CV_BDF, sunctx);
        if (!cvode_mem) {
            cerr << "Error creating CVODES memory" << endl;
            exit(1);
        }
        
        // Initialize with the batched vector
        int retval = CVodeInit(cvode_mem, batchedRhsFunction, 0.0, y_batch);
        if (retval != CV_SUCCESS) {
            cerr << "Error initializing batched CVODES: " << retval << endl;
            exit(1);
        }
        
        CVodeSetUserData(cvode_mem, this);
        
        // Set tolerances
        retval = CVodeSStolerances(cvode_mem, 1e-8, 1e-10);
        if (retval != CV_SUCCESS) {
            cerr << "Error setting tolerances: " << retval << endl;
            exit(1);
        }
        
        // Set linear solver
        retval = CVodeSetLinearSolver(cvode_mem, LS_batch, A_batch);
        if (retval != CV_SUCCESS) {
            cerr << "Error setting linear solver: " << retval << endl;
            exit(1);
        }
        
        // Set Jacobian function
        retval = CVodeSetJacFn(cvode_mem, batchedJacobianFunction);
        if (retval != CV_SUCCESS) {
            cerr << "Error setting Jacobian function: " << retval << endl;
            exit(1);
        }
        
        CVodeSetMaxNumSteps(cvode_mem, 50000);
    }
    
    void setBatchedInitialConditions() {
        vector<sunrealtype> y0_batch(n_batch * n_states);
        
        for (int i = 0; i < n_batch; i++) {
            int offset = i * n_states;
            for (int j = 0; j < n_states; j++) {
                y0_batch[offset + j] = trajectory_points[i].initial_state[j];
            }
        }
        
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y_batch);
        CUDA_CHECK(cudaMemcpy(d_y, y0_batch.data(), 
                              y0_batch.size() * sizeof(sunrealtype), cudaMemcpyHostToDevice));
    }
    
    void solveBatchedSimultaneous(bool verbose) {
        // For demonstration: solve to a common time (could be extended for individual times)
        sunrealtype max_dt = 0.0;
        for (const auto& point : trajectory_points) {
            max_dt = max(max_dt, point.dt);
        }
        
        cout << "Solving all trajectories to time t = " << max_dt << endl;
        
        sunrealtype t = 0.0;
        int retval = CVode(cvode_mem, max_dt, y_batch, &t, CV_NORMAL);
        
        if (retval < 0) {
            cerr << "Batched CVode error: " << retval << endl;
            return;
        }
        
        // Extract results
        extractBatchedResults();
    }
    


    void extractBatchedResults() {
        vector<sunrealtype> y_host(n_batch * n_states);
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y_batch);
        
        CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, 
                              y_host.size() * sizeof(sunrealtype), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < n_batch; i++) {
            int offset = i * n_states;
            results[i].resize(n_states);
            for (int j = 0; j < n_states; j++) {
                results[i][j] = y_host[offset + j];
            }
        }
    } 

    const vector<vector<sunrealtype>>& getResults() const { return results; }
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return solve_time; }
};

// Utility function to generate test trajectory
vector<TrajectoryPoint> generateTestTrajectory(int n_points) {
    vector<TrajectoryPoint> trajectory;
    
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<double> quat_dist(-0.1, 0.1);
    uniform_real_distribution<double> omega_dist(-0.05, 0.05);
    uniform_real_distribution<double> dt_dist(0.1, 2.0);
    
    for (int i = 0; i < n_points; i++) {
        // Generate initial quaternion (start from identity + small perturbation)
        vector<sunrealtype> state(7);
        state[0] = 1.0 + quat_dist(gen);  // q0
        state[1] = quat_dist(gen);        // q1
        state[2] = quat_dist(gen);        // q2
        state[3] = quat_dist(gen);        // q3
        
        // Normalize quaternion
        double norm = sqrt(state[0]*state[0] + state[1]*state[1] + 
                          state[2]*state[2] + state[3]*state[3]);
        for (int j = 0; j < 4; j++) {
            state[j] /= norm;
        }
        
        // Generate initial angular velocities
        state[4] = 0.01 + omega_dist(gen);  // wx
        state[5] = 0.02 + omega_dist(gen);  // wy
        state[6] = 0.005 + omega_dist(gen); // wz
        
        sunrealtype dt = dt_dist(gen);
        
        trajectory.emplace_back(state, dt);
    }
    
    return trajectory;
}

void runBatchedIntegrationDemo() {
    cout << "\n" << string(80, '=') << endl;
    cout << "BATCHED SPACECRAFT TRAJECTORY INTEGRATION DEMO" << endl;
    cout << string(80, '=') << endl;
    
    const int n_trajectories = 100;
    
    try {
        cout << "\n" << string(80, '-') << endl;
        
        cout << "Trying BatchQR solver..." << endl;
        
        // Create batched integrator
        BatchedSpacecraftIntegrator integrator(n_trajectories);
        
        // Generate test trajectory
        cout << "Generating " << n_trajectories << " test trajectory points..." << endl;
        auto trajectory = generateTestTrajectory(n_trajectories);
        
        // Add trajectory points to integrator
        for (const auto& point : trajectory) {
            integrator.addTrajectoryPoint(point.initial_state, point.dt);
        }
        
        // Solve batched trajectories
        integrator.solveBatchedTrajectories(true);
        
        cout << "\n" << string(80, '-') << endl;
        cout << "SUCCESS: Batched integration completed with BatchQR solver!"<< endl;

        cout << "Throughput: " << (n_trajectories * 1000.0 / integrator.getSolveTime()) 
                    << " trajectories/second" << endl;
        cout << string(80, '-') << endl;
        
    } catch (const exception& e) {
        cerr << "Failed with exception: " << e.what() << endl;
    }
}

int main() {
    try {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            cerr << "No CUDA devices found!" << endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << endl;
        cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << endl;
        
        runBatchedIntegrationDemo();
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}