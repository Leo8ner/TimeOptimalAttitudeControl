#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Sundials headers
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_diag.h> // For CVDiag
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

// CUDA headers
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

// Global handles
cusparseHandle_t cusparse_handle;
cusolverSpHandle_t cusolver_handle;

// Problem size
#define NEQ 10

// Time parameters
#define T0  0.0
#define T1  1.0

// Tolerances - very relaxed
#define RTOL 1e-3
#define ATOL 1e-3

// Integration control
#define MAX_NUM_STEPS 10000
#define INIT_STEP_SIZE 1e-3

// Simple problem: dy/dt = lambda * y
#define LAMBDA -0.1

// Initial condition
#define Y0 1.0

// CUDA block size
#define CUDA_BLOCK_SIZE 256

// Global debug counters
static int jac_eval_count = 0;
static int rhs_eval_count = 0;

// Simple RHS: dy/dt = lambda * y
__global__ void rhs_kernel(sunrealtype t, sunrealtype* y, sunrealtype* ydot, int n, sunrealtype lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ydot[idx] = lambda * y[idx];
    }
}

// Jacobian: df/dy = lambda (not scaled by gamma yet)
__global__ void jac_kernel(sunrealtype* jac_data, int n, sunrealtype lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Just store df/dy, not gamma*df/dy - I
        jac_data[idx] = lambda;
    }
}

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    rhs_eval_count++;
    
    sunrealtype *y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype *ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
    int n = N_VGetLength_Cuda(y);
    
    int numBlocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    rhs_kernel<<<numBlocks, CUDA_BLOCK_SIZE>>>(t, y_data, ydot_data, n, LAMBDA);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RHS kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    if (rhs_eval_count <= 3) {
        printf("  RHS eval %d at t=%.3e\n", rhs_eval_count, t);
    }
    
    return 0;
}

// Option 1: Jacobian function that returns df/dy (not the Newton matrix)
static int Jac_dfdy(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, 
                    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    jac_eval_count++;
    
    sunrealtype *jac_data = SUNMatrix_cuSparse_Data(J);
    int n = SUNMatrix_cuSparse_Rows(J);
    
    int numBlocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    jac_kernel<<<numBlocks, CUDA_BLOCK_SIZE>>>(jac_data, n, LAMBDA);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Jacobian kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    if (jac_eval_count <= 3) {
        printf("  JAC eval %d at t=%.3e (returning df/dy)\n", jac_eval_count, t);
    }
    
    return 0;
}

// Option 2: Linear system function that computes M = gamma*J - I
static int LinSys(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix M,
                  sunbooleantype jok, sunbooleantype *jcur, sunrealtype gamma,
                  void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    
    static int linsys_count = 0;
    linsys_count++;
    
    sunrealtype *m_data = SUNMatrix_cuSparse_Data(M);
    int n = SUNMatrix_cuSparse_Rows(M);
    
    printf("  LinSys eval %d: gamma=%.3e, jok=%d\n", linsys_count, gamma, jok);
    
    // For diagonal matrix, compute M = gamma * lambda - 1
    for (int i = 0; i < n; i++) {
        m_data[i] = gamma * LAMBDA - 1.0;
    }
    
    *jcur = SUNTRUE; // We always recompute
    
    return 0;
}

// Try different solver configurations
int test_configuration(const char* config_name, SUNContext sunctx) {
    printf("\n=== Testing Configuration: %s ===\n", config_name);
    
    // Reset counters
    jac_eval_count = 0;
    rhs_eval_count = 0;
    
    // Create vectors
    N_Vector y = N_VNew_Cuda(NEQ, sunctx);
    N_VConst(Y0, y);
    
    // Create diagonal matrix
    SUNMatrix A = SUNMatrix_cuSparse_NewCSR(NEQ, NEQ, NEQ, cusparse_handle, sunctx);
    
    // Setup matrix structure (diagonal)
    int *h_row_ptrs = (int*)malloc((NEQ+1) * sizeof(int));
    int *h_col_indices = (int*)malloc(NEQ * sizeof(int));
    sunrealtype *h_data = (sunrealtype*)malloc(NEQ * sizeof(sunrealtype));
    
    for (int i = 0; i < NEQ; i++) {
        h_row_ptrs[i] = i;
        h_col_indices[i] = i;
        h_data[i] = LAMBDA;
    }
    h_row_ptrs[NEQ] = NEQ;
    
    // Copy to device
    cudaMemcpy(SUNMatrix_cuSparse_IndexPointers(A), h_row_ptrs, 
               (NEQ+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(SUNMatrix_cuSparse_IndexValues(A), h_col_indices, 
               NEQ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(SUNMatrix_cuSparse_Data(A), h_data, 
               NEQ * sizeof(sunrealtype), cudaMemcpyHostToDevice);
    
    free(h_row_ptrs);
    free(h_col_indices);
    free(h_data);
    
    // Create CVODES
    void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
    
    // Initialize
    int flag = CVodeInit(cvode_mem, f, T0, y);
    
    // Set tolerances
    flag = CVodeSStolerances(cvode_mem, RTOL, ATOL);
    
    // Configuration-specific setup
    int success = 0;
    
    if (strcmp(config_name, "CVDiag") == 0) {
        // Use the diagonal linear solver
        flag = CVDiag(cvode_mem);
        printf("CVDiag: %d\n", flag);
        
    } else if (strcmp(config_name, "SPGMR") == 0) {
        // Use SPGMR (matrix-free)
        SUNLinearSolver LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
        printf("CVodeSetLinearSolver (SPGMR): %d\n", flag);
        
    } else if (strcmp(config_name, "cuSolverSp_batchQR with df/dy") == 0) {
        // Use cuSolverSp_batchQR with Jacobian function
        SUNLinearSolver LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, A);
        printf("CVodeSetLinearSolver: %d\n", flag);
        
        flag = CVodeSetJacFn(cvode_mem, Jac_dfdy);
        printf("CVodeSetJacFn: %d\n", flag);
        
    } else if (strcmp(config_name, "cuSolverSp_batchQR with LinSys") == 0) {
        // Use cuSolverSp_batchQR with linear system function
        SUNLinearSolver LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, A);
        printf("CVodeSetLinearSolver: %d\n", flag);
        
        flag = CVodeSetLinSysFn(cvode_mem, LinSys);
        printf("CVodeSetLinSysFn: %d\n", flag);
        
    } else if (strcmp(config_name, "cuSolverSp_batchQR with modified settings") == 0) {
        // Use cuSolverSp_batchQR with aggressive settings
        SUNLinearSolver LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        flag = CVodeSetLinearSolver(cvode_mem, LS, A);
        
        // Force frequent linear solver setups
        flag = CVodeSetLSetupFrequency(cvode_mem, 1); // Setup every step
        flag = CVodeSetJacEvalFrequency(cvode_mem, 1); // Update Jacobian every step
        flag = CVodeSetDeltaGammaMaxLSetup(cvode_mem, 0.0); // Any gamma change triggers setup
        
        // Use more Newton iterations
        flag = CVodeSetMaxNonlinIters(cvode_mem, 10);
        flag = CVodeSetNonlinConvCoef(cvode_mem, 0.01); // Tighter convergence
        
        // Disable linear solution scaling for BDF
        flag = CVodeSetLinearSolutionScaling(cvode_mem, SUNFALSE);
        
        flag = CVodeSetLinSysFn(cvode_mem, LinSys);
        printf("Aggressive settings applied\n");
        
    } else if (strcmp(config_name, "Fixed-point solver") == 0) {
        // Try fixed-point solver instead of Newton
        SUNNonlinearSolver NLS = SUNNonlinSol_FixedPoint(y, 0, sunctx);
        flag = CVodeSetNonlinearSolver(cvode_mem, NLS);
        printf("Fixed-point solver: %d\n", flag);
    }
    
    // Common settings
    flag = CVodeSetInitStep(cvode_mem, INIT_STEP_SIZE);
    flag = CVodeSetMaxNumSteps(cvode_mem, MAX_NUM_STEPS);
    
    // Try integration
    sunrealtype t = T0;
    sunrealtype tout = 0.1;
    
    printf("Attempting integration to t = %.3f\n", tout);
    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    
    printf("Integration result: %d\n", flag);
    if (flag >= 0) {
        N_VCopyFromDevice_Cuda(y);
        sunrealtype *y_host = N_VGetHostArrayPointer_Cuda(y);
        sunrealtype y_exact = Y0 * exp(LAMBDA * t);
        printf("At t=%.3f: y[0]=%.6f (exact=%.6f)\n", t, y_host[0], y_exact);
        success = 1;
    }
    
    // Get statistics
    long int nst, nfe, nni, ncfn, nsetups;
    CVodeGetNumSteps(cvode_mem, &nst);
    CVodeGetNumRhsEvals(cvode_mem, &nfe);
    CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    
    printf("Statistics: steps=%ld, rhs_evals=%ld, nonlin_iters=%ld, "
           "nonlin_fails=%ld, lin_setups=%ld\n", 
           nst, nfe, nni, ncfn, nsetups);
    printf("Our counts: rhs_evals=%d, jac_evals=%d\n", 
           rhs_eval_count, jac_eval_count);
    
    // Cleanup
    CVodeFree(&cvode_mem);
    SUNMatDestroy(A);
    N_VDestroy(y);
    
    return success;
}

int main() {
    printf("=== CVODES cuSolverSp_batchQR Solutions Test ===\n");
    printf("Problem: dy/dt = %.2f*y, y(0) = %.1f\n", LAMBDA, Y0);
    printf("Expected solution: y(t) = %.1f * exp(%.2f*t)\n\n", Y0, LAMBDA);
    
    // Initialize CUDA
    cudaError_t cuda_err = cudaSetDevice(0);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA init failed: %s\n", cudaGetErrorString(cuda_err));
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // Check CUDA/cuSOLVER version
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);
    printf("CUDA Runtime version: %d.%d\n", cuda_version/1000, (cuda_version%1000)/10);
    
    // Initialize libraries
    cusparseCreate(&cusparse_handle);
    cusolverSpCreate(&cusolver_handle);
    
    // Get cuSOLVER version
    int cusolver_version;
    cusolverGetVersion(&cusolver_version);
    printf("cuSOLVER version: %d\n\n", cusolver_version);
    
    // Create context
    SUNContext sunctx;
    SUNContext_Create(SUN_COMM_NULL, &sunctx);
    
    // Test different configurations
    const char* configs[] = {
        "CVDiag",
        "SPGMR",
        "Fixed-point solver",
        "cuSolverSp_batchQR with df/dy",
        "cuSolverSp_batchQR with LinSys",
        "cuSolverSp_batchQR with modified settings"
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    int successes = 0;
    
    for (int i = 0; i < num_configs; i++) {
        successes += test_configuration(configs[i], sunctx);
    }
    
    printf("\n=== SUMMARY ===\n");
    printf("Successful configurations: %d/%d\n", successes, num_configs);
    
    // Based on the documentation, suggest potential issues
    printf("\nBased on CVODES documentation analysis:\n");
    printf("1. cuSolverSp_batchQR might require specific matrix/system properties\n");
    printf("2. The solver might not support the Newton system format M = gamma*J - I\n");
    printf("3. There might be a version incompatibility between SUNDIALS and cuSOLVER\n");
    printf("4. Consider using CVodeSetLinearSolutionScaling(cvode_mem, SUNFALSE)\n");
    printf("5. The documentation mentions compatibility requirements in Section 9\n");
    
    // Cleanup
    cusparseDestroy(cusparse_handle);
    cusolverSpDestroy(cusolver_handle);
    SUNContext_Free(&sunctx);
    
    return 0;
}