#include <cvodes/cvodes.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <iostream>
#include <vector>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

#define N_ODE 100  // Number of ODEs in the system
#define CVODE_ACCEPTABLE_ERR 1e-8  // Acceptable error tolerance for CVODES
#define CVODE_ACCEPTABLE_REL_ERR 1e-6  // Relative error tolerance for CVODES
#define H_0 200  // Final time for integration
#define N_OUT 100  // Number of output steps

// CUDA kernel for RHS computation: dy/dt = p * y
__global__ void exponentialRHSKernel(int n, sunrealtype* y, sunrealtype* ydot, sunrealtype p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Simple exponential: ydot[i] = p * y[i]
    ydot[i] = p * y[i];
}

__global__ void setupDiagonalJacobianKernel(int n, sunindextype* rowptrs, 
                                           sunindextype* colvals, sunrealtype* data, 
                                           sunrealtype p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Diagonal matrix: only one entry per row
    rowptrs[i] = i;        // Row i starts at position i
    colvals[i] = i;        // Column index is i (diagonal)
    data[i] = p;           // Jacobian value is p
    
    // Last thread sets the final rowptr
    if (i == n-1) {
        rowptrs[n] = n;    // Total nnz = n
    }
}

void setupExponentialJacobian(SUNMatrix Jac, sunrealtype p, int n) {
    // Fill the Jacobian matrix: df/dy = p (diagonal matrix)
    // For dy/dt = p*y, the Jacobian is simply p*I
    
    // Get matrix pointers
    sunindextype* rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* colvals = SUNMatrix_cuSparse_IndexValues(Jac);
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    
    // Set up diagonal matrix pattern on GPU
    int blockSize;
    
    if (n < 512) {
        blockSize = 64;   // Small problems
    } else if (n < 10000) {
        blockSize = 256;  // Medium problems  
    } else {
        blockSize = 512;  // Large problems
    }
    
    int gridSize = (n + blockSize - 1) / blockSize;
    setupDiagonalJacobianKernel<<<gridSize, blockSize>>>(n, rowptrs, colvals, data, p);
    cudaDeviceSynchronize();
}



class CVODESGpuExample {
private:
    void* cvode_mem;
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector y;
    SUNContext sunctx;
    int n;
    sunrealtype parameter_p;  // The parameter in dy/dt = p*y
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
public:
    CVODESGpuExample(int system_size, sunrealtype p) : n(system_size), parameter_p(p) {
        // Initialize CUDA handles and SUNDIALS context
        SUNContext_Create(SUN_COMM_NULL, &sunctx);
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        // Create state vector
        y = N_VNew_Cuda(n, sunctx);
        
        // Create sparse Jacobian matrix structure (diagonal for dy/dt = p*y)
        int nnz = n; // Only diagonal entries for this system
        A = SUNMatrix_cuSparse_NewCSR(n, n, nnz, cusparse_handle, sunctx);
        
        // Create your GPU batch QR linear solver
        LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        
        initializeCVODES();
    }
    
    void initializeCVODES() {
        // Create CVODES memory
        cvode_mem = CVodeCreate(CV_BDF, sunctx);  // BDF for stiff problems
        
        // Set initial conditions
        setInitialConditions();
        
        // Initialize CVODES
        CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        
        // Set user data so static functions can access parameter_p
        CVodeSetUserData(cvode_mem, this);
        
        // Set tolerances
        CVodeSStolerances(cvode_mem, CVODE_ACCEPTABLE_REL_ERR, CVODE_ACCEPTABLE_ERR);
        
        // Attach your GPU linear solver to CVODES
        CVodeSetLinearSolver(cvode_mem, LS, A);
        
        // Set Jacobian function (optional - CVODES can approximate)
        CVodeSetJacFn(cvode_mem, jacobianFunction);
        
        // Enable sensitivity analysis if needed
        // CVodeSensInit(cvode_mem, ...);
    }
    
    // Your ODE right-hand side function: dy/dt = p * y
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
        CVODESGpuExample* example = static_cast<CVODESGpuExample*>(user_data);
        sunrealtype p = example->parameter_p;
        int n = N_VGetLength(y);
        
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
        // Launch CUDA kernel: ydot[i] = p * y[i] for all i
        int blockSize;
        
        if (n < 512) {
            blockSize = 64;   // Small problems
        } else if (n < 10000) {
            blockSize = 256;  // Medium problems  
        } else {
            blockSize = 512;  // Large problems
        }
        
        int gridSize = (n + blockSize - 1) / blockSize;
        exponentialRHSKernel<<<gridSize, blockSize>>>(n, y_data, ydot_data, p);
        cudaDeviceSynchronize();
        
        return 0;
    }
    
    // Jacobian function: For dy/dt = p*y, Jacobian = df/dy = p (diagonal matrix)
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        CVODESGpuExample* example = static_cast<CVODESGpuExample*>(user_data);
        sunrealtype p = example->parameter_p;
        
        // For dy/dt = p*y, the Jacobian is a diagonal matrix with p on diagonal
        int n = SUNMatrix_cuSparse_Rows(Jac);
        setupExponentialJacobian(Jac, p, n);
        
        return 0;
    }
    
    void solveODE(sunrealtype t_final) {
        std::cout << "Solving ODE system with GPU acceleration..." << std::endl;
        
        sunrealtype t = 0.0;
        sunrealtype dt_out = t_final / N_OUT;  // Output every 1% of simulation
        
        // Time integration loop
        for (int step = 0; step < N_OUT; step++) {
            sunrealtype t_out = (step + 1) * dt_out;
            
            // CVODES advances the solution using your GPU linear solver
            int retval = CVode(cvode_mem, t_out, y, &t, CV_NORMAL);
            
            if (retval < 0) {
                std::cerr << "CVode error: " << retval << std::endl;
                break;
            }
            
            // Print solution statistics
            if (step % 20 == 0) {
                printSolutionStats(t, step);
            }
        }
    }
    
    void printSolutionStats(sunrealtype t, int step) {
        // Get CVODES statistics
        long int nsteps, nfevals, nlinsetups, netfails;
        CVodeGetNumSteps(cvode_mem, &nsteps);
        CVodeGetNumRhsEvals(cvode_mem, &nfevals);
        CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
        CVodeGetNumErrTestFails(cvode_mem, &netfails);
        
        // Get linear solver statistics
        long int njevals, nliters, nlcfails;
        CVodeGetNumJacEvals(cvode_mem, &njevals);
        CVodeGetNumLinIters(cvode_mem, &nliters);
        CVodeGetNumLinConvFails(cvode_mem, &nlcfails);
        
        std::printf("Step %d, t=%.3f: nsteps=%ld, RHS=%ld, Jac=%ld, LinSetup=%ld\n",
                   step, t, nsteps, nfevals, njevals, nlinsetups);
        
        // Compare with analytical solution
        compareWithAnalytical(t);
    }
    
    void compareWithAnalytical(sunrealtype t) {
        // Analytical solution: y(t) = y(0) * exp(p*t)
        sunrealtype analytical_factor = std::exp(parameter_p * t);
        
        // Get numerical solution from GPU
        std::vector<sunrealtype> numerical(3);  // First 3 components
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        cudaMemcpy(numerical.data(), d_y, 3 * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
        
        std::printf("  Comparison (first 3 components):\n");
        for (int i = 0; i < 3; i++) {
            sunrealtype analytical = (i + 1) * analytical_factor;  // y0[i] = i+1
            sunrealtype error = std::abs(numerical[i] - analytical);
            std::printf("    y[%d]: numerical=%.6e, analytical=%.6e, error=%.2e\n", 
                       i, numerical[i], analytical, error);
        }
        
        // Print solution norm
        sunrealtype norm = N_VMaxNorm(y);
        std::printf("  Solution max norm: %.6e\n", norm);
    }
    
    void demonstrateAdvantages() {
        std::cout << "\n=== CVODES + GPU Batch QR Advantages ===" << std::endl;
        std::cout << "✓ Adaptive time-stepping (automatically adjusts dt)" << std::endl;
        std::cout << "✓ Error control (maintains accuracy)" << std::endl;
        std::cout << "✓ GPU-accelerated Jacobian solves" << std::endl;
        std::cout << "✓ Batch processing reduces GPU kernel overhead" << std::endl;
        std::cout << "✓ Sensitivity analysis capabilities" << std::endl;
        std::cout << "✓ Root finding during integration" << std::endl;
        std::cout << "✓ Handles stiff systems efficiently" << std::endl;
    }
    
private:
    void setInitialConditions() {
        // Set up initial state: y(0) = [1, 2, 3, ..., n] for testing
        std::vector<sunrealtype> y0(n);
        for (int i = 0; i < n; i++) {
            y0[i] = static_cast<sunrealtype>(i + 1);
        }
        
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        cudaMemcpy(d_y, y0.data(), n * sizeof(sunrealtype), cudaMemcpyHostToDevice);
        
        std::cout << "Initial conditions set: y(0) = [1, 2, 3, ..., " << n << "]" << std::endl;
        std::cout << "Parameter p = " << parameter_p << std::endl;
        std::cout << "Analytical solution: y(t) = y(0) * exp(p*t)" << std::endl;
    }
};



int main() {
    try {
        std::cout << "CVODES + GPU Batch QR Solver Example" << std::endl;
        std::cout << "System: dy/dt = p * y (exponential growth/decay)\n" << std::endl;
        int blockSize;
    
        if (N_ODE < 512) {
            blockSize = 64;   // Small problems
        } else if (N_ODE < 10000) {
            blockSize = 256;  // Medium problems  
        } else {
            blockSize = 512;  // Large problems
        }
        
        int gridSize = (N_ODE + blockSize - 1) / blockSize;
        std::printf("Using block size %d, grid size %d for %d ODEs\n", 
               blockSize, gridSize, N_ODE);

        
        // Test different parameter values
        std::vector<sunrealtype> test_params = {-0.1, 0.5, -1.0};
        
        for (sunrealtype p : test_params) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "Testing with parameter p = " << p << std::endl;
            std::cout << std::string(50, '=') << std::endl;
            
            CVODESGpuExample solver(N_ODE, p);  // 100 ODEs, parameter p
            solver.demonstrateAdvantages();
            solver.solveODE(H_0);  // Integrate to t=2
            
            std::cout << "\nExpected behavior for p = " << p << ":" << std::endl;
            if (p > 0) {
                std::cout << "  Exponential growth: y(t) grows like exp(" << p << "*t)" << std::endl;
            } else {
                std::cout << "  Exponential decay: y(t) decays like exp(" << p << "*t)" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}