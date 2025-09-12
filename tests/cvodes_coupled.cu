#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

// SUNDIALS headers
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>

// Define constants
#define COUPLING_STRENGTH 0.1  // Coupling strength for diffusion term
#define N_STATES 100           // Number of coupled ODEs
#define H_0 2.0                // Final time for integration

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Forward declarations
class CVODESGpuExample;

// CUDA kernel for RHS computation: dy/dt = p * y + coupling
__global__ void RHSKernel(int n, sunrealtype* y, sunrealtype* ydot, sunrealtype p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Exponential term: p * y[i]
    sunrealtype exponential_term = p * y[i];
    
    // Coupling term: c * (y[i+1] - 2*y[i] + y[i-1])
    sunrealtype coupling_term = 0.0;
    sunrealtype c = COUPLING_STRENGTH;
    
    if (i == 0) {
        // Left boundary: y[-1] = y[0] (zero flux)
        coupling_term = c * (y[1] - y[0]);
    } else if (i == n-1) {
        // Right boundary: y[n] = y[n-1] (zero flux)  
        coupling_term = c * (y[n-2] - y[n-1]);
    } else {
        // Interior points: standard diffusion
        coupling_term = c * (y[i+1] - 2.0*y[i] + y[i-1]);
    }
    
    // Combined: dy/dt = p*y + coupling
    ydot[i] = exponential_term + coupling_term;
}

// CUDA kernel for setting up diagonal Jacobian matrix
__global__ void setupTridiagonalJacobianKernel(int n, sunindextype* rowptrs, 
                                              sunindextype* colvals, sunrealtype* data, 
                                              sunrealtype p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    sunrealtype c = COUPLING_STRENGTH;
    int nnz_count = 0;
    
    // Calculate starting position for row i
    if (i == 0) {
        rowptrs[0] = 0;
        nnz_count = 0;
    } else if (i == 1) {
        rowptrs[1] = 2;  // Row 0 has 2 entries
        nnz_count = 2;
    } else if (i == n-1) {
        rowptrs[n-1] = 3*i - 2;  // Pattern for interior rows
        nnz_count = 3*i - 2;
    } else {
        rowptrs[i] = 3*i - 2;
        nnz_count = 3*i - 2;
    }
    
    // Fill the matrix entries for row i
    if (i == 0) {
        // First row: [p-c, c]
        colvals[0] = 0;        data[0] = p - c;
        colvals[1] = 1;        data[1] = c;
    } else if (i == n-1) {
        // Last row: [c, p-c]
        int start = 3*i - 2;
        colvals[start] = i-1;     data[start] = c;
        colvals[start+1] = i;     data[start+1] = p - c;
    } else {
        // Interior rows: [c, p-2c, c]
        int start = 3*i - 2;
        colvals[start] = i-1;     data[start] = c;
        colvals[start+1] = i;     data[start+1] = p - 2.0*c;
        colvals[start+2] = i+1;   data[start+2] = c;
    }
    
    // Last thread sets the final rowptr
    if (i == n-1) {
        rowptrs[n] = 3*n - 2;  // Total nnz for tridiagonal
    }
}

// Helper function to setup exponential Jacobian
void setupTridiagonalJacobian(SUNMatrix Jac, sunrealtype p, int n) {
    // Fill the Jacobian matrix: df/dy = p (diagonal matrix)
    // For dy/dt = p*y, the Jacobian is simply p*I
    
    // Get matrix pointers
    sunindextype* rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* colvals = SUNMatrix_cuSparse_IndexValues(Jac);
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    
    // Set up diagonal matrix pattern on GPU
    setupTridiagonalJacobianKernel<<<(n+255)/256, 256>>>(n, rowptrs, colvals, data, p);
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
        int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
        if (retval != 0) {
            std::cerr << "Error creating SUNDIALS context" << std::endl;
            exit(1);
        }
        
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        // Create state vector
        y = N_VNew_Cuda(n, sunctx);
        if (!y) {
            std::cerr << "Error creating CUDA vector" << std::endl;
            exit(1);
        }
        
        // Create sparse Jacobian matrix structure (tridiagonal for coupled system)
        int nnz = 3*n - 2; // Tridiagonal matrix: 3n-2 non-zeros
        A = SUNMatrix_cuSparse_NewCSR(n, n, nnz, cusparse_handle, sunctx);
        if (!A) {
            std::cerr << "Error creating cuSPARSE matrix" << std::endl;
            exit(1);
        }
        
        // Create your GPU batch QR linear solver
        LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        if (!LS) {
            std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
            exit(1);
        }
        
        initializeCVODES();
    }
    
    ~CVODESGpuExample() {
        std::cout << "Cleaning up CVODES GPU example..." << std::endl;
        
        // Cleanup CVODES
        if (cvode_mem) CVodeFree(&cvode_mem);
        
        // Cleanup SUNDIALS objects
        if (A) SUNMatDestroy(A);
        if (LS) SUNLinSolFree(LS);
        if (y) N_VDestroy(y);
        
        // Cleanup CUDA handles
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
        
        if (sunctx) SUNContext_Free(&sunctx);
    }
    
    void initializeCVODES() {
        // Create CVODES memory
        cvode_mem = CVodeCreate(CV_BDF, sunctx);  // BDF for stiff problems
        if (!cvode_mem) {
            std::cerr << "Error creating CVODES memory" << std::endl;
            exit(1);
        }
        
        // Set initial conditions
        setInitialConditions();
        
        // Initialize CVODES
        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error initializing CVODES: " << retval << std::endl;
            exit(1);
        }
        
        // Set user data so static functions can access parameter_p
        CVodeSetUserData(cvode_mem, this);
        
        // Set tolerances
        retval = CVodeSStolerances(cvode_mem, 1e-6, 1e-8);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting tolerances: " << retval << std::endl;
            exit(1);
        }
        
        // Attach your GPU linear solver to CVODES
        retval = CVodeSetLinearSolver(cvode_mem, LS, A);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting linear solver: " << retval << std::endl;
            exit(1);
        }
        
        // Set Jacobian function (optional - CVODES can approximate)
        retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error setting Jacobian function: " << retval << std::endl;
            exit(1);
        }
        
        std::cout << "CVODES initialized successfully!" << std::endl;
    }
    
    // Your ODE right-hand side function: dy/dt = p*y + coupling
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
        CVODESGpuExample* example = static_cast<CVODESGpuExample*>(user_data);
        sunrealtype p = example->parameter_p;
        int n = N_VGetLength(y);
        
        sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
        // Launch CUDA kernel: ydot[i] = p * y[i] for all i
        RHSKernel<<<(n+255)/256, 256>>>(n, y_data, ydot_data, p);
        cudaDeviceSynchronize();
        
        return 0;
    }
    
    // Jacobian function: For coupled system, Jacobian is tridiagonal
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        CVODESGpuExample* example = static_cast<CVODESGpuExample*>(user_data);
        sunrealtype p = example->parameter_p;
        
        // For coupled system, the Jacobian is tridiagonal
        int n = SUNMatrix_cuSparse_Rows(Jac);
        setupTridiagonalJacobian(Jac, p, n);
        
        return 0;
    }
    
    void solveODE(sunrealtype t_final) {
        std::cout << "Solving ODE system with GPU acceleration..." << std::endl;
        
        sunrealtype t = 0.0;
        sunrealtype dt_out = t_final / 100.0;  // Output every 1% of simulation
        
        // Time integration loop
        for (int step = 0; step < 100; step++) {
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
        
    }
    
    void runConvergenceTest(sunrealtype t_final) {
        std::cout << "\n=== Convergence Test ===" << std::endl;
        std::vector<sunrealtype> tolerances = {1e-4, 1e-6, 1e-8, 1e-10};
        std::vector<sunrealtype> final_masses;
        
        for (auto tol : tolerances) {
            // Reinitialize CVODES with new tolerance
            CVodeFree(&cvode_mem);
            cvode_mem = CVodeCreate(CV_BDF, sunctx);
            
            setInitialConditions();
            CVodeInit(cvode_mem, rhsFunction, 0.0, y);
            CVodeSetUserData(cvode_mem, this);
            CVodeSStolerances(cvode_mem, tol/100, tol);
            CVodeSetLinearSolver(cvode_mem, LS, A);
            CVodeSetJacFn(cvode_mem, jacobianFunction);
            
            // Solve to final time
            sunrealtype t = 0.0;
            CVode(cvode_mem, t_final, y, &t, CV_NORMAL);
            
            // Get final mass
            std::vector<sunrealtype> y_host(n);
            sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
            cudaMemcpy(y_host.data(), d_y, n * sizeof(sunrealtype), cudaMemcpyDeviceToHost);
            
            sunrealtype final_mass = 0.0;
            for (int i = 0; i < n; i++) {
                final_mass += y_host[i];
            }
            final_masses.push_back(final_mass);
            
            // Get solver stats
            long int nsteps, nfevals, njevals;
            CVodeGetNumSteps(cvode_mem, &nsteps);
            CVodeGetNumRhsEvals(cvode_mem, &nfevals);
            CVodeGetNumJacEvals(cvode_mem, &njevals);
            
            std::printf("Tol=%.0e: Mass=%.8e, Steps=%ld, RHS=%ld, Jac=%ld\n", 
                       tol, final_mass, nsteps, nfevals, njevals);
        }
        
        // Check convergence
        std::cout << "\nConvergence Analysis:" << std::endl;
        for (size_t i = 1; i < final_masses.size(); i++) {
            sunrealtype diff = std::abs(final_masses[i] - final_masses[i-1]);
            sunrealtype rel_diff = diff / std::abs(final_masses[i]);
            std::printf("Tol %.0e→%.0e: Mass diff=%.2e, Rel diff=%.2e\n", 
                       tolerances[i-1], tolerances[i], diff, rel_diff);
        }
        
        // Restore original tolerance
        CVodeFree(&cvode_mem);
        initializeCVODES();
    }
    
private:
    void setInitialConditions() {
        // Set up initial state: y(0) = [1, 2, 3, ..., n] for testing
        std::vector<sunrealtype> y0(n);
        for (int i = 0; i < n; i++) {
            y0[i] = static_cast<sunrealtype>(i + 1);
        }
        
        sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
        CUDA_CHECK(cudaMemcpy(d_y, y0.data(), n * sizeof(sunrealtype), cudaMemcpyHostToDevice));
        
        std::cout << "Initial conditions set: y(0) = [1, 2, 3, ..., " << n << "]" << std::endl;
        std::cout << "Parameter p = " << parameter_p << std::endl;
    }
};

int main() {
    try {
        // Check CUDA availability
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }
        
        std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
        
        // Print device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using device: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        
        std::cout << "CVODES + GPU Batch QR Solver Example" << std::endl;
        std::cout << "System: dy/dt = p*y + c*(diffusion coupling)\n" << std::endl;
        
        // Test different parameter values
        std::vector<sunrealtype> test_params = {-0.1, 0.5, -1.0};
        
        for (sunrealtype p : test_params) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "Testing with parameter p = " << p << std::endl;
            std::cout << std::string(50, '=') << std::endl;
            
            CVODESGpuExample solver(N_STATES, p);  // 100 ODEs, parameter p
            solver.solveODE(H_0);  // Integrate to H_0=2
            solver.runConvergenceTest(H_0);  // Run convergence test
            
            std::cout << "\nExpected behavior for p = " << p << ":" << std::endl;
            if (p > 0) {
                std::cout << "  Exponential growth with diffusion smoothing" << std::endl;
                std::cout << "  Coupling will reduce differences between neighbors" << std::endl;
            } else {
                std::cout << "  Exponential decay with diffusion smoothing" << std::endl;
                std::cout << "  Coupling will reduce differences between neighbors" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}