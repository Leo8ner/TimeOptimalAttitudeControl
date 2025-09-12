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

// SUNDIALS headers
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>

// Define constants
#define COUPLING_STRENGTH 0.1  
#define DIFUSION_COEFFICIENT 0.01
#define N_STATES 100           
#define H_0 2.0                
#define NUM_TRIALS 5  // Number of trials for averaging

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
        return duration.count() / 1000.0;  // Return milliseconds with microsecond precision
    }
};

// Double precision implementation
namespace DoublePrecision {
    
    __global__ void RHSKernel(int n, double* y, double* ydot, double p) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        
        double exponential_term = p * y[i];
        double coupling_term = 0.0;
        double c = COUPLING_STRENGTH;
        
        if (i == 0) {
            coupling_term = c * (y[1] - y[0]);
        } else if (i == n-1) {
            coupling_term = c * (y[n-2] - y[n-1]);
        } else {
            coupling_term = c * (y[i+1] - 2.0*y[i] + y[i-1]);
        }
        
        ydot[i] = exponential_term + coupling_term;
    }
    
    __global__ void JacobianKernel(int n, sunindextype* rowptrs, 
                                  sunindextype* colvals, double* data, 
                                  double p) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        
        double c = COUPLING_STRENGTH;
        
        if (i == 0) {
            rowptrs[0] = 0;
        } else if (i == 1) {
            rowptrs[1] = 2;
        } else if (i == n-1) {
            rowptrs[n-1] = 3*i - 2;
        } else {
            rowptrs[i] = 3*i - 2;
        }
        
        if (i == 0) {
            colvals[0] = 0;        data[0] = p - c;
            colvals[1] = 1;        data[1] = c;
        } else if (i == n-1) {
            int start = 3*i - 2;
            colvals[start] = i-1;     data[start] = c;
            colvals[start+1] = i;     data[start+1] = p - c;
        } else {
            int start = 3*i - 2;
            colvals[start] = i-1;     data[start] = c;
            colvals[start+1] = i;     data[start+1] = p - 2.0*c;
            colvals[start+2] = i+1;   data[start+2] = c;
        }
        
        if (i == n-1) {
            rowptrs[n] = 3*n - 2;
        }
    }
}

// Single precision implementation
namespace SinglePrecision {
    
    __global__ void RHSKernel(int n, float* y, float* ydot, float p) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        
        float exponential_term = p * y[i];
        float coupling_term = 0.0f;
        float c = (float)COUPLING_STRENGTH;
        
        if (i == 0) {
            coupling_term = c * (y[1] - y[0]);
        } else if (i == n-1) {
            coupling_term = c * (y[n-2] - y[n-1]);
        } else {
            coupling_term = c * (y[i+1] - 2.0f*y[i] + y[i-1]);
        }
        
        ydot[i] = exponential_term + coupling_term;
    }
    
    __global__ void JacobianKernel(int n, sunindextype* rowptrs, 
                                  sunindextype* colvals, float* data, 
                                  float p) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        
        float c = (float)COUPLING_STRENGTH;
        
        if (i == 0) {
            rowptrs[0] = 0;
        } else if (i == 1) {
            rowptrs[1] = 2;
        } else if (i == n-1) {
            rowptrs[n-1] = 3*i - 2;
        } else {
            rowptrs[i] = 3*i - 2;
        }
        
        if (i == 0) {
            colvals[0] = 0;        data[0] = p - c;
            colvals[1] = 1;        data[1] = c;
        } else if (i == n-1) {
            int start = 3*i - 2;
            colvals[start] = i-1;     data[start] = c;
            colvals[start+1] = i;     data[start+1] = p - c;
        } else {
            int start = 3*i - 2;
            colvals[start] = i-1;     data[start] = c;
            colvals[start+1] = i;     data[start+1] = p - 2.0f*c;
            colvals[start+2] = i+1;   data[start+2] = c;
        }
        
        if (i == n-1) {
            rowptrs[n] = 3*n - 2;
        }
    }
}

// Template class for both precisions
template<typename RealType>
class CVODESGpuSolver {
private:
    void* cvode_mem;
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector y;
    SUNContext sunctx;
    int n;
    RealType parameter_p;
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;
    
    // Timing members
    PrecisionTimer timer;
    double total_solve_time;
    double setup_time;
    
public:
    CVODESGpuSolver(int system_size, RealType p) : n(system_size), parameter_p(p), total_solve_time(0), setup_time(0) {
        timer.start();
        
        int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
        if (retval != 0) {
            std::cerr << "Error creating SUNDIALS context" << std::endl;
            exit(1);
        }
        
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        y = N_VNew_Cuda(n, sunctx);
        if (!y) {
            std::cerr << "Error creating CUDA vector" << std::endl;
            exit(1);
        }
        
        int nnz = 3*n - 2;
        A = SUNMatrix_cuSparse_NewCSR(n, n, nnz, cusparse_handle, sunctx);
        if (!A) {
            std::cerr << "Error creating cuSPARSE matrix" << std::endl;
            exit(1);
        }
        
        LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusolver_handle, sunctx);
        if (!LS) {
            std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
            exit(1);
        }
        
        initializeCVODES();
        setup_time = timer.getElapsedMs();
    }
    
    ~CVODESGpuSolver() {
        if (cvode_mem) CVodeFree(&cvode_mem);
        if (A) SUNMatDestroy(A);
        if (LS) SUNLinSolFree(LS);
        if (y) N_VDestroy(y);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
        if (sunctx) SUNContext_Free(&sunctx);
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
        
        // Use appropriate precision for tolerances
        if constexpr (std::is_same_v<RealType, float>) {
            retval = CVodeSStolerances(cvode_mem, 1e-12, 1e-14);  // Slightly relaxed for single precision
        } else {
            retval = CVodeSStolerances(cvode_mem, 1e-12, 1e-14);  // Standard double precision
        }
        
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
    }
    
    static int rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
        CVODESGpuSolver<RealType>* solver = static_cast<CVODESGpuSolver<RealType>*>(user_data);
        RealType p = solver->parameter_p;
        int n = N_VGetLength(y);
        
        // Cast to appropriate precision
        RealType* y_data = reinterpret_cast<RealType*>(N_VGetDeviceArrayPointer_Cuda(y));
        RealType* ydot_data = reinterpret_cast<RealType*>(N_VGetDeviceArrayPointer_Cuda(ydot));
        
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        
        if constexpr (std::is_same_v<RealType, float>) {
            SinglePrecision::RHSKernel<<<gridSize, blockSize>>>(n, y_data, ydot_data, p);
        } else {
            DoublePrecision::RHSKernel<<<gridSize, blockSize>>>(n, y_data, ydot_data, p);
        }
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "RHS kernel error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        cudaDeviceSynchronize();
        return 0;
    }
    
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        CVODESGpuSolver<RealType>* solver = static_cast<CVODESGpuSolver<RealType>*>(user_data);
        RealType p = solver->parameter_p;
        int n = SUNMatrix_cuSparse_Rows(Jac);
        
        sunindextype* rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
        sunindextype* colvals = SUNMatrix_cuSparse_IndexValues(Jac);
        RealType* data = reinterpret_cast<RealType*>(SUNMatrix_cuSparse_Data(Jac));
        
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        
        if constexpr (std::is_same_v<RealType, float>) {
            SinglePrecision::JacobianKernel<<<gridSize, blockSize>>>(n, rowptrs, colvals, data, p);
        } else {
            DoublePrecision::JacobianKernel<<<gridSize, blockSize>>>(n, rowptrs, colvals, data, p);
        }
        
        cudaDeviceSynchronize();
        return 0;
    }
    
    double solveODE(sunrealtype t_final, bool verbose = false) {
        if (verbose) {
            std::cout << "Solving with " << (std::is_same_v<RealType, float> ? "single" : "double") 
                      << " precision..." << std::endl;
        }
        
        timer.start();
        
        sunrealtype t = 0.0;
        sunrealtype dt_out = t_final / 100.0;
        
        for (int step = 0; step < 100; step++) {
            sunrealtype t_out = (step + 1) * dt_out;
            
            int retval = CVode(cvode_mem, t_out, y, &t, CV_NORMAL);
            
            if (retval < 0) {
                std::cerr << "CVode error: " << retval << std::endl;
                return -1;
            }
        }
        
        total_solve_time = timer.getElapsedMs();
        
        if (verbose) {
            printSolutionStats();
        }
        
        return total_solve_time;
    }
    
    void printSolutionStats() {
        // Get CVODES statistics
        long int nsteps, nfevals, nlinsetups, netfails;
        CVodeGetNumSteps(cvode_mem, &nsteps);
        CVodeGetNumRhsEvals(cvode_mem, &nfevals);
        CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
        CVodeGetNumErrTestFails(cvode_mem, &netfails);
        
        long int njevals, nliters, nlcfails;
        CVodeGetNumJacEvals(cvode_mem, &njevals);
        CVodeGetNumLinIters(cvode_mem, &nliters);
        CVodeGetNumLinConvFails(cvode_mem, &nlcfails);
        
        std::cout << "  Steps: " << nsteps << ", RHS evals: " << nfevals 
                  << ", Jac evals: " << njevals << std::endl;
        std::cout << "  Setup time: " << std::fixed << std::setprecision(2) << setup_time << " ms" << std::endl;
        std::cout << "  Solve time: " << std::fixed << std::setprecision(2) << total_solve_time << " ms" << std::endl;
    }
    
    RealType getFinalMass() {
        std::vector<RealType> y_host(n);
        RealType* d_y = reinterpret_cast<RealType*>(N_VGetDeviceArrayPointer_Cuda(y));
        cudaMemcpy(y_host.data(), d_y, n * sizeof(RealType), cudaMemcpyDeviceToHost);
        
        RealType total_mass = 0.0;
        for (int i = 0; i < n; i++) {
            total_mass += y_host[i];
        }
        return total_mass;
    }
    
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return total_solve_time; }
    
private:
    void setInitialConditions() {
        std::vector<RealType> y0(n);
        for (int i = 0; i < n; i++) {
            y0[i] = static_cast<RealType>(i + 1);
        }
        
        RealType* d_y = reinterpret_cast<RealType*>(N_VGetDeviceArrayPointer_Cuda(y));
        CUDA_CHECK(cudaMemcpy(d_y, y0.data(), n * sizeof(RealType), cudaMemcpyHostToDevice));
    }
};

// Performance comparison function
void runPrecisionComparison() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PRECISION PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "System size: " << N_STATES << " ODEs" << std::endl;
    std::cout << "Integration time: " << H_0 << std::endl;
    std::cout << "Number of trials: " << NUM_TRIALS << std::endl;
    std::cout << "Parameters: p=" << DIFUSION_COEFFICIENT << ", c=" << COUPLING_STRENGTH << std::endl;
    
    // Storage for timing results
    std::vector<double> double_setup_times, double_solve_times;
    std::vector<double> single_setup_times, single_solve_times;
    std::vector<double> double_masses, single_masses;
    
    std::cout << "\nRunning trials..." << std::endl;
    
    // Run multiple trials for statistical accuracy
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "\n--- Trial " << (trial + 1) << " ---" << std::endl;
        
        // Double precision test
        {
            CVODESGpuSolver<double> solver_double(N_STATES, DIFUSION_COEFFICIENT);
            double solve_time = solver_double.solveODE(H_0, trial == 0); // Verbose on first trial
            
            double_setup_times.push_back(solver_double.getSetupTime());
            double_solve_times.push_back(solve_time);
            double_masses.push_back(solver_double.getFinalMass());
        }
        
        // Single precision test
        {
            CVODESGpuSolver<float> solver_single(N_STATES, (float)DIFUSION_COEFFICIENT);
            double solve_time = solver_single.solveODE(H_0, trial == 0); // Verbose on first trial
            
            single_setup_times.push_back(solver_single.getSetupTime());
            single_solve_times.push_back(solve_time);
            single_masses.push_back(solver_single.getFinalMass());
        }
        
        // Show trial results
        double speedup_setup = double_setup_times[trial] / single_setup_times[trial];
        double speedup_solve = double_solve_times[trial] / single_solve_times[trial];
        
        std::cout << "  Setup speedup: " << std::fixed << std::setprecision(2) << speedup_setup << "x" << std::endl;
        std::cout << "  Solve speedup: " << std::fixed << std::setprecision(2) << speedup_solve << "x" << std::endl;
    }
    
    // Calculate averages
    auto average = [](const std::vector<double>& vec) {
        return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    };
    
    auto stddev = [&average](const std::vector<double>& vec) {
        double avg = average(vec);
        double sum_sq_diff = 0.0;
        for (double val : vec) {
            sum_sq_diff += (val - avg) * (val - avg);
        }
        return std::sqrt(sum_sq_diff / (vec.size() - 1));
    };
    
    double avg_double_setup = average(double_setup_times);
    double avg_double_solve = average(double_solve_times);
    double avg_single_setup = average(single_setup_times);
    double avg_single_solve = average(single_solve_times);
    
    double std_double_solve = stddev(double_solve_times);
    double std_single_solve = stddev(single_solve_times);
    
    // Print comprehensive results
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nSetup Times:" << std::endl;
    std::cout << "  Double precision: " << avg_double_setup << " ± " << stddev(double_setup_times) << " ms" << std::endl;
    std::cout << "  Single precision: " << avg_single_setup << " ± " << stddev(single_setup_times) << " ms" << std::endl;
    std::cout << "  Setup speedup:    " << (avg_double_setup / avg_single_setup) << "x" << std::endl;
    
    std::cout << "\nSolve Times:" << std::endl;
    std::cout << "  Double precision: " << avg_double_solve << " ± " << std_double_solve << " ms" << std::endl;
    std::cout << "  Single precision: " << avg_single_solve << " ± " << std_single_solve << " ms" << std::endl;
    std::cout << "  Solve speedup:    " << (avg_double_solve / avg_single_solve) << "x" << std::endl;
    
    std::cout << "\nTotal Times:" << std::endl;
    double total_double = avg_double_setup + avg_double_solve;
    double total_single = avg_single_setup + avg_single_solve;
    std::cout << "  Double precision: " << total_double << " ms" << std::endl;
    std::cout << "  Single precision: " << total_single << " ms" << std::endl;
    std::cout << "  Total speedup:    " << (total_double / total_single) << "x" << std::endl;
    
    // Accuracy comparison
    std::cout << "\nAccuracy Comparison:" << std::endl;
    double avg_double_mass = average(double_masses);
    double avg_single_mass = average(single_masses);
    double mass_diff = std::abs(avg_double_mass - avg_single_mass);
    double rel_error = mass_diff / avg_double_mass * 100.0;
    
    std::cout << "  Double precision final mass: " << std::setprecision(6) << avg_double_mass << std::endl;
    std::cout << "  Single precision final mass: " << std::setprecision(6) << avg_single_mass << std::endl;
    std::cout << "  Absolute difference:         " << std::setprecision(6) << mass_diff << std::endl;
    std::cout << "  Relative error:              " << std::setprecision(3) << rel_error << "%" << std::endl;
    
    // Memory usage comparison
    std::cout << "\nMemory Usage Comparison:" << std::endl;
    size_t double_mem = N_STATES * sizeof(double) * 4; // Approximate for vectors + matrix
    size_t single_mem = N_STATES * sizeof(float) * 4;
    std::cout << "  Double precision: ~" << (double_mem / 1024) << " KB" << std::endl;
    std::cout << "  Single precision: ~" << (single_mem / 1024) << " KB" << std::endl;
    std::cout << "  Memory reduction: " << std::setprecision(1) << (100.0 * (1.0 - (double)single_mem / double_mem)) << "%" << std::endl;
    
    // Performance summary
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    double overall_speedup = total_double / total_single;
    if (overall_speedup > 1.1) {
        std::cout << "✅ Single precision provides significant speedup: " << std::setprecision(2) << overall_speedup << "x faster" << std::endl;
    } else if (overall_speedup > 1.05) {
        std::cout << "⚡ Single precision provides modest speedup: " << std::setprecision(2) << overall_speedup << "x faster" << std::endl;
    } else {
        std::cout << "⚠️  Single precision provides minimal speedup: " << std::setprecision(2) << overall_speedup << "x faster" << std::endl;
    }
    
    if (rel_error < 0.1) {
        std::cout << "✅ Accuracy difference is negligible (<0.1%)" << std::endl;
    } else if (rel_error < 1.0) {
        std::cout << "⚠️  Accuracy difference is small (<1%)" << std::endl;
    } else {
        std::cout << "❌ Accuracy difference is significant (>1%)" << std::endl;
    }
    
    std::cout << "💾 Memory usage reduced by 50% with single precision" << std::endl;
}

int main() {
    try {
        // CUDA setup
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
        
        // Run the precision comparison
        runPrecisionComparison();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}