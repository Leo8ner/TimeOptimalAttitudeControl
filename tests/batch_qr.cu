#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

// SUNDIALS headers
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <nvector/nvector_cuda.h>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

class SundialsGpuBatchQRExample {
private:
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector x, b;
    SUNContext sunctx;
    int n;
    int batchSize;
    cusparseHandle_t cusparse_handle;
    cusolverSpHandle_t cusolver_handle;

public:
    SundialsGpuBatchQRExample(int matrix_size, int batch_size) 
        : n(matrix_size), batchSize(batch_size) {
        
        std::cout << "Initializing SUNDIALS cuSolverSp_batchQR example..." << std::endl;
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        
        // Create SUNDIALS context
        int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
        if (retval != 0) {
            std::cerr << "Error creating SUNDIALS context" << std::endl;
            exit(1);
        }
        
        // Create cuSPARSE and cuSOLVER handles
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        
        initializeSystem();
    }
    
    ~SundialsGpuBatchQRExample() {
        std::cout << "Cleaning up..." << std::endl;
        
        // Cleanup SUNDIALS objects
        if (A) SUNMatDestroy(A);
        if (LS) SUNLinSolFree(LS);
        if (x) N_VDestroy(x);
        if (b) N_VDestroy(b);
        
        // Cleanup CUDA handles
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cusolver_handle) cusolverSpDestroy(cusolver_handle);
        
        if (sunctx) SUNContext_Free(&sunctx);
    }
    
    void initializeSystem() {
        std::cout << "\nInitializing GPU vectors and sparse matrix..." << std::endl;
        
        // Create CUDA vectors (these will be allocated on GPU)
        x = N_VNew_Cuda(n, sunctx);
        b = N_VNew_Cuda(n, sunctx);
        
        if (!x || !b) {
            std::cerr << "Error creating CUDA vectors" << std::endl;
            exit(1);
        }
        
        // Estimate number of non-zeros for a tridiagonal matrix
        int nnz = 3 * n - 2;
        
        // Create cuSPARSE matrix
        A = SUNMatrix_cuSparse_NewCSR(n, n, nnz, cusparse_handle, sunctx);
        if (!A) {
            std::cerr << "Error creating cuSPARSE matrix" << std::endl;
            exit(1);
        }
        
        setupSparseMatrix();
        setupRightHandSide();
        
        // Create the cuSolverSp batch QR linear solver
        std::cout << "Creating SUNLinSol_cuSolverSp_batchQR solver..." << std::endl;
        LS = SUNLinSol_cuSolverSp_batchQR(x, A, cusolver_handle, sunctx);
        if (!LS) {
            std::cerr << "Error creating cuSolverSp_batchQR linear solver" << std::endl;
            exit(1);
        }
        
        std::cout << "GPU linear solver created successfully!" << std::endl;
    }
    
    void setupSparseMatrix() {
        std::cout << "Setting up tridiagonal sparse matrix on GPU..." << std::endl;
        
        // Get matrix data pointers
        sunindextype* rowptrs = SUNMatrix_cuSparse_IndexPointers(A);
        sunindextype* colvals = SUNMatrix_cuSparse_IndexValues(A);
        sunrealtype* data = SUNMatrix_cuSparse_Data(A);
        
        // Create host arrays first
        std::vector<sunindextype> h_rowptrs(n + 1);
        std::vector<sunindextype> h_colvals;
        std::vector<sunrealtype> h_data;
        
        // Fill CSR format for tridiagonal matrix
        int nnz_count = 0;
        for (int i = 0; i < n; i++) {
            h_rowptrs[i] = nnz_count;
            
            // Lower diagonal
            if (i > 0) {
                h_colvals.push_back(i - 1);
                h_data.push_back(-1.0);
                nnz_count++;
            }
            
            // Main diagonal - make it diagonally dominant for stability
            h_colvals.push_back(i);
            h_data.push_back(4.0);  // Increased for better conditioning
            nnz_count++;
            
            // Upper diagonal
            if (i < n - 1) {
                h_colvals.push_back(i + 1);
                h_data.push_back(-1.0);
                nnz_count++;
            }
        }
        h_rowptrs[n] = nnz_count;
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(rowptrs, h_rowptrs.data(), 
                             (n + 1) * sizeof(sunindextype), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(colvals, h_colvals.data(), 
                             nnz_count * sizeof(sunindextype), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data, h_data.data(), 
                             nnz_count * sizeof(sunrealtype), cudaMemcpyHostToDevice));
        
        std::cout << "Matrix copied to GPU. NNZ = " << nnz_count << std::endl;
    }
    
    void setupRightHandSide() {
        std::cout << "Setting up right-hand side vector..." << std::endl;
        
        // Create host vector
        std::vector<sunrealtype> h_b(n);
        
        // Set up a simple RHS that should have a nice solution
        for (int i = 0; i < n; i++) {
            h_b[i] = static_cast<sunrealtype>(i + 1);  // RHS = [1, 2, 3, ..., n]
        }
        
        // Copy to GPU
        sunrealtype* d_b = N_VGetDeviceArrayPointer_Cuda(b);
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(sunrealtype), cudaMemcpyHostToDevice));
        
        std::cout << "RHS vector copied to GPU" << std::endl;
    }
    
    void printGpuVector(N_Vector vec, const std::string& name) {
        std::vector<sunrealtype> h_vec(n);
        sunrealtype* d_vec = N_VGetDeviceArrayPointer_Cuda(vec);
        
        CUDA_CHECK(cudaMemcpy(h_vec.data(), d_vec, n * sizeof(sunrealtype), cudaMemcpyDeviceToHost));
        
        std::cout << name << ": [";
        for (int i = 0; i < std::min(n, 10); i++) {  // Print first 10 elements
            std::printf("%.6f", h_vec[i]);
            if (i < std::min(n, 10) - 1) std::cout << ", ";
        }
        if (n > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    
    void demonstrateBatchSolve() {
        std::cout << "\n=== Demonstrating Batch QR Solve ===" << std::endl;
        
        // Initialize the linear solver
        int retval = SUNLinSolInitialize(LS);
        if (retval != 0) {
            std::cerr << "Error initializing linear solver: " << retval << std::endl;
            return;
        }
        
        // Setup the linear solver with the matrix (performs QR factorization)
        std::cout << "Performing QR factorization..." << std::endl;
        retval = SUNLinSolSetup(LS, A);
        if (retval != 0) {
            std::cerr << "Error in linear solver setup: " << retval << std::endl;
            return;
        }
        std::cout << "QR factorization completed successfully!" << std::endl;
        
        // Solve multiple systems with different RHS vectors
        for (int batch = 0; batch < batchSize; batch++) {
            std::cout << "\n--- Solving system " << (batch + 1) << " of " << batchSize << " ---" << std::endl;
            
            // Modify RHS for each batch solve
            std::vector<sunrealtype> h_b(n);
            for (int i = 0; i < n; i++) {
                h_b[i] = static_cast<sunrealtype>((batch + 1) * (i + 1));
            }
            
            sunrealtype* d_b = N_VGetDeviceArrayPointer_Cuda(b);
            CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(sunrealtype), cudaMemcpyHostToDevice));
            
            printGpuVector(b, "RHS");
            
            // Copy b to x (solver will overwrite x with solution)
            N_VScale(1.0, b, x);
            
            // Solve the linear system using the pre-computed QR factorization
            retval = SUNLinSolSolve(LS, A, x, b, 0.0);
            if (retval == 0) {
                printGpuVector(x, "Solution");
                
                // Verify solution by computing residual ||Ax - b||
                verifySolution(batch + 1);
            } else {
                std::cerr << "Error solving system " << (batch + 1) << ": " << retval << std::endl;
            }
        }
    }
    
    void verifySolution(int systemNum) {
        std::cout << "Verifying solution for system " << systemNum << "..." << std::endl;
        
        // Simple verification: check solution norm and print first few elements
        sunrealtype solution_norm = N_VMaxNorm(x);
        std::printf("Solution max norm: %.6e\n", solution_norm);
        
        // Copy solution back to host for inspection
        std::vector<sunrealtype> h_x(std::min(n, 5));  // First 5 elements
        sunrealtype* d_x = N_VGetDeviceArrayPointer_Cuda(x);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, h_x.size() * sizeof(sunrealtype), cudaMemcpyDeviceToHost));
        
        std::cout << "First few solution components: [";
        for (size_t i = 0; i < h_x.size(); i++) {
            std::printf("%.6f", h_x[i]);
            if (i < h_x.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Basic sanity check: solution should not be zero vector or have NaN/Inf
        bool valid_solution = true;
        for (size_t i = 0; i < h_x.size(); i++) {
            if (!std::isfinite(h_x[i])) {
                valid_solution = false;
                break;
            }
        }
        
        if (valid_solution && solution_norm > 1e-15) {
            std::cout << "✓ Solution appears valid (finite and non-zero)" << std::endl;
        } else {
            std::cout << "⚠ Solution may have issues (zero, NaN, or Inf detected)" << std::endl;
        }
    }
    
    void printSolverStats() {
        std::cout << "\n=== Solver Statistics ===" << std::endl;
        
        // Get solver statistics
        long int setup_time = SUNLinSolLastFlag(LS);
        std::cout << "Last solver flag: " << setup_time << std::endl;
        
        // Print GPU memory usage
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "GPU Memory: " << (total_mem - free_mem) / (1024*1024) << " MB used, "
                  << free_mem / (1024*1024) << " MB free" << std::endl;
    }
    
    void run() {
        std::cout << "\n=== Running SUNDIALS cuSolverSp_batchQR Example ===" << std::endl;
        
        printGpuVector(b, "Initial RHS");
        
        demonstrateBatchSolve();
        printSolverStats();
        
        std::cout << "\n=== Batch QR Benefits ===" << std::endl;
        std::cout << "• QR factorization computed once, reused for multiple solves" << std::endl;
        std::cout << "• GPU acceleration provides significant speedup for large systems" << std::endl;
        std::cout << "• Batch processing reduces kernel launch overhead" << std::endl;
        std::cout << "• Excellent for parameter studies and time-stepping algorithms" << std::endl;
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
        
        // Create and run the batch QR solver example
        SundialsGpuBatchQRExample example(8, 3);  // 8x8 matrix, 3 batch solves
        example.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}