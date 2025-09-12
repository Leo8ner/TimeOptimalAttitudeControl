#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>

// SUNDIALS headers (CPU versions)
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_spgmr.h>

// Define constants for batch processing
#define N_STATES_PER_SYSTEM 7
#define N_STEPS 256
#define DELTA_T 0.01  // Time step for integration

// Default physical parameters
#define I_X 0.5                   
#define I_Y 1.2
#define I_Z 0.8

// Sparsity constants (same as parallel version)
#define NNZ_PER_BLOCK 37  // 4*6 + 3*2 = 24 + 6 + 7 diagonal = 37 nonzeros per block
#define I_X 0.5                   
#define I_Y 1.2
#define I_Z 0.8

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

// Serial spacecraft solver class
class SerialSpacecraftSolver {
private:
    SUNMatrix A;
    SUNLinearSolver LS;
    N_Vector y;
    
    std::vector<StepParams> step_params;
    std::vector<std::vector<sunrealtype>> solutions;
    
    void* cvode_mem;
    SUNContext sunctx;
    
    // Current step being processed (for RHS function)
    int current_step;
    
    // Timing
    PrecisionTimer timer;
    double setup_time;
    double solve_time;

public:
    SerialSpacecraftSolver() : current_step(0), setup_time(0), solve_time(0) {
        timer.start();
        
        // Initialize SUNDIALS context
        int retval = SUNContext_Create(NULL, &sunctx);
        if (retval != 0) {
            std::cerr << "Error creating SUNDIALS context" << std::endl;
            exit(1);
        }
        
        // Initialize vectors for single system
        y = N_VNew_Serial(N_STATES_PER_SYSTEM, sunctx);
        if (!y) {
            std::cerr << "Error creating serial vector" << std::endl;
            exit(1);
        }
        
        // Create sparse matrix for single system (same structure as parallel version)
        A = SUNSparseMatrix(N_STATES_PER_SYSTEM, N_STATES_PER_SYSTEM, NNZ_PER_BLOCK, CSR_MAT, sunctx);
        if (!A) {
            std::cerr << "Error creating sparse matrix" << std::endl;
            exit(1);
        }
        
        setupSparseJacobianStructure();
        
        // SPGMR iterative sparse linear solver (no external dependencies)
        // Still maintains sparse structure for fair comparison with GPU version
        LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 10, sunctx);  // 10 = max Krylov dimension
        if (!LS) {
            std::cerr << "Error creating SPGMR linear solver" << std::endl;
            exit(1);
        }
        
        generateRandomSteps();
        solutions.resize(N_STEPS);
        for (int i = 0; i < N_STEPS; i++) {
            solutions[i].resize(N_STATES_PER_SYSTEM);
        }
        
        setup_time = timer.getElapsedMs();
    }
    
    ~SerialSpacecraftSolver() {
        if (cvode_mem) CVodeFree(&cvode_mem);
        if (A) SUNMatDestroy(A);
        if (LS) SUNLinSolFree(LS);
        if (y) N_VDestroy(y);
        if (sunctx) SUNContext_Free(&sunctx);
    }

    void setupSparseJacobianStructure() {
        // Same sparsity pattern as parallel version
        sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(A);
        sunindextype* colvals = SUNSparseMatrix_IndexValues(A);
        
        // Build CSR structure (columns must be in ascending order)
        int nnz_count = 0;
        
        // Row 0: columns 0,1,2,3,4,5,6
        rowptrs[0] = nnz_count;
        for (int j = 0; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 1: columns 0,1,2,3,4,5,6
        rowptrs[1] = nnz_count;
        for (int j = 0; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 2: columns 0,1,2,3,4,5,6
        rowptrs[2] = nnz_count;
        for (int j = 0; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 3: columns 0,1,2,3,4,5,6
        rowptrs[3] = nnz_count;
        for (int j = 0; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 4: columns 4,5,6
        rowptrs[4] = nnz_count;
        for (int j = 4; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 5: columns 4,5,6
        rowptrs[5] = nnz_count;
        for (int j = 4; j < 7; j++) colvals[nnz_count++] = j;
        
        // Row 6: columns 4,5,6
        rowptrs[6] = nnz_count;
        for (int j = 4; j < 7; j++) colvals[nnz_count++] = j;
        
        // Final row pointer
        rowptrs[7] = nnz_count;
    }
    
    void generateRandomSteps() {
        std::random_device rd;
        std::mt19937 gen(1);  // Fixed seed for reproducibility (same as parallel version)

        // Distributions for random parameters
        std::uniform_real_distribution<float> torque_dist(-1, 1);
        std::uniform_real_distribution<float> quat_dist(0, 1);
        std::uniform_real_distribution<float> omega_dist(-1, 1);
        
        step_params.resize(N_STEPS);
        for (int i = 0; i < N_STEPS; i++) {
            auto& params = step_params[i];
            
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
    }
    
    void initializeCVODESForStep(int step) {
        current_step = step;
        
        // Create new CVODES instance for this step
        cvode_mem = CVodeCreate(CV_BDF, sunctx);
        if (!cvode_mem) {
            std::cerr << "Error creating CVODES memory" << std::endl;
            exit(1);
        }
        
        setInitialConditions(step);
        
        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != CV_SUCCESS) {
            std::cerr << "Error initializing CVODES: " << retval << std::endl;
            exit(1);
        }
        
        CVodeSetUserData(cvode_mem, this);
        
        // Same tolerances as parallel version
        retval = CVodeSStolerances(cvode_mem, 1e-6, 1e-8);
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
        SerialSpacecraftSolver* solver = static_cast<SerialSpacecraftSolver*>(user_data);
        
        sunrealtype* y_data = N_VGetArrayPointer(y);
        sunrealtype* ydot_data = N_VGetArrayPointer(ydot);
        
        // Get current step parameters
        const StepParams& params = solver->step_params[solver->current_step];
        
        // Extract state variables
        sunrealtype q0 = y_data[0], q1 = y_data[1];
        sunrealtype q2 = y_data[2], q3 = y_data[3];
        sunrealtype wx = y_data[4], wy = y_data[5], wz = y_data[6];
        
        const sunrealtype half = 0.5;
        const sunrealtype Ix_inv = 1.0 / I_X;
        const sunrealtype Iy_inv = 1.0 / I_Y; 
        const sunrealtype Iz_inv = 1.0 / I_Z;
        
        // Quaternion derivatives
        ydot_data[0] = half * (-q1*wx - q2*wy - q3*wz);
        ydot_data[1] = half * ( q0*wx - q3*wy + q2*wz);
        ydot_data[2] = half * ( q3*wx + q0*wy - q1*wz);
        ydot_data[3] = half * (-q2*wx + q1*wy + q0*wz);
        
        // Angular velocity derivatives
        sunrealtype Iw_x = I_X * wx, Iw_y = I_Y * wy, Iw_z = I_Z * wz;
        
        ydot_data[4] = Ix_inv * (params.tau_x - (wy * Iw_z - wz * Iw_y));
        ydot_data[5] = Iy_inv * (params.tau_y - (wz * Iw_x - wx * Iw_z));
        ydot_data[6] = Iz_inv * (params.tau_z - (wx * Iw_y - wy * Iw_x));
        
        return 0;
    }
    
    static int jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                               SUNMatrix Jac, void* user_data, 
                               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        SerialSpacecraftSolver* solver = static_cast<SerialSpacecraftSolver*>(user_data);
        
        sunrealtype* y_data = N_VGetArrayPointer(y);
        sunrealtype* jac_data = SUNSparseMatrix_Data(Jac);
        
        // Extract state variables
        sunrealtype q0 = y_data[0], q1 = y_data[1];
        sunrealtype q2 = y_data[2], q3 = y_data[3];
        sunrealtype wx = y_data[4], wy = y_data[5], wz = y_data[6];
        
        const sunrealtype Ix = I_X, Iy = I_Y, Iz = I_Z;
        const sunrealtype Ix_inv = 1.0/Ix, Iy_inv = 1.0/Iy, Iz_inv = 1.0/Iz;
        
        // Fill sparse Jacobian entries in CSR order (same as parallel version)
        int idx = 0;
        
        // Row 0: q0_dot = 0.5 * (-q1*wx - q2*wy - q3*wz)
        // Columns: 0,1,2,3,4,5,6
        jac_data[idx++] = 0.0;        // J[0,0] diagonal
        jac_data[idx++] = -0.5 * wx;  // J[0,1]
        jac_data[idx++] = -0.5 * wy;  // J[0,2]
        jac_data[idx++] = -0.5 * wz;  // J[0,3]
        jac_data[idx++] = -0.5 * q1;  // J[0,4]
        jac_data[idx++] = -0.5 * q2;  // J[0,5]
        jac_data[idx++] = -0.5 * q3;  // J[0,6]
        
        // Row 1: q1_dot = 0.5 * (q0*wx - q3*wy + q2*wz)
        // Columns: 0,1,2,3,4,5,6
        jac_data[idx++] =  0.5 * wx;  // J[1,0]
        jac_data[idx++] = 0.0;        // J[1,1] diagonal
        jac_data[idx++] =  0.5 * wz;  // J[1,2]
        jac_data[idx++] = -0.5 * wy;  // J[1,3]
        jac_data[idx++] =  0.5 * q0;  // J[1,4]
        jac_data[idx++] = -0.5 * q3;  // J[1,5]
        jac_data[idx++] =  0.5 * q2;  // J[1,6]
        
        // Row 2: q2_dot = 0.5 * (q3*wx + q0*wy - q1*wz)
        // Columns: 0,1,2,3,4,5,6
        jac_data[idx++] =  0.5 * wy;  // J[2,0]
        jac_data[idx++] = -0.5 * wz;  // J[2,1]
        jac_data[idx++] = 0.0;        // J[2,2] diagonal
        jac_data[idx++] =  0.5 * wx;  // J[2,3]
        jac_data[idx++] =  0.5 * q3;  // J[2,4]
        jac_data[idx++] =  0.5 * q0;  // J[2,5]
        jac_data[idx++] = -0.5 * q1;  // J[2,6]
        
        // Row 3: q3_dot = 0.5 * (-q2*wx + q1*wy + q0*wz)
        // Columns: 0,1,2,3,4,5,6
        jac_data[idx++] =  0.5 * wz;  // J[3,0]
        jac_data[idx++] =  0.5 * wy;  // J[3,1]
        jac_data[idx++] = -0.5 * wx;  // J[3,2]
        jac_data[idx++] = 0.0;        // J[3,3] diagonal
        jac_data[idx++] = -0.5 * q2;  // J[3,4]
        jac_data[idx++] =  0.5 * q1;  // J[3,5]
        jac_data[idx++] =  0.5 * q0;  // J[3,6]
        
        // Row 4: wx_dot
        // Columns: 4,5,6
        jac_data[idx++] = 0.0;                       // J[4,4] diagonal
        jac_data[idx++] = -Ix_inv * (Iz - Iy) * wz;  // J[4,5]
        jac_data[idx++] = -Ix_inv * (Iz - Iy) * wy;  // J[4,6]
        
        // Row 5: wy_dot
        // Columns: 4,5,6
        jac_data[idx++] = -Iy_inv * (Ix - Iz) * wz;  // J[5,4]
        jac_data[idx++] = 0.0;                       // J[5,5] diagonal
        jac_data[idx++] = -Iy_inv * (Ix - Iz) * wx;  // J[5,6]
        
        // Row 6: wz_dot
        // Columns: 4,5,6
        jac_data[idx++] = -Iz_inv * (Iy - Ix) * wy;  // J[6,4]
        jac_data[idx++] = -Iz_inv * (Iy - Ix) * wx;  // J[6,5]
        jac_data[idx++] = 0.0;                       // J[6,6] diagonal
        
        return 0;
    }
    
    double solveBatch() {
        timer.start();
        
        long int total_steps = 0, total_fevals = 0, total_linsetups = 0;
        long int total_jevals = 0, total_liters = 0;
        
        // Process each step sequentially
        for (int step = 0; step < N_STEPS; step++) {
            // Initialize CVODES for this step
            initializeCVODESForStep(step);
            
            // Define the time step size
            sunrealtype t_final = DELTA_T;
            sunrealtype t = 0.0;
            
            // Solve this step
            int retval = CVode(cvode_mem, t_final, y, &t, CV_NORMAL);
            
            if (retval < 0) {
                std::cerr << "CVode error for step " << step << ": " << retval << std::endl;
                CVodeFree(&cvode_mem);
                continue;
            }
            
            // Store solution
            sunrealtype* y_data = N_VGetArrayPointer(y);
            for (int i = 0; i < N_STATES_PER_SYSTEM; i++) {
                solutions[step][i] = y_data[i];
            }
            
            // Accumulate statistics
            long int nsteps, nfevals, nlinsetups, njevals, nliters;
            CVodeGetNumSteps(cvode_mem, &nsteps);
            CVodeGetNumRhsEvals(cvode_mem, &nfevals);
            CVodeGetNumLinSolvSetups(cvode_mem, &nlinsetups);
            CVodeGetNumJacEvals(cvode_mem, &njevals);
            CVodeGetNumLinIters(cvode_mem, &nliters);
            
            total_steps += nsteps;
            total_fevals += nfevals;
            total_linsetups += nlinsetups;
            total_jevals += njevals;
            total_liters += nliters;
            
            // Free CVODES memory for this step
            CVodeFree(&cvode_mem);
            cvode_mem = nullptr;
        }
        
        // Store accumulated statistics for printing
        accumulated_stats.nsteps = total_steps;
        accumulated_stats.nfevals = total_fevals;
        accumulated_stats.nlinsetups = total_linsetups;
        accumulated_stats.njevals = total_jevals;
        accumulated_stats.nliters = total_liters;
        
        solve_time = timer.getElapsedMs();
        return solve_time;
    }
    
    void printSolutionStats() {
        std::cout << "Serial integration stats:" << std::endl;
        std::cout << "  Setup time: " << setup_time << " ms" << std::endl;
        std::cout << "  Solve time: " << solve_time << " ms" << std::endl;
        std::cout << "  Total time: " << (setup_time + solve_time) << " ms" << std::endl;
        std::cout << "  Steps: " << accumulated_stats.nsteps << std::endl;
        std::cout << "  RHS evaluations: " << accumulated_stats.nfevals << std::endl;
        std::cout << "  Jacobian evaluations: " << accumulated_stats.njevals << std::endl;
        std::cout << "  Linear solver setups: " << accumulated_stats.nlinsetups << std::endl;
        std::cout << "  Linear iterations: " << accumulated_stats.nliters << std::endl;
        
        auto quaternion_norms = getQuaternionNorms();
        double avg_norm = std::accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / N_STEPS;
        double max_deviation = 0.0;
        for (auto norm : quaternion_norms) {
            max_deviation = std::max(max_deviation, std::abs(static_cast<double>(norm) - 1.0));
        }
        
        std::cout << "  Average quaternion norm: " << std::setprecision(6) << avg_norm << std::endl;
        std::cout << "  Maximum norm deviation: " << std::setprecision(2) << (max_deviation * 100.0) << "%" << std::endl;
    }
    
    std::vector<sunrealtype> getQuaternionNorms() {
        std::vector<sunrealtype> norms(N_STEPS);
        for (int i = 0; i < N_STEPS; i++) {
            norms[i] = sqrt(solutions[i][0]*solutions[i][0] + 
                           solutions[i][1]*solutions[i][1] + 
                           solutions[i][2]*solutions[i][2] + 
                           solutions[i][3]*solutions[i][3]);
        }
        return norms;
    }
    
    std::vector<std::vector<sunrealtype>> getAllSolutions() {
        return solutions;
    }
    
    double getSetupTime() const { return setup_time; }
    double getSolveTime() const { return solve_time; }
    double getTotalTime() const { return setup_time + solve_time; }
    
private:
    // Structure to store accumulated statistics
    struct {
        long int nsteps = 0;
        long int nfevals = 0;
        long int nlinsetups = 0;
        long int njevals = 0;
        long int nliters = 0;
    } accumulated_stats;
    
    void setInitialConditions(int step) {
        const auto& params = step_params[step];
        
        sunrealtype* y_data = N_VGetArrayPointer(y);
        y_data[0] = params.q0; // q0
        y_data[1] = params.q1; // q1
        y_data[2] = params.q2; // q2
        y_data[3] = params.q3; // q3
        y_data[4] = params.wx; // wx
        y_data[5] = params.wy; // wy
        y_data[6] = params.wz; // wz
    }
};

int main() {
    try {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "SERIAL SPACECRAFT TRAJECTORY SOLVER" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "Number of steps: " << N_STEPS << std::endl;
        std::cout << "States per step: " << N_STATES_PER_SYSTEM << std::endl;
        std::cout << "Total trajectories: " << N_STEPS << " (processed sequentially)" << std::endl;
        
        // Solve all trajectories
        SerialSpacecraftSolver solver;
        double solve_time = solver.solveBatch();
        solver.printSolutionStats();
        
        // Display a few sample solutions
        auto solutions = solver.getAllSolutions();
        std::cout << "\nSample final states (first 3 steps):" << std::endl;
        for (int i = 0; i < std::min(3, N_STEPS); i++) {
            std::cout << "  Step " << i << ": quat=[" << std::setprecision(4)
                      << solutions[i][0] << "," << solutions[i][1] << "," 
                      << solutions[i][2] << "," << solutions[i][3] << "], "
                      << "ω=[" << solutions[i][4] << "," << solutions[i][5] << "," 
                      << solutions[i][6] << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Compile with:
// g++ -O3 -o serial serial.cpp -I$SUNDIALS_DIR/include -L$SUNDIALS_DIR/lib -lsundials_cvodes -lsundials_nvecserial -lsundials_sunmatrixsparse -lsundials_sunlinsolspgmr -lsundials_core