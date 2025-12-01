/**
 * @file mcs_cgpops.cpp
 * @brief Monte Carlo simulation driver for CGPOPS spacecraft attitude control optimization
 * 
 * OVERVIEW:
 * =========
 * This program performs Monte Carlo simulations of time-optimal spacecraft attitude
 * maneuvers using the CGPOPS (C++ General Pseudospectral Optimal Control Software)
 * framework. It reads pre-generated initial and final state samples from a CSV file,
 * solves the time-optimal control problem for each state pair, and logs performance
 * statistics.
 * 
 * WORKFLOW:
 * =========
 * 1. Configure CGPOPS global parameters (mesh, solver, output settings)
 * 2. Load Latin Hypercube Sampling (LHS) or Monte Carlo state samples from CSV
 * 3. For each state pair:
 *    a. Solve time-optimal attitude control problem using CGPOPS
 *    b. Extract objective value (minimum time) and solver status
 *    c. Log results to CSV with atomic writes for crash recovery
 *    d. Report progress with time estimates
 * 4. Display summary statistics
 * 
 * INPUT FILES:
 * ============
 * - ../output/mcs/lhs_samples.csv
 *   Format: 14 columns (7 initial state + 7 final state)
 *   State: [q0, q1, q2, q3, wx, wy, wz]
 * 
 * OUTPUT FILES:
 * =============
 * - ../output/mcs/cgpops.csv
 *   Format: T (optimal time), time (computation time), status (solver code)
 * - ../output/solver_logs.log
 *   CGPOPS/IPOPT solver output (redirected, overwritten each iteration)
 * - ../output/cgpopsIPOPTSolution{XX}.m
 *   MATLAB format solution files (one per iteration)
 * 
 * USAGE:
 * ======
 * Compile: g++ -std=c++17 mcs_cgpops.cpp -o MCS_CGPOPS [link flags]
 * Run:     ./MCS_CGPOPS
 * 
 * NOTES:
 * ======
 * - Progress is saved after each iteration (crash-recoverable)
 * - Existing results are appended (set overwrite flag to restart)
 * - File I/O uses atomic operations (fflush + fsync) for data integrity
 * - CGPOPS output redirected to avoid console spam during batch processing
 * 
 * CGPOPS CONFIGURATION:
 * =====================
 * Key parameters configurable via constants:
 * - Derivative supplier: Bicomplex (BC), HyperDual (HD), Finite Differences
 * - Mesh: Initial intervals, collocation points, refinement settings
 * - Solver: IPOPT tolerance, max iterations, NLP settings
 * - Output: Save flags for various solution components
 * 
 * @see configureCGPOPSParameters() for detailed parameter documentation
 * @see cgpops_go() for optimization problem formulation
 * @see loadStateSamples() for input file format details
 * 
 * @author Leonardo Eitner
 * @date 2025
 */

// =============================================================================
// STANDARD LIBRARY HEADERS
// =============================================================================

#include <chrono>       ///< High-resolution timing for performance measurement
#include <cstdio>       ///< C-style file I/O (FILE*, fprintf, fflush)
#include <filesystem>   ///< C++17 filesystem operations (path existence, file size)
#include <fstream>      ///< C++ file streams (ifstream for reading)
#include <iomanip>      ///< I/O manipulators (setprecision, fixed)
#include <iostream>     ///< Console I/O (cout, cerr, endl)
#include <sstream>      ///< String stream operations (if needed for parsing)
#include <string>       ///< String class and operations
#include <vector>       ///< Dynamic array container (state vectors)

// =============================================================================
// POSIX SYSTEM HEADERS
// =============================================================================

#include <unistd.h>     ///< POSIX operating system API (fsync)

// =============================================================================
// EXTERNAL LIBRARY HEADERS
// =============================================================================

#include <helper_functions.h>   ///< Utility functions (I/O, solver interface, status checking)

// =============================================================================
// CGPOPS FRAMEWORK HEADERS
// =============================================================================

#include "nlpGlobVarDec.hpp"    ///< CGPOPS global variable declarations
#include <cgpops/cgpops_main.hpp>   ///< CGPOPS main interface (cgpops_go function)

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================

// -----------------------------------------------------------------------------
// File Paths
// -----------------------------------------------------------------------------

namespace FilePaths {
    constexpr const char* LHS_SAMPLES = "../output/mcs/lhs_samples.csv";
    constexpr const char* RESULTS_CSV = "../output/mcs/cgpops.csv";
    constexpr const char* SOLVER_LOG = "../output/solver_logs.log";
    constexpr const char* SOLVER_STATUS_LOG = "../output/cgpops_results.log";
}

// -----------------------------------------------------------------------------
// Derivative Supplier Codes
// -----------------------------------------------------------------------------
// Maps to automatic differentiation methods in CGPOPS framework

namespace DerivativeSupplier {
    constexpr int HYPERDUAL = 0;        ///< HyperDual numbers (1st + 2nd derivatives)
    constexpr int BICOMPLEX = 1;        ///< Bicomplex numbers (complex-step differentiation)
    constexpr int CENTRAL_FD = 2;       ///< Central finite differences
    constexpr int FORWARD_FD = 3;       ///< Forward/naive finite differences
}

// -----------------------------------------------------------------------------
// Mesh Configuration
// -----------------------------------------------------------------------------
// Initial mesh discretization and refinement settings

namespace MeshConfig {
    constexpr int INITIAL_INTERVALS = 50;       ///< Initial mesh intervals per phase
    constexpr int INITIAL_COLLOCATION_PTS = 6;  ///< Initial collocation points per interval
    constexpr int MIN_COLLOCATION_PTS = 4;      ///< Minimum collocation points (refinement)
    constexpr int MAX_COLLOCATION_PTS = 8;      ///< Maximum collocation points (refinement)
    constexpr int MAX_MESH_ITERATIONS = 1;      ///< Maximum mesh refinement iterations
    constexpr double MESH_TOLERANCE = 1e-5;     ///< Mesh error tolerance
    constexpr int MESH_REFINE_TYPE = 1;         ///< Mesh refinement algorithm type
}

// -----------------------------------------------------------------------------
// Solver Configuration
// -----------------------------------------------------------------------------
// IPOPT nonlinear programming solver settings

namespace SolverConfig {
    constexpr double NLP_TOLERANCE = 1e-7;      ///< NLP optimality tolerance
    constexpr int MAX_NLP_ITERATIONS = 1000;    ///< Maximum IPOPT iterations
    constexpr int RUN_IPOPT = 1;                ///< Enable/disable IPOPT solver
    constexpr int USE_BANG_BANG_DETECTION = 1;  ///< Enable bang-bang control detection
}

// -----------------------------------------------------------------------------
// Output Configuration
// -----------------------------------------------------------------------------
// Control what data is saved from CGPOPS solver

namespace OutputConfig {
    constexpr int SAVE_IPOPT_SOLUTION = 1;      ///< Save IPOPT solution to .m file
    constexpr int SAVE_MESH_HISTORY = 0;        ///< Save mesh refinement history
    constexpr int SAVE_HAMILTONIAN = 0;         ///< Save Hamiltonian values
    constexpr int SAVE_LINEAR_TERMS = 0;        ///< Save linear Hamiltonian terms
    constexpr int ENABLE_SCALING = 1;           ///< Enable variable scaling
    constexpr bool OVERWRITE_EXISTING = false;  ///< Overwrite existing results file
}

// -----------------------------------------------------------------------------
// Progress Reporting
// -----------------------------------------------------------------------------

namespace ProgressConfig {
    constexpr int REPORT_PERCENTAGE = 5;        ///< Report progress every N%
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Configure all CGPOPS global parameters for optimization
 * 
 * Sets global variables used by the CGPOPS framework to control optimization
 * behavior, mesh refinement, solver settings, and output options. These
 * variables are defined in nlpGlobVarDec.hpp and used throughout CGPOPS.
 * 
 * PARAMETER CATEGORIES:
 * - Derivative computation method
 * - Initial mesh discretization
 * - Adaptive mesh refinement
 * - IPOPT solver configuration
 * - Output file generation
 * 
 * @note Must be called before any CGPOPS optimization (cgpops_go)
 * @note Global variables are shared across all CGPOPS calls in process
 * 
 * @see nlpGlobVarDec.hpp for global variable declarations
 */
void configureCGPOPSParameters() {
    // Derivative computation
    derivativeSupplierG = DerivativeSupplier::BICOMPLEX;
    scaledG = OutputConfig::ENABLE_SCALING;
    
    // Initial mesh configuration
    numintervalsG = MeshConfig::INITIAL_INTERVALS;
    initcolptsG = MeshConfig::INITIAL_COLLOCATION_PTS;
    
    // Mesh refinement settings
    meshRefineTypeG = MeshConfig::MESH_REFINE_TYPE;
    minColPtsG = MeshConfig::MIN_COLLOCATION_PTS;
    maxColPtsG = MeshConfig::MAX_COLLOCATION_PTS;
    maxMeshIterG = MeshConfig::MAX_MESH_ITERATIONS;
    meshTolG = MeshConfig::MESH_TOLERANCE;
    
    // Output control
    saveIPOPTFlagG = OutputConfig::SAVE_IPOPT_SOLUTION;
    saveMeshRefineFlagG = OutputConfig::SAVE_MESH_HISTORY;
    saveHamiltonianG = OutputConfig::SAVE_HAMILTONIAN;
    saveLTIHG = OutputConfig::SAVE_LINEAR_TERMS;
    
    // IPOPT solver configuration
    runIPOPTFlagG = SolverConfig::RUN_IPOPT;
    NLPtolG = SolverConfig::NLP_TOLERANCE;
    NLPmaxiterG = SolverConfig::MAX_NLP_ITERATIONS;
    useLTIHDDG = SolverConfig::USE_BANG_BANG_DETECTION;
}

// =============================================================================
// MAIN PROGRAM
// =============================================================================

/**
 * @brief Monte Carlo simulation main entry point
 * 
 * Executes batch time-optimal attitude control optimization using CGPOPS
 * framework on pre-generated state samples. Results logged to CSV for
 * statistical analysis.
 * 
 * @return 0 on success, 1 on error
 */
int main() {
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    std::cout << "========================================" << std::endl;
    std::cout << "CGPOPS Monte Carlo Simulation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configure CGPOPS global parameters
    configureCGPOPSParameters();
    std::cout << "✓ CGPOPS parameters configured" << std::endl;
    
    // CGPOPS results container (reused each iteration)
    doubleMat cgpops_results;
    
    // =========================================================================
    // LOAD INPUT SAMPLES
    // =========================================================================
    
    std::cout << "\nLoading state samples..." << std::endl;
    
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, FilePaths::LHS_SAMPLES)) {
        std::cerr << "Error: Failed to load state samples from " 
                  << FilePaths::LHS_SAMPLES << std::endl;
        return 1;
    }
    
    const int total_samples = static_cast<int>(initial_states.size());
    std::cout << "✓ Loaded " << total_samples << " state pairs" << std::endl;
    
    // Validate sample consistency
    if (initial_states.size() != final_states.size()) {
        std::cerr << "Error: Mismatch between initial and final state counts" << std::endl;
        return 1;
    }
    
    if (total_samples == 0) {
        std::cerr << "Error: No samples loaded from input file" << std::endl;
        return 1;
    }
    
    // =========================================================================
    // SETUP OUTPUT FILE
    // =========================================================================
    
    std::cout << "\nInitializing results file..." << std::endl;
    
    // Determine crash recovery state
    bool output_exists = false;
    int completed_rows = countExistingResults(FilePaths::RESULTS_CSV, output_exists);
    
    if (completed_rows > 0) {
        std::cout << "✓ Found existing results: " << completed_rows << " samples already processed" 
                  << std::endl;
        std::cout << "  Resuming from iteration " << completed_rows << std::endl;
    }
    
    // Initialize results file
    FILE* results_file = initializeResultsFile(
        FilePaths::RESULTS_CSV, OutputConfig::OVERWRITE_EXISTING, output_exists);
    
    if (!results_file) {
        return 1;  // Error message already printed
    }
    
    std::cout << "✓ Results will be logged to: " << FilePaths::RESULTS_CSV << std::endl;
    
    // =========================================================================
    // MONTE CARLO SIMULATION LOOP
    // =========================================================================
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Monte Carlo Simulation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const auto simulation_start = std::chrono::high_resolution_clock::now();
    const int remaining_samples = total_samples - completed_rows;
    const int report_interval = std::max(1, total_samples * ProgressConfig::REPORT_PERCENTAGE / 100);
    
    std::cout << "Processing " << remaining_samples << " samples..." << std::endl;
    
    for (int i = completed_rows; i < total_samples; ++i) {
        // ---------------------------------------------------------------------
        // Solve Optimization Problem
        // ---------------------------------------------------------------------
        
        // Redirect CGPOPS/IPOPT output to log file (avoid console spam)
        redirect_output_to_file(FilePaths::SOLVER_LOG);
        
        // Time the optimization
        const auto iteration_start = std::chrono::high_resolution_clock::now();
        
        // Solve time-optimal attitude control problem
        cgpops_go(cgpops_results, initial_states[i], final_states[i]);
        
        const auto iteration_end = std::chrono::high_resolution_clock::now();
        const double computation_time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
            iteration_end - iteration_start).count() / 1000.0;
        
        // Restore console output
        restore_output_to_console();
        
        // ---------------------------------------------------------------------
        // Extract Results
        // ---------------------------------------------------------------------
        
        // Get optimal time (objective value)
        const double optimal_time = getCgpopsSolution(derivativeSupplierG);
        
        // Get solver status code
        const int solver_status = get_solver_status(FilePaths::SOLVER_STATUS_LOG);
        
        // ---------------------------------------------------------------------
        // Log Results (Atomic Write for Crash Recovery)
        // ---------------------------------------------------------------------
        
        std::fprintf(results_file, "%.3f,%.3f,%d\n", 
                    optimal_time, computation_time_sec, solver_status);
        std::fflush(results_file);       // Flush to OS buffer
        fsync(fileno(results_file));     // Force write to disk
        
        // ---------------------------------------------------------------------
        // Progress Reporting
        // ---------------------------------------------------------------------
        
        if ((i + 1) % report_interval == 0 || (i + 1) == total_samples) {
            reportProgress(i + 1, total_samples, simulation_start);
        }
    }
    
    // =========================================================================
    // CLEANUP AND SUMMARY
    // =========================================================================
    
    std::fclose(results_file);
    
    const auto simulation_end = std::chrono::high_resolution_clock::now();
    const auto total_elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
        simulation_end - simulation_start).count();
    
    const int total_minutes = total_elapsed_sec / 60;
    const int total_seconds = total_elapsed_sec % 60;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Simulation Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total samples processed: " << total_samples << std::endl;
    std::cout << "Total elapsed time: " << total_minutes << " min " << total_seconds << " sec" 
              << std::endl;
    std::cout << "Results saved to: " << FilePaths::RESULTS_CSV << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}