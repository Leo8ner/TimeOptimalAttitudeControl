// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved
//
// cgpops_main_test.cpp
// 
// Purpose: Test driver for CGPOPS optimal control solver
// 
// Usage: ./run_cgpops "phi_i,theta_i,psi_i,wx_i,wy_i,wz_i" 
//                            "phi_f,theta_f,psi_f,wx_f,wy_f,wz_f"
//
// Description:
//   This program solves a spacecraft attitude reorientation problem using
//   CGPOPS. It accepts initial and final state vectors (Euler angles and
//   angular velocities) as command-line arguments, configures the solver,
//   executes the optimization, and generates visualization via Python script.
//

#include "nlpGlobVarDec.hpp"
#include <cgpops/cgpops_main.hpp>
#include <helper_functions.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>


int main(int argc, char* argv[])
{

    // ========================================================================
    // Command-Line Argument Parsing
    // ========================================================================
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" "
                  << "\"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
        return 1;
    }
    
    // Parse initial and final state vectors from command-line arguments
    std::vector<std::vector<double>> input = VparseStateVector(argv[1], argv[2]);
    std::vector<double> initial_state = input[0];  // [q0, q1, q2, q3, wx, wy, wz]
    std::vector<double> final_state = input[1];    // [q0, q1, q2, q3, wx, wy, wz]
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "CGPOPS SPACECRAFT REORIENTATION SOLVER" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========================================================================
    // CGPOPS Solver Configuration
    // ========================================================================
    int numTestRuns = 1;                                 // Number of test runs for changing mesh settings
    int numDS = 1;                                       // Number of derivative suppliers to test
    doubleMatMat cgpopsResultsMatMat(numTestRuns*numDS); // To store results from multiple runs
    doubleMat cgpopsResults;                             // To store results from a single run
    
    // ========================================================================
    // Derivative and Scaling Configuration
    // ========================================================================
    
    derivativeSupplierG = 1;  // Select derivative supplier (0=Hyperdual, 1=Bicomplex, 2=CentralDiff, 3=CentralNaive, 4=AD)
    scaledG             = 1;  // Enable variable scaling for numerical stability
    
    // ========================================================================
    // Mesh Initialization Settings
    // ========================================================================
    
    numintervalsG = 50;  // Initial mesh intervals per phase
    initcolptsG   = 5;   // Initial Legendre-Gauss-Radau collocation points per interval
    
    // ========================================================================
    // Mesh Refinement Configuration
    // ========================================================================
    
    meshRefineTypeG = 1;       // Refinement strategy: (1=hp-Patterson, 2=hp-Darby, 3=hp-Liu, 4=hp-Legendre)
    minColPtsG      = 4;       // Minimum collocation points per interval
    maxColPtsG      = 10;      // Maximum collocation points per interval
    maxMeshIterG    = 1;       // Maximum mesh refinement iterations
    meshTolG        = 1e-7;    // Relative error tolerance for mesh refinement
    
    // ========================================================================
    // Output and Diagnostics Configuration
    // ========================================================================
    
    saveIPOPTFlagG      = 1;  // Save IPOPT solution trajectory
    saveMeshRefineFlagG = 0;  // Disable mesh refinement history (reduces file I/O)
    saveHamiltonianG    = 0;  // Disable Hamiltonian output (not needed for this test)
    saveLTIHG           = 0;  // Disable linear-term-in-Hamiltonian output
    
    // ========================================================================
    // NLP Solver (IPOPT) Configuration
    // ========================================================================
    
    runIPOPTFlagG  = 1;      // Execute IPOPT optimization
    NLPtolG        = 1e-7;   // Convergence tolerance for optimality conditions
    NLPmaxiterG    = 1000;   // Maximum IPOPT iterations
    useLTIHDDG     = 1;      // Enable bang-bang control detection for time-optimal problems
    
    for (int ds=0; ds<numDS; ds++)
    {
        // derivativeSupplierG = ds; // Uncomment to test multiple derivative suppliers

        for (int ni=0; ni<numTestRuns; ni++)
        {
            // numintervalsG += 10; // Uncomment to test varying mesh sizes

            std::cout << "Configuration:" << std::endl;
            std::cout << "  Derivative Supplier: " << derivativeSupplierG << std::endl;
            std::cout << "  Initial Mesh Intervals: " << numintervalsG << std::endl;
            std::cout << "  NLP Tolerance: " << NLPtolG << std::endl;
            std::cout << "\nSolving...\n" << std::endl;

            // ========================================================================
            // Solve Optimal Control Problem
            // ========================================================================
            cgpops_go(cgpopsResults, initial_state, final_state);

            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns] = getDoubleMat(6); // Placeholder for storing multiple results
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[0] = derivativeSupplierG;
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[1] = numintervalsG;
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[2] = cgpopsResults.val[0];
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[3] = cgpopsResults.val[1];
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[4] = cgpopsResults.val[2];
            // cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[5] = cgpopsResults.val[3];

        }

    }
    
    // ========================================================================
    // Extract and Display Results
    // ========================================================================
    
    double final_time = getCgpopsSolution(derivativeSupplierG);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Final Time: " << final_time << " seconds" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========================================================================
    // Generate Visualization
    // ========================================================================
    
    std::string command = "python3 ../src/lib/cgpops/animation.py " 
                        + fileSufix(derivativeSupplierG) + " "
                        + std::to_string(i_x) + " "
                        + std::to_string(i_y) + " "
                        + std::to_string(i_z);
    
    std::cout << "Launching visualization..." << std::endl;
    int system_result = std::system(command.c_str());
    
    if (system_result != 0) {
        std::cerr << "Warning: Visualization script returned non-zero exit code (" 
                  << system_result << ")" << std::endl;
    }
    
    return 0;
}


