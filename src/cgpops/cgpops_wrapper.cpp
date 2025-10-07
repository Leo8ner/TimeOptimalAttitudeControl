// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao.  All Rights Reserved
//
// cgpops_main_test.cpp
// Test Main
// Test CGPOPS functions
//

#include "nlpGlobVarDec.hpp"
#include <cgpops/cgpops_main.hpp>
#include <string>
#include <vector>
#include <cgpops/helper_functions.h>


int main(int argc, char* argv[])
{

    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::vector<double> initial_state, final_state;
    initial_state = parseStateVector(argv[1]);
    final_state = parseStateVector(argv[2]);

    printf("\nCGPOPS TESTING\n\n\n");
    
    int numTestRuns = 1;
    int numDS = 1;
    doubleMatMat cgpopsResultsMatMat(numTestRuns*numDS);
    doubleMat cgpopsResults;
    
    // Derivative supplier settings
    derivativeSupplierG = 1;    // Derivative supplier (default=0)
    scaledG             = 1;    // Scaling flag (default 1=on)
    
    // Mesh initialization settings
    numintervalsG   = 50;   // Initial number of mesh intervals per phase (default=10)
    initcolptsG     = 5;    // Initial number of collocation points per interval
                            // (default=4)
    
    // Mesh refinement settings
    meshRefineTypeG = 1;    // Select mesh refinement technique to be used (default=1)
    minColPtsG      = 4;    // Mininum number of collocation points used in an interval
                            // (default=4)
    maxColPtsG      = 10;    // Maximum number of collocation points used in an interval
                            // (default=10)
    maxMeshIterG    = 1;   // Maximum number of mesh iterations (default=20)
    meshTolG        = 1e-4; // Mesh tolerance (default=1e-7)
    
    // Output save settings
    saveIPOPTFlagG       = 1;   // Save IPOPT solution (default=1)
    saveMeshRefineFlagG  = 0;   // Save mesh refinement history (default=0)
    saveHamiltonianG     = 0;   // Save Hamiltonian values (default=0)
    saveLTIHG            = 0;   // Save linear terms in Hamiltonian values (default=0)
    
    // IPOPT settings
    runIPOPTFlagG   = 1;    // Run IPOPT (default=1)
    NLPtolG         = 1e-9; // NLP Solver tolerance (default=1e-7)
    NLPmaxiterG     = 5000; // Maximum number of iterations allowed for NLP solver
    useLTIHDDG      = 0;    // Indicates usage of bang-bang control detection
    
    /*-----------------------Changes to global parameter settings-----------------------*/
    
    for (int ds=0; ds<numDS; ds++)
    {
        //derivativeSupplierG = ds;
        for (int ni=0; ni<numTestRuns; ni++)
        {
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns] = getDoubleMat(6);
            printf("\nDerivativeSupplier = %d",derivativeSupplierG);
            //numintervalsG += 10;
            //printf("\nNumIntervals = %d",numintervalsG);
            cgpops_go(cgpopsResults, initial_state, final_state);
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[0] = derivativeSupplierG;
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[1] = numintervalsG;
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[2] = cgpopsResults.val[0];
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[3] = cgpopsResults.val[1];
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[4] = cgpopsResults.val[2];
            cgpopsResultsMatMat.mat[ni+ds*numTestRuns].val[5] = cgpopsResults.val[3];

        }

    }
    
    //printf4MSCRIPT("cgpopsResultsMatMat",cgpopsResultsMatMat);

    std::string command = "python3 ../src/lib/cgpops/animation.py ../output/cgpopsIPOPTSolutionBC.m "
        + std::to_string(i_x) + " "
        + std::to_string(i_y) + " "
        + std::to_string(i_z);
    std::system(command.c_str());
    
    return 0;
}


