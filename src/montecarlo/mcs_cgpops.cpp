#include <iostream>
#include <chrono>
#include <cstdlib>
#include <helper_functions.h>
#include <fstream>
#include <sstream>
#include <cmath>

//cgpops
#include "nlpGlobVarDec.hpp"
#include <cgpops/cgpops_main.hpp>
#include <string>
#include <vector>    
    /*-----------------------Changes to global parameter settings-----------------------*/

int main() {

    ////// CGPOPS variables
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
    minColPtsG      = 4;    // Minimum number of collocation points used in an interval
                            // (default=4)
    maxColPtsG      = 10;   // Maximum number of collocation points used in an interval
                            // (default=10)
    maxMeshIterG    = 0;   // Maximum number of mesh iterations (default=20)
    meshTolG        = 1e-4; // Mesh tolerance (default=1e-7)
    
    // Output save settings
    saveIPOPTFlagG       = 1;   // Save IPOPT solution (default=1)
    saveMeshRefineFlagG  = 0;   // Save mesh refinement history (default=0)
    saveHamiltonianG     = 0;   // Save Hamiltonian values (default=0)
    saveLTIHG            = 0;   // Save linear terms in Hamiltonian values (default=0)
    
    // IPOPT settings
    runIPOPTFlagG   = 1;    // Run IPOPT (default=1)
    NLPtolG         = 1e-7; // NLP Solver tolerance (default=1e-7)
    NLPmaxiterG     = 10000; // Maximum number of iterations allowed for NLP solver
    useLTIHDDG      = 0;    // Indicates usage of bang-bang control detection

        /*-----------------------End of CGPOPS parameter settings-----------------------*/

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, "../output/mcs/lhs_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    // Open CSV file for logging results
    std::ofstream results_file("../output/mcs/cgpops.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open results CSV file for writing" << std::endl;
        return 1;
    }

    // Write header
    results_file << "T, time, status\n";
    results_file << std::fixed << std::setprecision(3);

    // Progress tracking
    auto total_start = std::chrono::high_resolution_clock::now();
    int report_interval = std::max(1, iterations / 20); // Report every 5%
    std::cout << "Starting optimization of " << iterations << " samples..." << std::endl;

    for (int i = 0; i < iterations; ++i) {

        redirect_output_to_file("../output/solver_logs.log");

        auto start = std::chrono::high_resolution_clock::now();
        cgpops_go(cgpopsResults, initial_states[i], final_states[i]);
        auto end = std::chrono::high_resolution_clock::now();

        double cgpops_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        double T = getCgpopsSolution(derivativeSupplierG);

        restore_output_to_console();
        int solver_status = get_solver_status("../output/cgpops_results.log");

        // Log results to CSV
        results_file << T << "," << cgpops_time << "," << solver_status << "\n";

        // Progress report
        if ((i + 1) % report_interval == 0 || (i + 1) == iterations) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - total_start).count();
            double time_left = (static_cast<double>(elapsed) / (i + 1)) * (iterations - (i + 1));
            double progress = 100.0 * (i + 1) / iterations;
            int mins = elapsed / 60;
            int secs = elapsed % 60;
            int mins_left = time_left / 60;
            int secs_left = fmod(time_left, 60);
            std::cout << "Progress: " << (i + 1) << "/" << iterations
                    << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl
                    << "Elapsed time: " << std::setprecision(1) << mins << " min " << secs << " sec, "
                    << "Estimated time left: " << std::setprecision(1) << mins_left << " min " << secs_left << " sec" << std::endl;
        }

    }

    results_file.close();
    std::cout << "Results logged to output/mcs/cgpops.csv" << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}