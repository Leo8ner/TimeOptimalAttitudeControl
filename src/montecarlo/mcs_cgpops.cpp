#include <iostream>
#include <chrono>
#include <cstdlib>
#include <helper_functions.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <unistd.h>

//cgpops
#include "nlpGlobVarDec.hpp"
#include <cgpops/cgpops_main.hpp>
#include <string>
#include <vector>    
    /*-----------------------Changes to global parameter settings-----------------------*/

namespace fs = std::filesystem;


int main() {

    ////// CGPOPS variables
    doubleMat cgpopsResults;
    
    // Derivative supplier settings
    derivativeSupplierG = 1;    // Derivative supplier (default=0)
    scaledG             = 1;    // Scaling flag (default 1=on)
    
    // Mesh initialization settings
    numintervalsG   = 50;   // Initial number of mesh intervals per phase (default=10)
    initcolptsG     = 6;    // Initial number of collocation points per interval
                            // (default=4)
    
    // Mesh refinement settings
    meshRefineTypeG = 1;    // Select mesh refinement technique to be used (default=1)
    minColPtsG      = 4;    // Minimum number of collocation points used in an interval
                            // (default=4)
    maxColPtsG      = 8;   // Maximum number of collocation points used in an interval
                            // (default=10)
    maxMeshIterG    = 1;   // Maximum number of mesh iterations (default=20)
    meshTolG        = 1e-5; // Mesh tolerance (default=1e-7)
    
    // Output save settings
    saveIPOPTFlagG       = 1;   // Save IPOPT solution (default=1)
    saveMeshRefineFlagG  = 0;   // Save mesh refinement history (default=0)
    saveHamiltonianG     = 0;   // Save Hamiltonian values (default=0)
    saveLTIHG            = 0;   // Save linear terms in Hamiltonian values (default=0)
    
    // IPOPT settings
    runIPOPTFlagG   = 1;    // Run IPOPT (default=1)
    NLPtolG         = 1e-7; // NLP Solver tolerance (default=1e-7)
    NLPmaxiterG     = 1000; // Maximum number of iterations allowed for NLP solver
    useLTIHDDG      = 1;    // Indicates usage of bang-bang control detection

        /*-----------------------End of CGPOPS parameter settings-----------------------*/

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, "../output/mcs/lhs_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    // Determine how many rows are already written in output_file (excluding header)
    std::string output_file = "../output/mcs/cgpops.csv";
    bool overwrite = false; // Set to true to overwrite existing file
    int rows_written = 0;
    bool output_exists = fs::exists(output_file);
    if (output_exists && !overwrite) {
        std::ifstream ofcheck(output_file);
        if (ofcheck.is_open()) {
            std::string line;
            // Assume first line is header; read it
            if (std::getline(ofcheck, line)) {
                // count remaining non-empty lines
                while (std::getline(ofcheck, line)) {
                    if (!line.empty()) ++rows_written;
                }
            }
            ofcheck.close();
        }
    }

    // Open results file (use FILE* so we can fflush+fsync after every write)
    FILE* results_fp = nullptr;
    bool need_header = false;
    if (overwrite) {
        results_fp = std::fopen(output_file.c_str(), "w");
        need_header = true;
    } else {
        if (!output_exists || fs::file_size(output_file) == 0) {
            // create and write header
            results_fp = std::fopen(output_file.c_str(), "w");
            need_header = true;
        } else {
            // append to existing file
            results_fp = std::fopen(output_file.c_str(), "a");
            need_header = false;
        }
    }
    if (!results_fp) {
        std::cerr << "Error: Could not open results file for writing: " << output_file << std::endl;
        return 1;
    }

    // Write header if needed
    if (need_header) {
        std::fprintf(results_fp, "T, time, status\n");
        std::fflush(results_fp);
        fsync(fileno(results_fp));
    }

    // Progress tracking
    auto total_start = std::chrono::high_resolution_clock::now();
    int report_interval = std::max(1, iterations / 20); // Report every 5%
    std::cout << "Starting optimization of " << iterations << " samples..." << std::endl;

    for (int i = rows_written; i < iterations; ++i) {

        redirect_output_to_file("../output/solver_logs.log");

        auto start = std::chrono::high_resolution_clock::now();
        cgpops_go(cgpopsResults, initial_states[i], final_states[i]);
        auto end = std::chrono::high_resolution_clock::now();

        double cgpops_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        double T = getCgpopsSolution(derivativeSupplierG);

        restore_output_to_console();
        int solver_status = get_solver_status("../output/cgpops_results.log");

        // Log results to CSV
        if (results_fp) {
            std::fprintf(results_fp, "%.3f,%.3f,%d\n", T, cgpops_time, solver_status);
            std::fflush(results_fp);
            fsync(fileno(results_fp));
        }

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

    if (results_fp) std::fclose(results_fp);
    std::cout << "Results logged to " << output_file << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}