#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include <toac/pso.h>
#include <cstdlib>
#include <helper_functions.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>
#include <toac/lhs.h>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <cstdio>
#include <unistd.h>

namespace fs = std::filesystem;

using namespace casadi;

int main(int argc, char** argv) {
    int start_index = 0;
    bool overwrite = false;
    std::string select_method = "all"; // "sto", "full", or "all"

    // Parse command line:
    // ./pso_tuner [start_index=0] [overwrite=0] [method=all|sto|full]
    if (argc > 1) {
        start_index = std::atoi(argv[1]);
        if (start_index < 0) {
            std::cerr << "Warning: start_index must be non-negative. Using 0." << std::endl;
            start_index = 0;
        }
    }
    if (argc > 2) {
        std::string ov = argv[2];
        std::transform(ov.begin(), ov.end(), ov.begin(), ::tolower);
        overwrite = (ov == "1" || ov == "true" || ov == "yes");
    }
    if (argc > 3) {
        select_method = argv[3];
        std::transform(select_method.begin(), select_method.end(), select_method.begin(), ::tolower);
        if (select_method != "sto" && select_method != "full" && select_method != "all") {
            std::cerr << "Warning: unknown method '" << argv[3] << "'. Using 'all'.\n";
            return 1;
        }
    }

    std::string params_dir = "../output/pso_params/";
    if (!fs::exists(params_dir)) {
        std::cerr << "Error: directory " << params_dir << " does not exist\n";
        return 1;
    }

    // discover files and their method/index
    std::regex re(R"(lhs_pso_params_(sto|full)_samples_([0-9]+)\.csv)", std::regex::icase);
    struct FileEntry { fs::path path; std::string method; int index; };
    std::vector<FileEntry> entries;

    for (const auto &entry : fs::directory_iterator(params_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        std::smatch m;
        if (std::regex_match(fname, m, re)) {
            std::string method = m[1].str();
            std::string idxs = m[2].str();
            int idx = std::stoi(idxs);
            std::transform(method.begin(), method.end(), method.begin(), ::tolower);
            // respect select_method filter
            if (select_method == "all" || select_method == method) {
                entries.push_back({ entry.path(), method, idx });
            }
        }
    }

    if (entries.empty()) {
        std::cerr << "Error: No parameter files matching lhs_pso_params_{sto|full}_samples_n.csv found in "
                  << params_dir << " for method='" << select_method << "'\n";
        return 1;
    }

    // sort by numeric index ascending
    std::sort(entries.begin(), entries.end(), [](const FileEntry &a, const FileEntry &b){
        return a.index < b.index;
    });

    std::cout << "Found " << entries.size() << " parameter files (method filter='" << select_method
              << "'). Starting at index " << start_index << ". Overwrite: " << (overwrite ? "yes" : "no") << "\n";

    // Load pre-generated quaternion samples (done once)
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    if (!loadStateSamples(initial_states, final_states, "../output/pso_params/lhs_pso_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    // Load GPOPS results
    std::vector<double> gpops_T;
    std::vector<int> gpops_status;
    {
        std::ifstream gpops_file("../output/mcs/cgpops.csv");
        if (!gpops_file.is_open()) {
            std::cerr << "Error: Could not open GPOPS results file\n";
            return 1;
        }

        std::string line;
        // Skip header
        std::getline(gpops_file, line);
        
        while (std::getline(gpops_file, line)) {
            std::stringstream ss(line);
            std::string token;
            
            // Read T
            std::getline(ss, token, ',');
            gpops_T.push_back(std::stod(token));
            
            // Skip time
            std::getline(ss, token, ',');
            
            // Read status
            std::getline(ss, token, ',');
            gpops_status.push_back(std::stoi(token));
        }
    }

    if (gpops_T.size() != initial_states.size()) {
        std::cerr << "Warning: GPOPS results size (" << gpops_T.size() 
                  << ") doesn't match states samples size (" << initial_states.size() << ")\n";
        return 1;
    }

    Function solver = get_solver();

    // Process each discovered file whose index >= start_index
    for (const auto &fe : entries) {
        if (fe.index < start_index) continue;

        std::string method_string = fe.method; // "sto" or "full"
        PSOMethod pso_method;
        if (method_string == "sto") pso_method = PSOMethod::STO;
        else pso_method = PSOMethod::FULL;

        std::string params_file = fe.path.string();
        std::string output_file = params_dir + "pso_" + method_string + "_tuning_" + std::to_string(fe.index) + ".csv";

        // Check if input file exists (should, since discovered) but be safe
        if (!fs::exists(params_file)) {
            std::cout << "Skipping index " << fe.index << ": parameter file not found: " << params_file << std::endl;
            continue;
        }

        // Load PSO parameter samples
        std::vector<std::vector<double>> pso_params;
        if (!loadPSOSamples(pso_params, params_file)) {
            std::cout << "Error loading parameter file " << params_file << ", skipping" << std::endl;
            continue;
        }
        int tuning_iterations = static_cast<int>(pso_params.size());

        // Determine how many rows are already written in output_file (excluding header)
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

        // If output already fully written, skip file
        if (!overwrite && rows_written >= tuning_iterations && tuning_iterations > 0) {
            std::cout << "Skipping index " << fe.index << ": output already complete (" << rows_written << "/" << tuning_iterations << ")\n";
            continue;
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
            continue;
        }

        // Write header if needed
        if (need_header) {
            std::fprintf(results_fp, "avg_time,avg_time_on_pso,avg_time_on_solver,n_bad_status,n_runs\n");
            std::fflush(results_fp);
            fsync(fileno(results_fp));
        }

        std::cout << "\nProcessing parameter file " << fe.path.filename().string() << " (method=" << method_string
                  << ", index=" << fe.index << ")\n";

        // Progress tracking for this file
        auto file_start = std::chrono::high_resolution_clock::now();
        int report_interval = std::max(1, tuning_iterations / 20);
        std::cout << "Starting optimization of " << tuning_iterations << " parameter sets..." << std::endl;
        int sample_count = rows_written +1;

        for (int sample_idx = rows_written; sample_idx < tuning_iterations; ++sample_idx) {
            const auto &sample = pso_params[sample_idx];
            bool is_sto = (pso_method == PSOMethod::STO);

            int n_particles = 0;
            int n_iterations = 0;
            double inertia_weight = 0.0;
            double cognitive_coeff = 0.0;
            double social_coeff = 0.0;
            double min_inertia = 0.0;
            double min_cognitive = 0.0;
            double min_social = 0.0;
            double sigmoid_alpha = 0.0;
            double sigmoid_saturation = 0.0;

            // safely extract columns (some files may omit alpha/saturation)
            if (sample.size() >= 2) {
                n_particles = static_cast<int>(sample[0]);
                n_iterations = static_cast<int>(sample[1]);
            }
            if (sample.size() > 2) inertia_weight = sample[2];
            if (sample.size() > 3) cognitive_coeff = sample[3];
            if (sample.size() > 4) social_coeff = sample[4];
            if (sample.size() > 5) min_inertia = sample[5];
            if (sample.size() > 6) min_cognitive = sample[6];
            if (sample.size() > 7) min_social = sample[7];
            if (is_sto) {
                if (sample.size() > 8) sigmoid_alpha = sample[8];
                if (sample.size() > 9) sigmoid_saturation = sample[9];
            }

            bool decay_inertia = true;
            bool decay_cognitive = true;
            bool decay_social = true;

            // Prepare initial guesses (sizes used must match solver expectations)
            DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1);

            PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, pso_method, false, n_particles);
            // pass alpha/saturation even for FULL (they'll be ignored by PSO implementation if unused)
            initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
                                        decay_inertia, decay_cognitive, decay_social,
                                        min_inertia, min_cognitive, min_social, sigmoid_alpha, sigmoid_saturation);

            double total_time_accum = 0.0;
            double pso_time_accum = 0.0;
            double solver_time_accum = 0.0;
            int n_bad_status = 0;
            double min_time = 0.2;
            int min_bad_status = 100;
            int n_runs = iterations;

            for (int i = 0; i < iterations; ++i) {
                DM X_0 = DM::vertcat({
                    initial_states[i][0], initial_states[i][1], initial_states[i][2], initial_states[i][3],
                    initial_states[i][4], initial_states[i][5], initial_states[i][6]
                });
                DM X_f = DM::vertcat({
                    final_states[i][0], final_states[i][1], final_states[i][2], final_states[i][3],
                    final_states[i][4], final_states[i][5], final_states[i][6]
                });

                // Start the timer
                auto start = std::chrono::high_resolution_clock::now();
                initial_guess.setStates(X_0->data(), X_f->data());
                initial_guess.optimize(false);
                auto end = std::chrono::high_resolution_clock::now();
                double pso_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

                // Call the solver with parsed inputs
                DMDict inputs = {{"X0", X_0}, {"Xf", X_f},
                                {"X_guess", X_guess},
                                {"U_guess", U_guess},
                                {"dt_guess", dt_guess}};

                redirect_output_to_file("../output/solver_logs.log");
                start = std::chrono::high_resolution_clock::now();
                DMDict result = solver(inputs);
                end = std::chrono::high_resolution_clock::now();
                double solver_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

                double total_time = pso_time + solver_time;

                restore_output_to_console();

                int solver_status = get_solver_status("../output/solver_logs.log");
                if (solver_status < 0 && abs(result["T"].scalar() - gpops_T[i]) > 1e-1) {
                    n_bad_status++;
                }

                total_time_accum += total_time;
                pso_time_accum += pso_time;
                solver_time_accum += solver_time;
                double min_total_time = (total_time_accum + (iterations - (i+1)) * pso_time_accum/(i+1))/iterations;

                if ((min_total_time > min_time && n_bad_status > min_bad_status) || (n_bad_status > 100) || (min_total_time > 0.2)) {
                    // early stopping
                    n_runs = i + 1;
                    break;
                }

            }

            // Immediately write this sample's result to file (flush+fsync)
            double avg_total = (total_time_accum / n_runs);
            double avg_pso = (pso_time_accum / n_runs);
            double avg_solver = (solver_time_accum / n_runs);

            min_time = std::min(min_time, avg_total);
            min_bad_status = std::min(min_bad_status, n_bad_status);

            if (results_fp) {
                std::fprintf(results_fp, "%.3f,%.3f,%.3f,%d,%d\n", avg_total, avg_pso, avg_solver, n_bad_status, n_runs);
                std::fflush(results_fp);
                fsync(fileno(results_fp));
            }

            // Progress report
            if ((sample_count) % report_interval == 0 || (sample_count) == tuning_iterations) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - file_start).count();
                double time_left = (static_cast<double>(elapsed) / (sample_count)) * (tuning_iterations - (sample_count));
                double progress = 100.0 * (sample_count) / tuning_iterations;
                int minutes = elapsed / 60;
                int secs = elapsed % 60;
                int mins_left = time_left / 60;
                int secs_left = static_cast<int>(fmod(time_left, 60.0));
                std::cout << "Progress: " << (sample_count) << "/" << tuning_iterations
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl
                          << "Elapsed time: " << std::setprecision(1) << minutes << " min " << secs << " sec, "
                          << "Estimated time left: " << std::setprecision(1) << mins_left << " min " << secs_left << " sec" << std::endl;
            }
            sample_count++;
        }


        if (results_fp) std::fclose(results_fp);
        std::cout << "Results logged to " << output_file << std::endl;

        auto file_end = std::chrono::high_resolution_clock::now();
        auto file_elapsed = std::chrono::duration_cast<std::chrono::seconds>(file_end - file_start).count();
        int minutes = file_elapsed / 60;
        int secs = file_elapsed % 60;
        std::cout << "File " << fe.index << " completed in " << minutes
                  << " minutes and " << secs << " seconds" << std::endl;
    }

    std::cout << "\nAll files processed." << std::endl;
    return 0;
}