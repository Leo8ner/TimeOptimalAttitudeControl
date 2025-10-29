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

using namespace casadi;

int main(int argc, char** argv) {
    // PSO parameter tuning

    int file_index;
    PSOMethod pso_method;
    std::string method_string;

    // Parse arguments:
    // usage: prog [pso method] [file_index]
    if (argc > 1) {
        method_string = argv[1];
        if (method_string == "sto") {
            pso_method = PSOMethod::STO;
        } else if (method_string == "full") {
            pso_method = PSOMethod::FULL;
        } else {
            std::cerr << "Error: Unknown PSO method '" << method_string << "'. Use 'sto' or 'full'." << std::endl;
            return 1;
        }
    }

    if (argc > 2) {
        file_index = std::atoi(argv[2]);
        if (file_index <= 0) {
            std::cerr << "Error: file_index must be a positive integer." << std::endl;
            return 1;
        }
    } else {
        std::cout << "Correct usage: " << argv[0] << " [pso method (sto|full)] [file_index]\n";
        return 1;
    }

    // Load PSO parameter samples from CSV
    std::vector<std::vector<double>> pso_params;
    std::string params_file = "../output/pso_params/lhs_pso_params_samples_" + std::to_string(file_index) + ".csv";
    if (!loadPSOSamples(pso_params, params_file)) {
        return 1;
    }
    int tuning_iterations = pso_params.size();

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;

    if (!loadStateSamples(initial_states, final_states, "../output/pso_params/lhs_pso_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    Function solver = get_solver();

    // Open CSV file for logging results
    std::string output_file = "../output/pso_params/pso_" + method_string + "_tuning_" + std::to_string(file_index) + ".csv";
    std::ofstream results_file(output_file);
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open results CSV file for writing" << std::endl;
        return 1;
    }

    // Write header
    results_file << "avg_time,avg_time_on_pso,avg_time_on_solver,n_bad_status\n";
    results_file << std::fixed << std::setprecision(3);

    // Progress tracking
    auto total_start = std::chrono::high_resolution_clock::now();
    int report_interval = std::max(1, tuning_iterations / 20); // Report every 5%
    std::cout << "Starting optimization of " << tuning_iterations << " samples..." << std::endl;
    int sample_count = 0;
    for (const auto& sample : pso_params) {
        int n_particles = static_cast<int>(sample[0]);
        int n_iterations = static_cast<int>(sample[1]);
        double inertia_weight = sample[2];
        double cognitive_coeff = sample[3];
        double social_coeff = sample[4];
        double min_inertia = sample[5];
        double min_cognitive = sample[6];
        double min_social = sample[7];
        double sigmoid_alpha = sample[8];
        double sigmoid_saturation = sample[9];
        bool decay_inertia = true;
        bool decay_cognitive = true;
        bool decay_social = true;

        DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps


        PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, pso_method, false, n_particles); // Create PSO optimizer instance
        initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
                                    decay_inertia, decay_cognitive, decay_social,
                                    min_inertia, min_cognitive, min_social, sigmoid_alpha, sigmoid_saturation);  


        double total_time_accum = 0.0;
        double pso_time_accum = 0.0;
        double solver_time_accum = 0.0;
        int n_bad_status = 0;
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
            if (solver_status < 0) {
                n_bad_status++;
            }

            total_time_accum += total_time;
            pso_time_accum += pso_time;
            solver_time_accum += solver_time;

        }
        // Log results to CSV
        results_file << (total_time_accum / iterations) << ","
                     << (pso_time_accum / iterations) << ","
                     << (solver_time_accum / iterations) << ","
                     << n_bad_status << "\n";

        // Progress report
        if ((sample_count + 1) % report_interval == 0 || (sample_count + 1) == tuning_iterations) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - total_start).count();
            double time_left = (static_cast<double>(elapsed) / (sample_count + 1)) * (tuning_iterations - (sample_count + 1));
            double progress = 100.0 * (sample_count + 1) / tuning_iterations;
            int minutes = elapsed / 60;
            int secs = elapsed % 60;
            int mins_left = time_left / 60;
            int secs_left = fmod(time_left, 60);
            std::cout << "Progress: " << (sample_count + 1) << "/" << tuning_iterations
                    << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl
                    << "Elapsed time: " << std::setprecision(1) << minutes << " min " << secs << " sec, "
                    << "Estimated time left: " << std::setprecision(1) << mins_left << " min " << secs_left << " sec" << std::endl;
        }
        sample_count++;

        }

    results_file.close();
    std::cout << "Results logged to " << output_file << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int minutes = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << minutes << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}