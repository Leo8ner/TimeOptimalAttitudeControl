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

int main() {

    int tuning_iterations = 1000;

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, "../output/mcs/lhs_pso_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    // PSO parameters
    int n_cores = 640;
    double min_particles = 1.0;       // Minimum number of particles in swarm
    double max_particles = 10.0;       // Maximum number of particles in swarm
    double max_iterations = 500.0;       // Number of PSO iterations
    double min_iterations = 50.0;       // Minimum number of PSO iterations
    double min_inertia_weight = 0.1;  // Inertia weight
    double max_inertia_weight = 10.0;  // Inertia weight
    double min_cognitive_coeff = 0.1; // Cognitive coefficient
    double max_cognitive_coeff = 10.0; // Cognitive coefficient
    double min_social_coeff = 0.1;    // Social coefficient
    double max_social_coeff = 10.0;    // Social coefficient
    bool decay_inertia = true;    // Enable inertia weight decay
    bool decay_cognitive = true;  // Enable cognitive coefficient decay
    bool decay_social = true;     // Enable social coefficient decay
    double min_min_inertia = 0.0;     // Minimum inertia weight
    double min_min_cognitive = 0.0;   // Minimum cognitive coefficient
    double min_min_social = 0.0;      // Minimum social coefficient
    double max_min_inertia = 10.0;     // Minimum inertia weight
    double max_min_cognitive = 10.0;   // Minimum cognitive coefficient
    double max_min_social = 10.0;      // Minimum social coefficient
    double max_sigmoid_alpha = 10.0;  // Sigmoid alpha for stochastic control sign
    double min_sigmoid_alpha = 0.1;  // Sigmoid alpha for stochastic control sign
    double min_sigmoid_saturation = 0.5; // Minimum sigmoid saturation limit for control sign
    double max_sigmoid_saturation = 1.0; // Maximum sigmoid saturation limit for control sign

    LHS lhs(tuning_iterations, 10);
    
    std::vector<double> mins = {
        min_particles, min_iterations, min_inertia_weight, min_cognitive_coeff, min_social_coeff,
        min_min_inertia, min_min_cognitive, min_min_social, min_sigmoid_alpha, min_sigmoid_saturation
    };
    std::vector<double> maxs = {
        max_particles, max_iterations, max_inertia_weight, max_cognitive_coeff, max_social_coeff,
        max_min_inertia, max_min_cognitive, max_min_social, max_sigmoid_alpha, max_sigmoid_saturation
    };

    auto samples = lhs.sampleBounded(mins, maxs);

    Function solver = get_solver();

    // Open CSV file for logging results
    std::ofstream results_file("../output/mcs/pso_sto_tuning.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open results CSV file for writing" << std::endl;
        return 1;
    }

    // Write header
    results_file << "avg_time, avg_time_on_pso, avg_time_on_solver, n_bad_status\n";
    results_file << std::fixed << std::setprecision(3);

    // Progress tracking
    auto total_start = std::chrono::high_resolution_clock::now();
    int report_interval = std::max(1, tuning_iterations / 20); // Report every 5%
    std::cout << "Starting optimization of " << tuning_iterations << " samples..." << std::endl;
    int sample_count = 0;
    for (const auto& sample : samples) {
        int n_particles = static_cast<int>(std::round(sample[0] * n_cores));
        int n_iterations = static_cast<int>(std::round(sample[1]));
        double inertia_weight = sample[2];
        double cognitive_coeff = sample[3];
        double social_coeff = sample[4];
        double min_inertia = sample[5];
        double min_cognitive = sample[6];
        double min_social = sample[7];
        double sigmoid_alpha = sample[8];
        double sigmoid_saturation = sample[9];

        if (min_social > social_coeff || min_cognitive > cognitive_coeff || min_inertia > inertia_weight) {
            continue;
        }

        DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps
        PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, PSOMethod::STO, false, n_particles); // Create PSO optimizer instance
        initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
                                    decay_inertia, decay_cognitive, decay_social,
                                    min_inertia, min_cognitive, min_social);  

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
        results_file << (total_time_accum / iterations) << ", "
                     << (pso_time_accum / iterations) << ", "
                     << (solver_time_accum / iterations) << ", "
                     << n_bad_status << "\n";

        // Progress report
        if ((sample_count + 1) % report_interval == 0 || (sample_count + 1) == iterations) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - total_start).count();
            double time_left = (static_cast<double>(elapsed) / (sample_count + 1)) * (iterations - (sample_count + 1));
            double progress = 100.0 * (sample_count + 1) / iterations;
            int mins = elapsed / 60;
            int secs = elapsed % 60;
            int mins_left = time_left / 60;
            int secs_left = fmod(time_left, 60);
            std::cout << "Progress: " << (sample_count + 1) << "/" << iterations
                    << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl
                    << "Elapsed time: " << std::setprecision(1) << mins << " min " << secs << " sec, "
                    << "Estimated time left: " << std::setprecision(1) << mins_left << " min " << secs_left << " sec" << std::endl;
        }
        sample_count++;

        }

    results_file.close();
    std::cout << "Results logged to output/mcs/pso_sto_tuning.csv" << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}