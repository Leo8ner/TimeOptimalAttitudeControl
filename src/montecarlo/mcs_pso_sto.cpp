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

using namespace casadi;

int main() {

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, "../output/mcs/lhs_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    // PSO parameters
    int n_particles = 3200;       // Number of particles in swarm
    int n_iterations = 200;       // Number of PSO iterations
    double inertia_weight = 2.0;  // Inertia weight
    double cognitive_coeff = 3.0; // Cognitive coefficient
    double social_coeff = 1.0;    // Social coefficient
    bool decay_inertia = true;    // Enable inertia weight decay
    bool decay_cognitive = true;  // Enable cognitive coefficient decay
    bool decay_social = true;     // Enable social coefficient decay
    double min_inertia = 0.1;     // Minimum inertia weight
    double min_cognitive = 0.5;   // Minimum cognitive coefficient
    double min_social = 0.2;      // Minimum social coefficient
    double sigmoid_alpha = 1.0;  // Sigmoid alpha for stochastic control sign
    double sigmoid_saturation = 0.99; // Sigmoid saturation limit for control sign

    DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps
    PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, PSOMethod::STO, false, n_particles); // Create PSO optimizer instance
    initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
                                  decay_inertia, decay_cognitive, decay_social,
                                  min_inertia, min_cognitive, min_social);  
    Function solver = get_solver();

    // Open CSV file for logging results
    std::ofstream results_file("../output/mcs/pso_sto.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open results CSV file for writing" << std::endl;
        return 1;
    }

    // Write header
    results_file << "T, time, time_on_pso, time_on_solver, status\n";
    results_file << std::fixed << std::setprecision(3);

    // Progress tracking
    auto total_start = std::chrono::high_resolution_clock::now();
    int report_interval = std::max(1, iterations / 20); // Report every 5%
    std::cout << "Starting optimization of " << iterations << " samples..." << std::endl;

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

        // Log results to CSV
        results_file << result["T"].scalar() << "," << total_time << "," 
                    << pso_time << "," << solver_time << "," << solver_status << "\n";

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
    std::cout << "Results logged to output/mcs/pso_sto.csv" << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}