#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <helper_functions.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

using namespace casadi;
    
    /*-----------------------Changes to global parameter settings-----------------------*/

int main() {

    // Load pre-generated quaternion samples from CSV
    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;
    
    if (!loadStateSamples(initial_states, final_states, "../output/lhs_samples.csv")) {
        return 1;
    }
    int iterations = initial_states.size();

    DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps
    extractInitialGuess("../input/initial_guess.csv", X_guess, U_guess, dt_guess);
    Function solver = get_solver();

    // Open CSV file for logging results
    std::ofstream results_file("../output/mcs/no_pso.csv");
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
        DM X_0 = DM::vertcat({
        initial_states[i][0], initial_states[i][1], initial_states[i][2], initial_states[i][3],
        initial_states[i][4], initial_states[i][5], initial_states[i][6]
        });
        DM X_f = DM::vertcat({
        final_states[i][0], final_states[i][1], final_states[i][2], final_states[i][3],
        final_states[i][4], final_states[i][5], final_states[i][6]
        });
        
        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                        {"X_guess", X_guess}, 
                        {"U_guess", U_guess}, 
                        {"dt_guess", dt_guess}};

        redirect_output_to_file("../output/solver_logs.log");
        auto start = std::chrono::high_resolution_clock::now();
        DMDict result = solver(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        double solver_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        restore_output_to_console();

        int solver_status = get_solver_status("../output/solver_logs.log");

        // Log results to CSV
        results_file << result["T"].scalar() << "," << solver_time << "," << solver_status << "\n";

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
    std::cout << "Results logged to output/mcs/no_pso.csv" << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}