#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include <toac/pso.h>
#include <cstdlib>
#include <toac/lhs.h>
#include <toac/helper_functions.h>
//#include <cgpops/cgpops_main.hpp>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace casadi;

int main() {

    int status = 0;
    double sol_comparison = 0;
    double time_comparison = 0;

    // Read pre-generated quaternion samples from CSV
    std::ifstream csv_file("../output/lhs_samples.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open lhs_samples.csv" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> initial_states;
    std::vector<std::vector<double>> final_states;

    std::string line;
    std::getline(csv_file, line); // Skip header

    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> initial(7), final(7);
        
        // Read initial state: q0, q1, q2, q3, wx, wy, wz
        for (int i = 0; i < 7; ++i) {
            std::getline(ss, value, ',');
            initial[i] = std::stod(value);
        }
        // Read final state: q0, q1, q2, q3, wx, wy, wz
        for (int i = 0; i < 7; ++i) {
            std::getline(ss, value, ',');
            final[i] = std::stod(value);
        }
        
        initial_states.push_back(initial);
        final_states.push_back(final);
    }

    csv_file.close();
    int iterations = initial_states.size();

    DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps
    extractInitialGuess("../input/initial_guess.csv", X_guess, U_guess, dt_guess);

    DM X_guess_pso(n_states, (n_stp + 1)), U_guess_pso(n_controls, n_stp), dt_guess_pso(n_stp, 1); // Initial guesses for states, controls, and time steps
    PSOOptimizer initial_guess(X_guess_pso, U_guess_pso, dt_guess_pso, false); // Create PSO optimizer instance
    Function solver = get_solver();

    // Open CSV file for logging results
    std::ofstream results_file("../output/mcs_results.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open results CSV file for writing" << std::endl;
        return 1;
    }

    // Write header
    results_file << "T_default,solve_time,T_pso,pso_time,solve_time,"
                << "total_time,sol_comparison,time_comparison,status\n";
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
        try {
            if(!initial_guess.optimize(false)) {
                status = 1;
                if (!initial_guess.optimize(true)) {
                    status = -1;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "PSO Exception: " << e.what() << std::endl;
            status = -2;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double pso_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                        {"X_guess", X_guess}, 
                        {"U_guess", U_guess}, 
                        {"dt_guess", dt_guess}};

        DMDict inputs_pso = {{"X0", X_0}, {"Xf", X_f}, 
                        {"X_guess", X_guess_pso}, 
                        {"U_guess", U_guess_pso}, 
                        {"dt_guess", dt_guess_pso}};
        
        DMDict result, result_pso;

        start = std::chrono::high_resolution_clock::now();
        try {
            result = solver(inputs);
        } catch (const std::exception& e) {
            std::cerr << "Solver Exception: " << e.what() << std::endl;
            result["T"] = -1;

            if (status < 0) {
                status = status * 10 - 3;
            } else {
                status = -3;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double solver_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

        start = std::chrono::high_resolution_clock::now();
        try {
            result_pso = solver(inputs_pso);
        } catch (const std::exception& e) {
            std::cerr << "Solver Exception: " << e.what() << std::endl;
            result_pso["T"] = -1;
            if (status < 0) {
                status = status * 10 - 4;
            } else {
                status = -4;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double solver_pso_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        double total_pso_time = pso_time + solver_pso_time;

        if (result["T"].scalar() < 0 || result_pso["T"].scalar() < 0) {
            sol_comparison = 0;
            time_comparison = 0;
        } else {
            sol_comparison = result["T"].scalar() - result_pso["T"].scalar();
            time_comparison = solver_time - total_pso_time;
        }


        // Log results to CSV
        results_file << result["T"].scalar() << "," << solver_time << "," 
                    << result_pso["T"].scalar() << "," << pso_time << "," 
                    << solver_pso_time << "," << total_pso_time << ","
                    << sol_comparison << "," << time_comparison << "," << status << "\n";

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
        
        // Reset status for next iteration
        status = 0;

    }

    results_file.close();
    std::cout << "Results logged to output/mcs_results.csv" << std::endl;
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    int mins = total_elapsed / 60;
    int secs = total_elapsed % 60;
    std::cout << "Completed in " << mins << " minutes and " << secs << " seconds" << std::endl;    
    return 0;
}