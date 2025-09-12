#include "OptimizedSpacecraftSolver.h"
#include <random>
#include <iostream>
#include <iomanip>

int main() {
    try {
        // Check CUDA availability
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }
        
        std::cout << "CUDA devices available: " << deviceCount << std::endl;
        
        // Constructor called ONCE outside the loop - expensive setup
        std::cout << "Initializing solver..." << std::endl;
        OptimizedSpacecraftSolver solver;
        std::cout << "Setup completed in " << solver.getSetupTime() << " ms" << std::endl;
        
        // Example 1: Single spacecraft solution using convenience function
        std::cout << "\n=== Example 1: Single Spacecraft ===" << std::endl;
        
        // Initial state: [q0, q1, q2, q3, wx, wy, wz]
        std::vector<sunrealtype> single_initial_state = {1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3};
        
        // Control torques: [tau_x, tau_y, tau_z]
        StepParams single_control(0.1, -0.05, 0.2);
        
        sunrealtype delta_t = 0.01;  // 10ms time step
        
        auto single_result = solveSingleSpacecraft(single_initial_state, single_control, delta_t);
        
        if (!single_result.empty()) {
            std::cout << "Initial state: [" << std::setprecision(4);
            for (size_t i = 0; i < single_initial_state.size(); ++i) {
                std::cout << single_initial_state[i];
                if (i < single_initial_state.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "Final state:   [";
            for (size_t i = 0; i < single_result.size(); ++i) {
                std::cout << single_result[i];
                if (i < single_result.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // Check quaternion norm
            sunrealtype quat_norm = sqrt(single_result[0]*single_result[0] + 
                                        single_result[1]*single_result[1] + 
                                        single_result[2]*single_result[2] + 
                                        single_result[3]*single_result[3]);
            std::cout << "Quaternion norm: " << quat_norm << std::endl;
        }
        
        // Example 2: Batch processing - called multiple times in a loop
        std::cout << "\n=== Example 2: Batch Processing Loop ===" << std::endl;
        
        // Random number generator for different scenarios
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> quat_dist(-0.5, 0.5);
        std::uniform_real_distribution<float> omega_dist(-1.0, 1.0);
        std::uniform_real_distribution<float> torque_dist(-0.5, 0.5);
        std::uniform_real_distribution<float> dt_dist(0.005, 0.02);
        
        // Simulate multiple scenarios (this is your repetitive loop)
        int num_scenarios = 5;
        std::vector<double> solve_times;
        
        for (int scenario = 0; scenario < num_scenarios; ++scenario) {
            std::cout << "\nScenario " << (scenario + 1) << ":" << std::endl;
            
            // Generate random initial conditions for all N_STEPS spacecraft
            std::vector<std::vector<sunrealtype>> batch_initial_states(N_STEPS);
            std::vector<StepParams> batch_control_params(N_STEPS);
            
            for (int i = 0; i < N_STEPS; ++i) {
                // Random normalized quaternion
                sunrealtype q0 = 1.0 + quat_dist(gen);
                sunrealtype q1 = quat_dist(gen);
                sunrealtype q2 = quat_dist(gen);
                sunrealtype q3 = quat_dist(gen);
                
                // Normalize quaternion
                sunrealtype norm = sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
                q0 /= norm; q1 /= norm; q2 /= norm; q3 /= norm;
                
                // Random angular velocities
                sunrealtype wx = omega_dist(gen);
                sunrealtype wy = omega_dist(gen);
                sunrealtype wz = omega_dist(gen);
                
                batch_initial_states[i] = {q0, q1, q2, q3, wx, wy, wz};
                
                // Random control torques
                batch_control_params[i] = StepParams(torque_dist(gen), 
                                                    torque_dist(gen), 
                                                    torque_dist(gen));
            }
            
            // Random time step for this scenario
            sunrealtype scenario_dt = dt_dist(gen);
            
            // SOLVE - this is the call that would be in your loop
            double solve_time = solver.solve(batch_initial_states, batch_control_params, scenario_dt);
            
            if (solve_time > 0) {
                solve_times.push_back(solve_time);
                std::cout << "  Time step: " << scenario_dt << " s" << std::endl;
                std::cout << "  Solve time: " << solve_time << " ms" << std::endl;
                std::cout << "  Throughput: " << (N_STEPS / (solve_time / 1000.0)) << " systems/second" << std::endl;
                
                // Get some results for validation
                auto quaternion_norms = solver.getQuaternionNorms();
                double avg_norm = std::accumulate(quaternion_norms.begin(), quaternion_norms.end(), 0.0) / N_STEPS;
                std::cout << "  Average quaternion norm: " << std::setprecision(6) << avg_norm << std::endl;
                
                // Show a sample result
                auto all_solutions = solver.getAllSolutions();
                std::cout << "  Sample result (system 0): [" << std::setprecision(4);
                for (int j = 0; j < N_STATES_PER_SYSTEM; ++j) {
                    std::cout << all_solutions[0][j];
                    if (j < N_STATES_PER_SYSTEM - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            } else {
                std::cout << "  ERROR: Solve failed!" << std::endl;
            }
        }
        
        // Performance summary
        if (!solve_times.empty()) {
            double avg_time = std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
            double min_time = *std::min_element(solve_times.begin(), solve_times.end());
            double max_time = *std::max_element(solve_times.begin(), solve_times.end());
            
            std::cout << "\n=== Performance Summary ===" << std::endl;
            std::cout << "Average solve time: " << avg_time << " ms" << std::endl;
            std::cout << "Min solve time: " << min_time << " ms" << std::endl;
            std::cout << "Max solve time: " << max_time << " ms" << std::endl;
            std::cout << "Average throughput: " << (N_STEPS / (avg_time / 1000.0)) << " systems/second" << std::endl;
        }
        
        // Example 3: Using different time steps
        std::cout << "\n=== Example 3: Different Time Steps ===" << std::endl;
        
        // Create a consistent set of initial conditions
        std::vector<std::vector<sunrealtype>> consistent_initial_states(N_STEPS);
        std::vector<StepParams> consistent_control_params(N_STEPS);
        
        for (int i = 0; i < N_STEPS; ++i) {
            // Simple initial conditions for comparison
            consistent_initial_states[i] = {1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1};
            consistent_control_params[i] = StepParams(0.1, 0.1, 0.1);
        }
        
        std::vector<sunrealtype> time_steps = {0.001, 0.005, 0.01, 0.02, 0.05};
        
        for (auto dt : time_steps) {
            double solve_time = solver.solve(consistent_initial_states, consistent_control_params, dt);
            
            if (solve_time > 0) {
                auto solutions = solver.getAllSolutions();
                std::cout << "dt = " << dt << " s: solve time = " << solve_time 
                          << " ms, final q0 = " << std::setprecision(6) << solutions[0][0] << std::endl;
            }
        }
        
        std::cout << "\n=== Integration Complete ===" << std::endl;
        std::cout << "Note: Constructor/destructor are called outside the loop for optimal performance" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Destructor called automatically when solver goes out of scope
    return 0;
}

/* 
Compilation instr