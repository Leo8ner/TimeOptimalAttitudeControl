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
    std::vector<double> initial_states = {1,0,0,0,0,0,0};
    std::vector<double> final_states = {0,1,0,0,0,0,0};


    // PSO parameters
    int n_particles = 0;        // Number of particles in swarm
    int n_iterations = 0;       // Number of PSO iterations
    double inertia_weight = 2.0;  // Inertia weight
    double cognitive_coeff = 3.0; // Cognitive coefficient
    double social_coeff = 1.0;    // Social coefficient
    bool decay_inertia = true;    // Enable inertia weight decay
    bool decay_cognitive = true;  // Enable cognitive coefficient decay
    bool decay_social = true;     // Enable social coefficient decay
    double min_inertia = 0.1;     // Minimum inertia weight
    double min_cognitive = 0.5;   // Minimum cognitive coefficient
    double min_social = 0.2;      // Minimum social coefficient

    DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps
    DM X_0 = DM::vertcat({
        initial_states[0], initial_states[1], initial_states[2], initial_states[3],
        initial_states[4], initial_states[5], initial_states[6]
        });
    DM X_f = DM::vertcat({
        final_states[0], final_states[1], final_states[2], final_states[3],
        final_states[4], final_states[5], final_states[6]
        });

    double time = 0.0;
    double time_max = 0.1; // seconds

    while (true) {
        time = 0.0;
        n_particles += 640;
        PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, PSOMethod::FULL, false, n_particles); // Create PSO optimizer instance

        while (time < time_max) {
            n_iterations += 10;
            initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
                                        decay_inertia, decay_cognitive, decay_social,
                                        min_inertia, min_cognitive, min_social);  

            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < 10; ++j) {
                initial_guess.setStates(X_0->data(), X_f->data());
                initial_guess.optimize(false);
            }
            auto end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10000.0;
        }
        if (n_iterations == 10) {
            break;
        }
        std::cout << "Particles: " << n_particles << ", Iterations: " << n_iterations << ", Time: " << time << " s\n";
        n_iterations = 0;
    }
    return 0;
}