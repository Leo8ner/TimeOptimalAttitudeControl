#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include <helper_functions.h>
#include <toac/pso.h>
#include <cstdlib>

using namespace casadi;

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
        return 1;
    }
    
    try {

        // Parse command line arguments
        DM X_0, X_f, angles_0, angles_f;
        std::tie(X_0, X_f, angles_0, angles_f) = parseInput(argv[1], argv[2]);

        // PSO parameters
        int n_particles = 5120;       // Number of particles in swarm
        int n_iterations = 500;       // Number of PSO iterations
        double inertia_weight = 7;  // Inertia weight
        double cognitive_coeff = 5; // Cognitive coefficient
        double social_coeff = 3;    // Social coefficient
        bool decay_inertia = true;    // Enable inertia weight decay
        bool decay_cognitive = true;  // Enable cognitive coefficient decay
        bool decay_social = true;     // Enable social coefficient decay
        double min_inertia = 0.4;     // Minimum inertia weight
        double min_cognitive = 0.8;   // Minimum cognitive coefficient
        double min_social = 1.0;      // Minimum social coefficient
        double sigmoid_alpha = 8.0;  // Sigmoid alpha for stochastic control sign
        double sigmoid_saturation = 0.85; // Sigmoid saturation limit for control sign

        DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps

        auto prepare_pso = std::chrono::high_resolution_clock::now();
        PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, PSOMethod::FULL, true, n_particles); // Create PSO optimizer instance
        // initial_guess.setPSOParameters(n_iterations, inertia_weight, cognitive_coeff, social_coeff,
        //                             decay_inertia, decay_cognitive, decay_social,
        //                             min_inertia, min_cognitive, min_social, sigmoid_alpha, sigmoid_saturation);  
        initial_guess.setStates(X_0->data(), X_f->data());
        auto start_pso = std::chrono::high_resolution_clock::now();
        if(!initial_guess.optimize(true)) {
            std::cerr << "Error: PSO initial guess optimization failed." << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_pso = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_pso) / 1000.0;
        auto total_pso = std::chrono::duration_cast<std::chrono::milliseconds>(end - prepare_pso) / 1000.0;

        DMDict result = {{"X", X_guess}, {"U", U_guess}, {"T", sum(dt_guess)}, {"dt", dt_guess}};
        processResults(result, angles_0, angles_f);

        // Stop the timer
        std::cout << "PSO Time: " << elapsed_pso.count() << " s" << std::endl;
        std::cout << "Total PSO + Setup Time: " << total_pso.count() << " s" << std::endl;
        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}