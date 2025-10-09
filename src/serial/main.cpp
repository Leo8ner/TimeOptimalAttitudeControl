#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <iostream>
#include <chrono>
#include <toac/helper_functions.h>
#include <cstdlib>
#include <toac/pso.h>

using namespace casadi;

// Modified main function
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\" all in degrees" << std::endl;
        return 1;
    }
    
    try {

        // Start the timer
        // This is used to measure the time taken by the optimization process
        auto start = std::chrono::high_resolution_clock::now();
        
        // Parse command line arguments
        DM X_0, X_f, angles_0, angles_f;
        std::tie(X_0, X_f, angles_0, angles_f) = parseInput(argv[1], argv[2]);

        // DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps

        // extractInitialGuess("../input/initial_guess.csv", X_guess, U_guess, dt_guess);

        DM X_guess(n_states, (n_stp + 1)), U_guess(n_controls, n_stp), dt_guess(n_stp, 1); // Initial guesses for states, controls, and time steps

        auto prepare_pso = std::chrono::high_resolution_clock::now();
        PSOOptimizer initial_guess(X_guess, U_guess, dt_guess, true); // Create PSO optimizer instance
        initial_guess.setStates(X_0->data(), X_f->data());
        // double w = 5.0; // Inertia weight
        // double c1 = 2.0; // Cognitive weight
        // double c2 = 1.0; // Social weight

        // initial_guess.setPSOParameters(100,w,c1,c2); // Set PSO parameters: iterations, particles, inertia, cognitive, social
        auto start_pso = std::chrono::high_resolution_clock::now();
        if(!initial_guess.optimize(true)) {
            std::cerr << "Error: PSO initial guess optimization failed." << std::endl;
            return -1;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_pso = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_pso) / 1000.0;
        auto total_pso = std::chrono::duration_cast<std::chrono::milliseconds>(end - prepare_pso) / 1000.0;


        std::string plugin = "ipopt"; // Specify the solver plugin to use
        std::string method = "collocation"; // Specify the integration method to use
        bool fixed_step = true; // Use fixed step size for the integrator

        // Dynamics
        Dynamics dyn(plugin, method); // Create an instance of the Dynamics class

        // Solver
        Optimizer opti(dyn, fixed_step);     // Create an instance of the Optimizer class

        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                         {"X_guess", X_guess}, 
                         {"U_guess", U_guess}, 
                         {"dt_guess", dt_guess}};
                         
        DMDict result = opti.solver(inputs);
        
        // Stop the timer
        end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "PSO Time: " << elapsed_pso.count() << " s" << std::endl;
        std::cout << "Total PSO + Setup Time: " << total_pso.count() << " s" << std::endl;
        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;

        processResults(result, angles_0, angles_f);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}