#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <iostream>
#include <chrono>
#include <toac/helper_functions.h>
#include <toac/new_pso.h>
#include <cstdlib>

using namespace casadi;

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
        return 1;
    }
    
    try {

        auto start = std::chrono::high_resolution_clock::now();

        // Parse command line arguments
        DM X_0, X_f, angles_0, angles_f;
        std::tie(X_0, X_f, angles_0, angles_f) = parseInput(argv[1], argv[2]);

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

        // DMDict PSOresults = {{"X", X_guess}, {"U", U_guess}, {"T", sum(dt_guess)}, {"dt", dt_guess}};
        // processResults(PSOresults, angles_0, angles_f);

        Function solver = get_solver();
        
        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                         {"X_guess", X_guess}, 
                         {"U_guess", U_guess}, 
                         {"dt_guess", dt_guess}};

        auto start_solver = std::chrono::high_resolution_clock::now();
        DMDict result = solver(inputs);

        // Stop the timer
        end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        auto solver_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_solver) / 1000.0;
        std::cout << "PSO Time: " << elapsed_pso.count() << " s" << std::endl;
        std::cout << "Total PSO + Setup Time: " << total_pso.count() << " s" << std::endl;
        std::cout << "Solver Time: " << solver_elapsed.count() << " s" << std::endl;
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;

        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;

        DMDict PSOresults = {{"X", X_guess}, {"U", U_guess}, {"T", sum(dt_guess)}, {"dt", dt_guess}};
        processResults(PSOresults, angles_0, angles_f);

        processResults(result, angles_0, angles_f);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}