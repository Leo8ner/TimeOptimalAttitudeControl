#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <iostream>
#include <chrono>
#include <toac/helper_functions.h>
#include <cstdlib>

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

        DM X_0, angles_0;
        std::tie(X_0, angles_0) = parseStateVector(argv[1]);
        DM X_f, angles_f;
        std::tie(X_f, angles_f) = parseStateVector(argv[2]);

        DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps

        extractInitialGuess("../output/initial_guess.csv", X_guess, U_guess, dt_guess);

        std::string plugin = "fatrop"; // Specify the solver plugin to use
        bool fixed_step = true; // Use fixed step size for the integrator

        // Dynamics
        Dynamics dyn(plugin); // Create an instance of the Dynamics class

        // Solver
        Optimizer opti(dyn.F, plugin, fixed_step);     // Create an instance of the Optimizer class

        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                         {"X_guess", X_guess}, 
                         {"U_guess", U_guess}, 
                         {"dt_guess", dt_guess}};
                         
        DMDict result = opti.solver(inputs);
        
        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;

        processResults(result, angles_0, angles_f);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}