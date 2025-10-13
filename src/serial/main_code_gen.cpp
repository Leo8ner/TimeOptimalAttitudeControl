#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <iostream>
#include <chrono>
#include <toac/helper_functions.h>
#include <cstdlib>

using namespace casadi;

// main function
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

        DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps
        std::string csv_data = "../input/initial_guess.csv"; // Path to the CSV file for initial guess
        extractInitialGuess(csv_data, X_guess, U_guess, dt_guess);

        // Start the timer
        Function solver = get_solver();
        
        
        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                         {"X_guess", X_guess}, 
                         {"U_guess", U_guess}, 
                         {"dt_guess", dt_guess}};
        redirect_fatrop_to_file("../output/fatropINFO.txt");
        DMDict result;
        {
        result = solver(inputs);
        }
        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        restore_fatrop_to_console();
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;

        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;

        processResults(result, angles_0, angles_f);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}