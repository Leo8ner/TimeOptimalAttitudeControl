#include <toac/cuda_optimizer.h>
#include <toac/helper_functions.h>

using namespace casadi;

// Modified main function
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
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
        std::string csv_data = "../input/initial_guess.csv"; // Path to the CSV file for initial guess
        extractInitialGuess(csv_data, X_guess, U_guess, dt_guess);

        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_parsolver.so";

        // use this function
        Function solver = external("parsolver", lib_full_name);

        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}, 
                         {"X_guess", X_guess}, 
                         {"U_guess", U_guess}, 
                         {"dt_guess", dt_guess}};
        DMDict result = solver(inputs);
        
        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;

        processResults(result, angles_0, angles_f);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}