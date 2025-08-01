#include <casadi/casadi.hpp>
#include <toac/cuda_optimizer.h>
#include <toac/constraints.h>
#include <iostream>
#include <chrono>
#include <toac/plots.h>
#include <cstdlib>

using namespace casadi;

int main() {

    // Start the timer
    // This is used to measure the time taken by the optimization process
    auto start = std::chrono::high_resolution_clock::now();

    std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
    std::string lib_full_name = prefix_lib + "lib_parsolver.so";

    // use this function
    Function solver = external("parsolver", lib_full_name);

    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    // Call the solver
    DMDict inputs = {{"X0", cons.X_0}, {"Xf", cons.X_f}};
    DMDict result = solver(inputs);

    DM X = result["X"];
    DM U = result["U"];
    DM T = result["T"];
    DM dt = result["dt"];

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
    // Print the elapsed time
    float T_opt = 0.0f; // Optimal duration for the maneuver
    if (phi_f == 90.0f * DEG) {
        T_opt = 2.42112; // Optimal duration for 90 deg turn
    } else if (phi_f == 180.0f * DEG){
        T_opt = 3.2430;
    }
    std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Computed Maneuver Duration: " << T << " s" << std::endl;
    std::cout << "Theoretical Optimal Duration: " << T_opt << std::endl;


    // Export the trajectory to a CSV file
    exportTrajectory(X, U, T, dt, "trajectory.csv"); 

    // Plot the trajectory
    std::system("python3 ../src/lib/plot_csv_data.py trajectory.csv"); // Call the Python script to plot the data
    std::string command = "python3 ../src/lib/animation.py trajectory.csv "
                        + std::to_string(i_x) + " " 
                        + std::to_string(i_y) + " " 
                        + std::to_string(i_z);
    std::system(command.c_str()); // Call the Python script to animate the data
    return 0;
}