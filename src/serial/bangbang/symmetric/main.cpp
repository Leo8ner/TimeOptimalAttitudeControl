#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
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

    // Dynamics
    ImplicitDynamics dyn; // Create an instance of the dynamics class

    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    Optimizer opti(dyn.F, cons); // Create an instance of the optimizer class

    // Call the solver
    DMDict inputs = {{"X0", cons.X_0}, {"Xf", cons.X_f}};
    DMDict result = opti.solver(inputs);

    DM X = result["X"];
    DM U = result["U"];
    DM T = result["T"];
    DM dt = result["dt"];

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
    // Print the elapsed time
    std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Maneuver Duration: " << T << " s" << std::endl;


    // Export the trajectory to a CSV file
    exportTrajectory(X, U, T, dt, "trajectory.csv"); 

    // Plot the trajectory
    std::system("python3 ../src/lib/plot_csv_data.py trajectory.csv"); // Call the Python script to plot the data
    
    return 0;
}