#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace casadi;

int main() {

    // Start the timer
    // This is used to measure the time taken by the optimization process
    auto start = std::chrono::high_resolution_clock::now();

    // Dynamics
    Dynamics dyn; // Create an instance of the Dynamics class

    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    // Solver
    Optimizer opti(dyn, cons);     // Create an instance of the Optimizer class
    auto [X, U, T] = opti.solve(); // Solve the optimization problem

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
    // Print the elapsed time
    std::cout << "Computation time: " << elapsed.count() << " s" << std::endl;
    return 0;
}