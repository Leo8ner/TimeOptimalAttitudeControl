#include <casadi/casadi.hpp>
#include <toac/cuda_optimizer.h>
#include <toac/constraints.h>
#include <toac/casadi_callback.h>
#include <iostream>
#include <chrono>
#include <toac/plots.h>
#include <cstdlib>

using namespace casadi;

// Add this after your existing function declaration
void test_dynamics_and_jacobian(Function &dyn) {

    // Create symbolic variables for Jacobian computation
    MX X_sym = MX::sym("X", 7, 50);
    MX U_sym = MX::sym("U", 3, 50);  
    MX dt_sym = MX::sym("dt", 50);
    
    // Create test data
    DM X_test = DM::rand(7, 50);
    DM U_test = DM::rand(3, 50);
    DM dt_test = DM::ones(50) * 0.01; // 10ms time steps
    
    // Test function output
    std::vector<DM> inputs = {X_test, U_test, dt_test};
    DM output = dyn(inputs)[0];
    
    std::cout << "Output shape: " << output.size1() << "x" << output.size2() << std::endl;
    std::cout << "Output sample:\n" << output(Slice(), Slice(0,3)) << std::endl;
    
    // Create function for Jacobian computation
    MX dyn_out = dyn(std::vector<MX>{X_sym, U_sym, dt_sym})[0];
    Function jac_func = Function("jac_func", {X_sym, U_sym, dt_sym}, {dyn_out});
    
    // Compute Jacobians
    Function jac = jac_func.jacobian(); 

    // Test function output
    std::vector<DM> jac_inputs = {X_test, U_test, dt_test, output};

    // Evaluate Jacobians
    DM jac_X = jac(jac_inputs)[0];
    DM jac_U = jac(jac_inputs)[1];
    DM jac_dt = jac(jac_inputs)[2];

    std::cout << "Jacobian w.r.t. X shape: " << jac_X.size1() << "x" << jac_X.size2() << std::endl;
    std::cout << "Jacobian w.r.t. U shape: " << jac_U.size1() << "x" << jac_U.size2() << std::endl;
    std::cout << "Jacobian w.r.t. dt shape: " << jac_dt.size1() << "x" << jac_dt.size2() << std::endl;
}

int main() {

    // Start the timer
    // This is used to measure the time taken by the optimization process
    auto start = std::chrono::high_resolution_clock::now();

    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    //DynamicsCallback callback("F");
    //Function dyn = callback; // Create an instance of the optimized dynamics integrator
    Function dyn = external("F", "libtoac_shared.so");
    //std::cout << dyn;
    //BatchDynamics batch_dyn; // Create an instance of the BatchDynamics class

    //test_dynamics_and_jacobian(dyn);
    //Optimizer opti(batch_dyn.F, cons); // Create an instance of the optimizer class
    Optimizer opti(dyn, cons); // Create an instance of the optimizer class

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