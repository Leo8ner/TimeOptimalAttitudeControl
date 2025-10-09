#include <toac/cuda_optimizer.h>
#include <toac/casadi_callback.h>
#include <toac/helper_functions.h>

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
        DM X_0, X_f, angles_0, angles_f;
        std::tie(X_0, X_f, angles_0, angles_f) = parseInput(argv[1], argv[2]);

        DM X_guess, U_guess, dt_guess; // Initial guesses for states, controls, and time steps
        std::string csv_data = "../input/initial_guess.csv"; // Path to the CSV file for initial guess
        extractInitialGuess(csv_data, X_guess, U_guess, dt_guess);

        //DynamicsCallback callback("F");
        //Function dyn = callback; // Create an instance of the optimized dynamics integrator
        Function dyn = external("F", "libtoac_shared.so");
        //BatchDynamics batch_dyn; // Create an instance of the BatchDynamics class

        //test_dynamics_and_jacobian(dyn);
        //Optimizer opti(batch_dyn.F, cons); // Create an instance of the optimizer class
        Optimizer opti(dyn); // Create an instance of the optimizer class

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
        std::cout << "Maneuver duration: " << result["T"] << " s" << std::endl;

        // Process and display results
        processResults(result, angles_0, angles_f);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}