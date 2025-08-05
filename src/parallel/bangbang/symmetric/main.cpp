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

// Add this helper function before main()
std::tuple<DM, DM> parseStateVector(const std::string& input) {
    std::vector<double> values;
    std::istringstream stream(input);
    std::string token;
    
    while (std::getline(stream, token, ',')) {
        values.push_back(std::stod(token));
    }
    
    if (values.size() != 6) {
        throw std::invalid_argument("State vector must have exactly 6 elements");
    }

    DM quat = euler2quat(values[0] * DEG, values[1] * DEG, values[2] * DEG);
    DM omega = DM::vertcat({values[3] * DEG, values[4] * DEG, values[5] * DEG});

    DM angles_deg = DM::vertcat({values[0], values[1], values[2]});
    DM state = DM::vertcat({quat, omega});

    return  std::make_tuple(state, angles_deg);
}

// Modified main function
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " \"phi_i,theta_i,psi_i,wx_i,wy_i,wz_i\" \"phi_f,theta_f,psi_f,wx_f,wy_f,wz_f\"" << std::endl;
        return 1;
    }
    
    try {
        // Parse command line arguments

        DM X_0, angles_0;
        std::tie(X_0, angles_0) = parseStateVector(argv[1]);
        DM X_f, angles_f;
        std::tie(X_f, angles_f) = parseStateVector(argv[2]);

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

        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}};
        DMDict result = opti.solver(inputs);
        DM X = result["X"];
        DM U = result["U"];
        DM T = result["T"];
        DM dt = result["dt"];
        
        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 1000.0;
        
        // Print results (rest unchanged)
        double T_opt = 0.0;
        if (angles_0(0).scalar() == 90.0) {
            T_opt = 2.42112;
        } else if (angles_0(0).scalar() == 180.0){
            T_opt = 3.2430;
        }
        
        std::cout << "Computation Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Computed Maneuver Duration: " << T << " s" << std::endl;
        std::cout << "Theoretical Optimal Duration: " << T_opt << std::endl;

        DM X_expanded = DM::vertcat({DM::zeros(3, X.size2()), X});
        X_expanded(Slice(0, 3), 0) = angles_0 * DEG; // Initial Euler angles
        // Export and plot (unchanged)
        exportTrajectory(X_expanded, U, T, dt, "trajectory.csv");
        std::system("python3 ../src/lib/plot_csv_data.py trajectory.csv");
        std::string command = "python3 ../src/lib/animation.py trajectory.csv "
            + std::to_string(i_x) + " "
            + std::to_string(i_y) + " "
            + std::to_string(i_z);
        std::system(command.c_str());
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}