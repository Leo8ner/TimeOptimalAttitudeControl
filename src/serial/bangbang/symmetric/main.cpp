#include <casadi/casadi.hpp>
#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>
#include <iostream>
#include <chrono>
#include <toac/plots.h>
#include <cstdlib>

using namespace casadi;

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

        // Dynamics
        ImplicitDynamics dyn; // Create an instance of the dynamics class

        // Constraints
        Constraints cons; // Create an instance of the Constraints class

        Optimizer opti(dyn.F, cons); // Create an instance of the optimizer class

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