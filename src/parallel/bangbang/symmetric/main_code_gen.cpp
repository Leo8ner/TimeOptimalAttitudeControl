#include <casadi/casadi.hpp>
#include <toac/cuda_optimizer.h>
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

// Add this helper function before main()
std::tuple<DM, DM> parseInput(const std::string& initial_state, const std::string& final_state) {
    std::vector<double> initial_values, final_values;
    std::istringstream initial_stream(initial_state);
    std::istringstream final_stream(final_state);
    std::string token;

    while (std::getline(initial_stream, token, ',')) {
        initial_values.push_back(std::stod(token));
    }

    while (std::getline(final_stream, token, ',')) {
        final_values.push_back(std::stod(token));
    }

    if (initial_values.size() != 6 || final_values.size() != 6) {
        throw std::invalid_argument("State vector must have exactly 6 elements");
    }

    DM quat_i = euler2quat(0, 0, 0);
    DM omega_i = DM::vertcat({initial_values[3] * DEG, initial_values[4] * DEG, initial_values[5] * DEG});

    double phi_f = (final_values[0] - initial_values[0]) * DEG;
    double theta_f = (final_values[1] - initial_values[1]) * DEG;
    double psi_f = (final_values[2] - initial_values[2]) * DEG;
    DM quat_f = euler2quat(phi_f, theta_f, psi_f);
    DM omega_f = DM::vertcat({final_values[3] * DEG, final_values[4] * DEG, final_values[5] * DEG});
    DM X_0 = DM::vertcat({quat_i, omega_i});
    DM X_f = DM::vertcat({quat_f, omega_f});

    return std::make_tuple(X_0, X_f);
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

        std::tie(X_0, X_f) = parseInput(argv[1], argv[2]);

        // Start the timer
        // This is used to measure the time taken by the optimization process
        auto start = std::chrono::high_resolution_clock::now();

        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_parsolver.so";

        // use this function
        Function solver = external("parsolver", lib_full_name);

        // Call the solver with parsed inputs
        DMDict inputs = {{"X0", X_0}, {"Xf", X_f}};
        DMDict result = solver(inputs);
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