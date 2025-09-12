#include <toac/helper_functions.h>

using namespace casadi;

// Converts Euler angles to a quaternion
DM euler2quat(const double& phi, const double& theta, const double& psi) {
    double q0{cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2)};
    double q1{sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2)};
    double q2{cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2)};
    double q3{cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2)};

    // Normalize the quaternion to eliminate numerical errors
    DM q{DM::vertcat({q0, q1, q2, q3})}; 
    q = q / norm_2(q); 

    return q;
}

// Converts a quaternion to Euler angles with continuity preservation
DM quat2euler(const DM& euler_angles, const DM& q) {
    double q0 = static_cast<double>(q(0));
    double q1 = static_cast<double>(q(1));
    double q2 = static_cast<double>(q(2));
    double q3 = static_cast<double>(q(3));
    
    double phi_prev = static_cast<double>(euler_angles(0));
    double theta_prev = static_cast<double>(euler_angles(1));
    double psi_prev = static_cast<double>(euler_angles(2));
    
    double phi, theta, psi;
    
    // Calculate sin(theta) for singularity detection
    double sin_theta = 2 * (q0*q2 - q3*q1);
    sin_theta = std::max(-1.0, std::min(1.0, sin_theta));
    
    // Singularity threshold
    const double singularity_threshold = 0.99;
    
    if (std::abs(sin_theta) > singularity_threshold) {
        // Gimbal lock case - use alternative formulation
        theta = std::asin(sin_theta);
        
        if (sin_theta > singularity_threshold) {
            // Positive singularity (theta ≈ +90°)
            phi = atan2(2*(q1*q2 + q0*q3), q0*q0 + q1*q1 - q2*q2 - q3*q3);
            psi = 0.0; // Set psi to zero by convention
        } else {
            // Negative singularity (theta ≈ -90°)
            phi = atan2(-2*(q1*q2 + q0*q3), q0*q0 + q1*q1 - q2*q2 - q3*q3);
            psi = 0.0; // Set psi to zero by convention
        }
    } else {
        // Normal case - use standard formulation
        phi = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2));
        theta = std::asin(sin_theta);
        psi = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
    }
    
    // Apply continuity correction to all angles
    phi = unwrapAngle(phi, phi_prev);
    theta = unwrapAngle(theta, theta_prev);
    psi = unwrapAngle(psi, psi_prev);

    return DM::vertcat({phi, theta, psi});
}

// Helper function to unwrap angle
double unwrapAngle(double current_angle, double previous_angle) {
    double diff = current_angle - previous_angle;
    
    // If difference is greater than π, subtract 2π
    while (diff > M_PI) {
        current_angle -= 2.0 * M_PI;
        diff = current_angle - previous_angle;
    }
    
    // If difference is less than -π, add 2π
    while (diff < -M_PI) {
        current_angle += 2.0 * M_PI;
        diff = current_angle - previous_angle;
    }
    
    return current_angle;
}

// Skew-symmetric matrix
SX skew4(const SX& w) {
    SX S = SX::zeros(4, 4);
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);
    return S;
}

// RK4 integrator
SX rk4(const SX& x_dot, const SX& x, const SX& dt) {
    SX k1{x_dot};
    SX k2{substitute(x_dot, x, x + dt / 2 * k1)};
    SX k3{substitute(x_dot, x, x + dt / 2 * k2)};
    SX k4{substitute(x_dot, x, x + dt * k3)};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

Function get_solver() {
        // library prefix and full name
        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_solver.so";

        // use this function
        return external("solver", lib_full_name);
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

void extractInitialGuess(const std::string& csv_data, DM& X_guess, DM& U_guess, DM& dt_guess) {
    // Read CSV file
    std::ifstream file(csv_data);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + csv_data);
    }
    
    // Load file content
    std::string csv_content((std::istreambuf_iterator<char>(file)), 
                            std::istreambuf_iterator<char>());
    file.close();
    
    // Parse CSV data with header detection
    std::istringstream stream(csv_content);
    std::string line;
    std::vector<std::vector<double>> x_data, u_data, dt_data;
    std::string current_section = "";
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        // Check if line is a header
        if (line == "X" || line == "U" || line == "T" || line == "dt") {
            current_section = line;
            continue;
        }
        
        // Parse numeric data
        std::vector<double> row;
        std::istringstream line_stream(line);
        std::string cell;
        
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        
        // Store data in appropriate section
        if (current_section == "X") {
            x_data.push_back(row);
        } else if (current_section == "U") {
            u_data.push_back(row);
        } else if (current_section == "dt") {
            dt_data.push_back(row);
        }
        // Skip "T" section as it's not needed
    }
    
    // Extract X (rows 3-9, which are the last 7 rows)
    int n_cols = x_data[0].size();
    X_guess = DM::zeros(7, n_cols);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < n_cols; j++) {
            X_guess(i, j) = x_data[3 + i][j];  // Start from row 3
        }
    }
    
    // Extract U (all 3 rows)
    U_guess = DM::zeros(3, n_cols-1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < n_cols-1; j++) {
            U_guess(i, j) = u_data[i][j];
        }
    }
    
    // Extract dt (1 row)
    dt_guess = DM::zeros(1, n_cols-1);
    for (int j = 0; j < n_cols-1; j++) {
        dt_guess(0, j) = dt_data[0][j];
    }
}

void processResults(DMDict& results, const DM& angles_0, const DM& angles_f) {
        DM X = results["X"];
        DM U = results["U"];
        DM T = results["T"];
        DM dt = results["dt"];
        
        // Print results (rest unchanged)
        double T_opt = 0.0;
        if (angles_0(0).scalar() == 90.0) {
            T_opt = 2.42112;
        } else if (angles_0(0).scalar() == 180.0){
            T_opt = 3.2430;
        }
        
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
}

void exportTrajectory(DM& X, const DM& U, const DM& T, const DM& dt, const std::string& filename) {
    
    for (int i = 1; i < X.columns(); ++i) {
        // X(Slice(0, 3), i) = quat2euler(X(Slice(0, 3), i-1) - X(Slice(0, 3), 0), X(Slice(3, 7), i)) + X(Slice(0, 3), 0);
        X(Slice(0,3), 0) = DM::zeros(3); // Set initial Euler angles to zero for continuity
        X(Slice(0, 3), i) = quat2euler(X(Slice(0, 3), i-1), X(Slice(3, 7), i));
    }
    std::ofstream file("../output/" + filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write X
    file << "X\n";
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.columns(); ++j) {
            file << X(i, j);
            if (j < X.columns() - 1) file << ",";
        }
        file << "\n";
    }

    // Write U
    file << "\nU\n";
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.columns(); ++j) {
            file << U(i, j);
            if (j < U.columns() - 1) file << ",";
        }
        file << "\n";
    }

    // Write T
    file << "\nT\n";
    file << T << "\n";

    // Write dt
    file << "\ndt\n";
    if (dt.size1() == 1)  {
        file << dt << "\n";
    } else {
        for (int j = 0; j < dt.rows(); ++j) {
            file << dt(j);
            if (j < dt.rows() - 1) file << ",";
        }
        return;
    }

    file << "\n";

    file.close();
    std::cout << "Exported trajectory to " << filename << "\n";
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

