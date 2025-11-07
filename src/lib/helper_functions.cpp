#include <helper_functions.h>

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

    if (q(0).scalar() < 0) {
        q = -q; // Ensure scalar part is non-negative
    }

    return q;
}

// Converts quaternion to Euler (XYZ convention assumed here)
DM quat2euler(const DM& euler_prev, const DM& q, const DM& euler_target) {
    double q0 = q(0).scalar();
    double q1 = q(1).scalar();
    double q2 = q(2).scalar();
    double q3 = q(3).scalar();

    double phi_prev   = euler_prev(0).scalar();
    double theta_prev = euler_prev(1).scalar();
    double psi_prev   = euler_prev(2).scalar();

    double phi_target   = euler_target(0).scalar();
    double theta_target = euler_target(1).scalar();
    double psi_target   = euler_target(2).scalar();

    double phi, theta, psi;

    // Compute sin(theta) for singularity detection
    double sin_theta = 2 * (q0*q2 - q3*q1);
    sin_theta = std::max(-1.0, std::min(1.0, sin_theta));

    const double singularity_threshold = 1 - 1e-6; // Threshold to detect gimbal lock

    if (std::abs(sin_theta) > singularity_threshold) {
        // Near gimbal lock
        theta = std::asin(sin_theta);

        psi = psi_prev; // Retain previous yaw
        phi = phi_prev; // Retain previous roll

    } else {
        // Standard case
        phi   = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2));
        theta = std::asin(sin_theta);
        psi   = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
    }

    // Apply continuity + target bias
    phi   = unwrapAngle(phi,   phi_prev,   phi_target);
    theta = unwrapAngle(theta, theta_prev, theta_target);
    psi   = unwrapAngle(psi,   psi_prev,   psi_target);

    return DM::vertcat({phi, theta, psi});
}

// Helper function to unwrap angle
double unwrapAngle(double current, double previous, double target) {
    double diff = current - previous;

    // Step 1: Basic unwrap around previous
    while (diff > M_PI) {
        current -= 2.0 * M_PI;
        diff = current - previous;
    }
    while (diff < -M_PI) {
        current += 2.0 * M_PI;
        diff = current - previous;
    }

    // Step 2: Bias toward target
    double best_angle = current;
    // double best_dist = std::abs(best_angle - target);

    // // Check alternative branches (±2π)
    // double alt1 = current + 2.0 * M_PI;
    // double alt2 = current - 2.0 * M_PI;

    // if (std::abs(alt1 - target) < best_dist) {
    //     best_angle = alt1;
    //     best_dist = std::abs(best_angle - target);
    // }
    // if (std::abs(alt2 - target) < best_dist) {
    //     best_angle = alt2;
    //     best_dist = std::abs(best_angle - target);
    // }

    return best_angle;
}

// Skew-symmetric matrix
MX skew4(const MX& w) {
    MX S = MX::zeros(4, 4);
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);
    return S;
}

// RK4 integrator
MX rk4(const MX& x_dot, const MX& x, const MX& dt) {
    MX k1{x_dot};
    MX k2{substitute(x_dot, x, x + dt / 2 * k1)};
    MX k3{substitute(x_dot, x, x + dt / 2 * k2)};
    MX k4{substitute(x_dot, x, x + dt * k3)};
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
    
    double normalized_angles[3];
    for (int i = 0; i < 3; ++i) {
        // Normalize angle to (-180, 180]
        double angle = fmod(values[i], 360.0);
        if (angle <= -180.0) {
            angle += 360.0; // Ensure non-negative
        } else if (angle > 180.0) {
            angle -= 360.0; // Ensure within (-180, 180]
        }
        normalized_angles[i] = angle;

    }

    DM angles_deg = DM::vertcat({normalized_angles[0], normalized_angles[1], normalized_angles[2]});


    DM quat = euler2quat(normalized_angles[0] * DEG, normalized_angles[1] * DEG, normalized_angles[2] * DEG);
    DM omega = DM::vertcat({values[3] * DEG, values[4] * DEG, values[5] * DEG});

    DM state = DM::vertcat({quat, omega});

    for (int i = 0; i < n_states; ++i) {
        if (abs(state(i).scalar()) < 1e-6) {
            state(i) = 0.0;  // Clean up near-zero values
        }
    }

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
        
        // // Print results (rest unchanged)
        // double T_opt = 0.0;
        // if (angles_0(0).scalar() == 90.0) {
        //     T_opt = 2.42112;
        // } else if (angles_0(0).scalar() == 180.0){
        //     T_opt = 3.2430;
        // }
        
        // std::cout << "Computed Maneuver Duration: " << T << " s" << std::endl;
        // std::cout << "Theoretical Optimal Duration: " << T_opt << std::endl;

        // Export and plot (unchanged)
        exportTrajectory(X, U, T, dt, angles_0, angles_f, "trajectory.csv");
        std::system("python3 ../src/lib/toac/plot_csv_data.py trajectory.csv");
        std::string command = "python3 ../src/lib/toac/animation.py trajectory.csv "
            + std::to_string(i_x) + " "
            + std::to_string(i_y) + " "
            + std::to_string(i_z);
        std::system(command.c_str());
}

void exportTrajectory(DM& X, const DM& U, const DM& T, const DM& dt, const DM& angles_0, const DM& angles_f, const std::string& filename) {

    DM euler_traj = DM::zeros(3, X.columns());
    euler_traj(Slice(), 0) = angles_0 * DEG; // Initial angles in radians
    // Convert quaternion trajectory to Euler angles for output
    for (int i = 1; i < X.columns(); ++i) {
        euler_traj(Slice(), i) = quat2euler(euler_traj(Slice(), i-1), X(Slice(0, 4), i), angles_f * DEG);
    }
    std::ofstream file("../output/" + filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write X
    file << "X\n";
    for (int i = 0; i < euler_traj.rows(); ++i) {
        for (int j = 0; j < euler_traj.columns(); ++j) {
            file << euler_traj(i, j) * RAD;
            if (j < euler_traj.columns() - 1) file << ",";
        }
        file << "\n";
    }

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

double normalizeAngle(double angle) {
    angle = fmod(angle + 180.0, 360.0);
    if (angle < 0)
        angle += 360.0;
    return angle - 180.0;
}

// Add this helper function before main()
std::tuple<DM, DM, DM, DM> parseInput(const std::string& initial_state, const std::string& final_state) {
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

    double normalized_angles_0[3], normalized_angles_f[3];
    for (int i = 0; i < 3; ++i) {
        normalized_angles_0[i] = normalizeAngle(initial_values[i]);
        normalized_angles_f[i] = normalizeAngle(final_values[i]);
    }

    DM angles_0 = DM::vertcat({normalized_angles_0[0], normalized_angles_0[1], normalized_angles_0[2]});
    DM angles_f = DM::vertcat({normalized_angles_f[0], normalized_angles_f[1], normalized_angles_f[2]});

    DM quat_i = euler2quat(initial_values[0] * DEG, initial_values[1] * DEG, initial_values[2] * DEG);
    DM omega_i = DM::vertcat({initial_values[3] * DEG, initial_values[4] * DEG, initial_values[5] * DEG});

    DM quat_f = euler2quat(final_values[0] * DEG, final_values[1] * DEG, final_values[2] * DEG);
    DM omega_f = DM::vertcat({final_values[3] * DEG, final_values[4] * DEG, final_values[5] * DEG});

    if (dot(quat_i, quat_f).scalar() < 0) {
        quat_f = -quat_f;
    }

    DM X_0 = DM::vertcat({quat_i, omega_i});
    DM X_f = DM::vertcat({quat_f, omega_f});

    for (int i = 0; i < n_states; ++i) {
        if (abs(X_0(i).scalar()) < 1e-6) {
            X_0(i) = 0.0;  // Clean up near-zero values
        }
        if (abs(X_f(i).scalar()) < 1e-6) {
            X_f(i) = 0.0;  // Clean up near-zero values
        }
    }

    return std::make_tuple(X_0, X_f, angles_0, angles_f);
}

// Global variables to store original file descriptors
static int original_stdout_fd = -1;
static int original_stderr_fd = -1;
int log_fd = -1;

void redirect_output_to_file(const std::string& filename) {

    // Flush all streams
    fflush(stdout);
    fflush(stderr);
    
    // Create output directory if it doesn't exist
    std::system("mkdir -p ../output");
    
    // Save original file descriptors BEFORE redirecting
    original_stdout_fd = dup(STDOUT_FILENO);
    original_stderr_fd = dup(STDERR_FILENO);
    
    // Open the log file
    log_fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (log_fd == -1) {
        std::cerr << "Warning: Could not open log file: " << filename << std::endl;
        restore_output_to_console();
        return;
    }

    // Redirect stdout and stderr to the log file
    if (dup2(log_fd, STDOUT_FILENO) == -1 || dup2(log_fd, STDERR_FILENO) == -1) {
        std::cerr << "Warning: Could not redirect output to log file: " << filename << std::endl;
        restore_output_to_console();
        return;
    }
    
}

void restore_output_to_console() {

    // Flush all streams
    fflush(stdout);
    fflush(stderr);

    // Restore original stdout and stderr
    if (original_stdout_fd != -1) {
        dup2(original_stdout_fd, STDOUT_FILENO);
        close(original_stdout_fd);
        original_stdout_fd = -1;
    }
    if (original_stderr_fd != -1) {
        dup2(original_stderr_fd, STDERR_FILENO);
        close(original_stderr_fd);
        original_stderr_fd = -1;
    }

    if (log_fd != -1) {
        close(log_fd);
        log_fd = -1;
    }

}

int get_solver_status(const std::string& filename) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return -2; // File not found
    }
    
    std::string line;
    std::string last_line;
    
    // Read all lines and keep the last one
    while (std::getline(file, line)) {
        if (!line.empty()) {
            last_line = line;
        }
    }
    file.close();
    
    if (last_line.empty()) {
        std::cerr << "Error: File is empty or contains no valid lines" << std::endl;
        return -3; // Empty file
    }
    
    // Interpret the last line
    // Look for "success 1" in the last line
    if (last_line.find("success 1") != std::string::npos) {
        return 0; // Success
    } else if (last_line.find("EXIT:") != std::string::npos) {
        // Other EXIT messages indicate different types of termination
        if (last_line.find("Optimal Solution Found") != std::string::npos) {
            return 0; // Optimal Solution Found
        } else if (last_line.find("Solved To Acceptable Level") != std::string::npos) {
            return 1; // Non-optimal but acceptable solution
        } else if (last_line.find("Restoration Failed!") != std::string::npos) {
            return -2; // Restoration Failed
        } else if (last_line.find("Maximum CPU Time Exceeded") != std::string::npos) {
            return -1; // Maximum CPU Time Exceeded
        } else if (last_line.find("Maximum Number of Iterations Exceeded") != std::string::npos) {
            return -3; // Maximum Iterations Exceeded
        } else if (last_line.find("Converged to a point of local infeasibility") != std::string::npos) {
            return -4; // Converged to a point of local infeasibility
        }
    }
    return -5;
 }

using namespace std;

// Computes the dot product of two vectors of doubles
double dot(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) throw invalid_argument("Vectors must be the same size for dot product");
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Converts Euler angles to a quaternion
vector<double> Veuler2quat(const double& phi, const double& theta, const double& psi) {
    double q0 = cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2);
    double q1 = sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2);
    double q2 = cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2);
    double q3 = cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2);
    
    vector<double> q{q0, q1, q2, q3};
    
    // Normalize quaternion
    double sum = 0.0;
    for (double val : q) {
        sum += val * val;
    }
    double norm = sqrt(sum);

    for (double& val : q) {
        val /= norm;
    }
    
    // Ensure scalar part is non-negative
    if (q[0] < 0) {
        for (double& val : q) {
            val = -val;
        }
    }
    
    return q;
}

// Parse state vector from string
vector<vector<double>> VparseStateVector(const string& initial_state, const string& final_state) {
    vector<double> values_0, values_f;
    istringstream stream_0(initial_state), stream_f(final_state);
    string token;

    while (getline(stream_0, token, ',')) {
        values_0.push_back(stod(token));
    }

    while (getline(stream_f, token, ',')) {
        values_f.push_back(stod(token));
    }

    if (values_0.size() != 6 || values_f.size() != 6) {
        throw invalid_argument("State vector must have exactly 6 elements");
    }
    
    // Convert to quaternion
    vector<double> quat_0 = Veuler2quat(
        values_0[0] * DEG, 
        values_0[1] * DEG, 
        values_0[2] * DEG
    );

    vector<double> quat_f = Veuler2quat(
        values_f[0] * DEG, 
        values_f[1] * DEG, 
        values_f[2] * DEG
    );

    if (dot(quat_0, quat_f) < 0) {
        for (double& q : quat_f) {
            q = -q;
        }
    }

    // Build state vector: [q0, q1, q2, q3, omega_x, omega_y, omega_z]
    vector<double> state_0, state_f;
    state_0.reserve(n_states);
    state_f.reserve(n_states);

    // Add quaternion
    for (double q : quat_0) {
        state_0.push_back(q);
    }

    for (double q : quat_f) {
        state_f.push_back(q);
    }
    
    // Add angular velocities (converted to radians)
    for (int i = 3; i < 6; ++i) {
        state_0.push_back(values_0[i] * DEG);
        state_f.push_back(values_f[i] * DEG);
    }
    
    // Clean up near-zero values
    for (int i = 0; i < 7; ++i) {
        if (abs(state_0[i]) < 1e-6) {
            state_0[i] = 0.0;
        }
        if (abs(state_f[i]) < 1e-6) {
            state_f[i] = 0.0;
        }
    }

    return {state_0, state_f};
}

void parseMatlab(
    const int& derivativeSupplier,
    vector<vector<double>>& states,      // Output: 7×201 matrix [w,x,y,z,ωx,ωy,ωz]
    vector<vector<double>>& controls,    // Output: 3×201 matrix [ux,uy,uz]
    vector<double>& dt,          // Output: 1×201 vector [t0, t1, ..., t200]
    double& T            // Output: Total maneuver time (scalar)
) {
    string sufix = fileSufix(derivativeSupplier);
    string filename = "../output/cgpopsIPOPTSolution" + sufix + ".m";
    // Read entire file into string
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();
    
    // Storage: map[state_index][point_index] = value
    map<int, map<int, double>> state_data;
    map<int, map<int, double>> control_data;
    map<int, double> time_data;
    
    // Regex patterns - using sufix variable
    string state_pattern_str = "system" + sufix + R"(\.phase\(1\)\.x\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    string control_pattern_str = "system" + sufix + R"(\.phase\(1\)\.u\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    string time_pattern_str = "system" + sufix + R"(\.phase\(1\)\.t\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    
    regex state_pattern(state_pattern_str);
    regex control_pattern(control_pattern_str);
    regex time_pattern(time_pattern_str);

    // Parse states x(1-7)
    auto states_begin = sregex_iterator(content.begin(), content.end(), state_pattern);
    auto states_end = sregex_iterator();
    for (sregex_iterator i = states_begin; i != states_end; ++i) {
        smatch match = *i;
        int state_idx = stoi(match[1].str());
        int point_idx = stoi(match[2].str());
        double value = stod(match[3].str());
        state_data[state_idx][point_idx] = value;
    }
    
    // Parse controls u(1-3)
    auto controls_begin = sregex_iterator(content.begin(), content.end(), control_pattern);
    auto controls_end = sregex_iterator();
    for (sregex_iterator i = controls_begin; i != controls_end; ++i) {
        smatch match = *i;
        int control_idx = stoi(match[1].str());
        int point_idx = stoi(match[2].str());
        double value = stod(match[3].str());
        control_data[control_idx][point_idx] = value;
    }
    
    // Parse time t
    auto time_begin = sregex_iterator(content.begin(), content.end(), time_pattern);
    auto time_end = sregex_iterator();
    for (sregex_iterator i = time_begin; i != time_end; ++i) {
        smatch match = *i;
        int point_idx = stoi(match[1].str());
        double value = stod(match[2].str());
        time_data[point_idx] = value;
    }
    
    // Determine number of points
    int n_points = time_data.size();
    
    // Convert to casadi::DM
    // States: 7 rows × n_points columns
    states.reserve(n_states);
    for (int i = 1; i <= n_states; ++i) {
        states[i-1].reserve(n_points);
        for (int j = 1; j <= n_points; ++j) {
            states[i-1].push_back(state_data[i][j]);
        }
    }
    
    // Controls: 3 rows × n_points columns
    controls.reserve(n_controls);
    for (int i = 1; i <= n_controls; ++i) {
        controls[i-1].reserve(n_points);
        for (int j = 1; j <= n_points; ++j) {
            controls[i-1].push_back(control_data[i][j]);
        }
    }
    
    // Time: 1 row × n_points columns
    dt.reserve(n_points);
    for (int j = 1; j <= n_points; ++j) {
        dt.push_back(time_data[j]);
    }
    T = dt[n_points-1];                       // Total maneuver time is the last element
}

double getCgpopsSolution(const int& derivativeSupplier) {
    string sufix = fileSufix(derivativeSupplier);
    string filename = "../output/cgpopsIPOPTSolution" + sufix + ".m";
    
    // Read entire file into string
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();
    
    // Regex pattern to match the tf value (last line)
    string tf_pattern_str = "system" + sufix + R"(\.phase\(1\)\.tf\s*=\s*([-\d.e+-]+);)";
    regex tf_pattern(tf_pattern_str);
    
    smatch match;
    if (regex_search(content, match, tf_pattern)) {
        double tf = stod(match[1].str());
        return tf;
    } else {
        throw runtime_error("Could not find tf value in file: " + filename);
    }
}

std::string fileSufix(const int& derivative_provider) {
    std::string prefix;
    switch (derivative_provider) {
        case 0:
            prefix = "HD";
            break;
        case 1:
            prefix = "BC";
            break;
        case 2:
            prefix = "CD";
            break;
        case 3:
            prefix = "CN";
            break;
        default:    
            throw std::invalid_argument("Invalid derivative provider");
    }
    return prefix;
}

double rnd(double value, int precision) {
    double factor = std::pow(10.0, precision);
    return std::round(value * factor) / factor;
}

/**
 * @brief Load initial and final state samples from CSV file
 * @param initial_states Output vector of initial state vectors [q0,q1,q2,q3,wx,wy,wz]
 * @param final_states Output vector of final state vectors [q0,q1,q2,q3,wx,wy,wz]
 * @param filename Path to the CSV file containing state samples
 * @return true if successful, false otherwise
 */
bool loadStateSamples(std::vector<std::vector<double>>& initial_states,
                      std::vector<std::vector<double>>& final_states,
                      const std::string& filename) {
    
    std::ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(csv_file, line); // Skip header

    int line_count = 0;
    while (std::getline(csv_file, line)) {
        line_count++;
        std::stringstream ss(line);
        std::string value;
        std::vector<double> initial(n_states), final(n_states);
        
        try {
            // Read initial state: q0, q1, q2, q3, wx, wy, wz
            for (int i = 0; i < n_states; ++i) {
                if (!std::getline(ss, value, ',')) {
                    std::cerr << "Error: Incomplete initial state data at line " << line_count << std::endl;
                    csv_file.close();
                    return false;
                }
                initial[i] = std::stod(value);
            }
            
            // Read final state: q0, q1, q2, q3, wx, wy, wz
            for (int i = 0; i < n_states; ++i) {
                if (!std::getline(ss, value, ',')) {
                    std::cerr << "Error: Incomplete final state data at line " << line_count << std::endl;
                    csv_file.close();
                    return false;
                }
                final[i] = std::stod(value);
            }
            
            initial_states.push_back(initial);
            final_states.push_back(final);
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_count << ": " << e.what() << std::endl;
            csv_file.close();
            return false;
        }
    }

    csv_file.close();
    
    if (initial_states.empty()) {
        std::cerr << "Error: No valid samples loaded from " << filename << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << initial_states.size() << " samples from " << filename << std::endl;
    return true;
}

/**
 * @brief Load PSO parameter samples from CSV file
 * @param params_vector Output vector of parameter vectors
 * @param filename Path to the CSV file containing PSO samples
 * @return true if successful, false otherwise
 */
bool loadPSOSamples(std::vector<std::vector<double>>& params_vector,
                      const std::string& filename) {
    
    std::ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return false;
    }

    int n_cols = 10;

    std::string line;
    std::getline(csv_file, line); // Skip header

    int line_count = 0;
    while (std::getline(csv_file, line)) {
        line_count++;
        std::stringstream ss(line);
        std::string value;
        std::vector<double> params(n_cols);
        
        try {
            for (int i = 0; i < n_cols; ++i) {
                if (!std::getline(ss, value, ',')) {
                    if (i == 8) {
                        break;
                    } else {
                        std::cerr << "Error: Incomplete PSO parameter data at line " << line_count << std::endl;
                        csv_file.close();
                        return false;
                    }
                } 
                params[i] = std::stod(value);
            }

            params_vector.push_back(params);
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_count << ": " << e.what() << std::endl;
            csv_file.close();
            return false;
        }
    }

    csv_file.close();

    if (params_vector.empty()) {
        std::cerr << "Error: No valid samples loaded from " << filename << std::endl;
        return false;
    }

    std::cout << "Successfully loaded " << params_vector.size() << " samples from " << filename << std::endl;
    return true;
}