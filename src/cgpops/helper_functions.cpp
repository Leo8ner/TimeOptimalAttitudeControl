#include "helper_functions.h"

using namespace std;

// Converts Euler angles to a quaternion
vector<double> euler2quat(const double& phi, const double& theta, const double& psi) {
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
vector<double> parseStateVector(const string& input) {
    vector<double> values;
    istringstream stream(input);
    string token;
    
    while (getline(stream, token, ',')) {
        values.push_back(stod(token));
    }
    
    if (values.size() != 6) {
        throw invalid_argument("State vector must have exactly 6 elements");
    }
    
    // Normalize angles to (-180, 180]
    double normalized_angles[3];
    for (int i = 0; i < 3; ++i) {
        double angle = fmod(values[i], 360.0);
        if (angle <= -180.0) {
            angle += 360.0;
        } else if (angle > 180.0) {
            angle -= 360.0;
        }
        normalized_angles[i] = angle;
    }
    
    // Convert to quaternion
    vector<double> quat = euler2quat(
        normalized_angles[0] * DEG, 
        normalized_angles[1] * DEG, 
        normalized_angles[2] * DEG
    );
    
    // Build state vector: [q0, q1, q2, q3, omega_x, omega_y, omega_z]
    vector<double> state;
    state.reserve(7);
    
    // Add quaternion
    for (double q : quat) {
        state.push_back(q);
    }
    
    // Add angular velocities (converted to radians)
    for (int i = 3; i < 6; ++i) {
        state.push_back(values[i] * DEG);
    }
    
    // Clean up near-zero values
    for (int i = 0; i < 7; ++i) {
        if (abs(state[i]) < 1e-6) {
            state[i] = 0.0;
        }
    }
    
    return state;
}

void parseMatlab(
    const string& filename,
    vector<vector<double>>& states,      // Output: 7×201 matrix [w,x,y,z,ωx,ωy,ωz]
    vector<vector<double>>& controls,    // Output: 3×201 matrix [ux,uy,uz]
    vector<double>& dt,          // Output: 1×201 vector [t0, t1, ..., t200]
    double& T            // Output: Total maneuver time (scalar)
) {
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
    
    // Regex patterns
    regex state_pattern(R"(systemBC\.phase\(1\)\.x\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)");
    regex control_pattern(R"(systemBC\.phase\(1\)\.u\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)");
    regex time_pattern(R"(systemBC\.phase\(1\)\.t\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)");
    
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



