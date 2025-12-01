/**
 * @file helper_functions.cpp
 * @brief Implementation of comprehensive utility library for spacecraft attitude control
 * 
 * This file implements all utility functions declared in helper_functions.h, organized
 * into seven functional categories matching the header structure. Functions are ordered
 * to match declaration order in the header for easy cross-referencing.
 * 
 * @see helper_functions.h for complete function documentation and usage examples
 * @author [Leonardo Eitner]
 * @version 1.0
 * @date 2025
 */

#include <helper_functions.h>

using namespace casadi;
using namespace std;

// =============================================================================
// GLOBAL VARIABLE DEFINITIONS
// =============================================================================
// Actual storage allocation for global variables declared in header.
// These variables maintain state for output redirection functionality across
// multiple function calls.
//
// CRITICAL: These variables MUST be initialized to -1 to indicate "not active"
// before any redirection operations. The redirect/restore functions use these
// values to validate state and prevent invalid file descriptor operations.

int original_stdout_fd = -1;  ///< Original stdout fd (saved before redirection)
int original_stderr_fd = -1;  ///< Original stderr fd (saved before redirection)  
int log_fd = -1;              ///< Current log file fd (-1 = no redirection active)

// =============================================================================
// MATHEMATICAL CONSTANTS
// =============================================================================
// Named constants for improved code readability and maintainability

namespace {
    // Gimbal lock detection threshold
    constexpr double SINGULARITY_EPSILON = 1e-6;
    
    // Maximum number of 2π shifts to try for angle unwrapping
    constexpr int MAX_2PI_SHIFTS = 1;
    
    // Small value threshold for numerical cleanup
    constexpr double NUMERICAL_ZERO_THRESHOLD = 1e-6;
    
    // CSV file section identifiers
    const string CSV_SECTION_STATE = "X";
    const string CSV_SECTION_CONTROL = "U";
    const string CSV_SECTION_TIME = "T";
    const string CSV_SECTION_DT = "dt";
}

// =============================================================================
// QUATERNION MATHEMATICS
// =============================================================================
// Functions for quaternion operations and conversions between attitude
// representations. All functions maintain unit quaternion normalization
// and use scalar-first convention: q = [q0, q1, q2, q3] = [w, x, y, z].

/**
 * @brief Converts Euler angles (ZYX sequence) to unit quaternion
 * 
 * Implementation uses the standard aerospace ZYX (yaw-pitch-roll) sequence:
 *   q = qz(ψ) ⊗ qy(θ) ⊗ qx(φ)
 * 
 * The conversion formulas derive from the composition of rotation matrices.
 * Normalization ensures numerical stability and unit quaternion constraint.
 * 
 * @note Output quaternion scalar component is constrained to be non-negative
 *       to maintain consistent quaternion representation (q and -q represent
 *       the same rotation, so we choose the positive scalar convention)
 */
DM euler2quat(const double& phi, const double& theta, const double& psi) {
    // Compute quaternion components using half-angle formulas
    // These formulas come from quaternion composition: q = qz ⊗ qy ⊗ qx
    double q0 = cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2);
    double q1 = sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2);
    double q2 = cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2);
    double q3 = cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2);

    // Construct quaternion and normalize to eliminate numerical errors
    DM q = DM::vertcat({q0, q1, q2, q3});
    q = q / norm_2(q);

    // Enforce positive scalar convention: if q0 < 0, negate entire quaternion
    // This ensures q and -q ambiguity is resolved consistently
    if (q(0).scalar() < 0) {
        q = -q;
    }

    return q;
}

/**
 * @brief Converts quaternion to Euler angles with continuity preservation
 * 
 * ALGORITHM:
 * 1. Detect gimbal lock condition (theta ≈ ±90°)
 * 2. If regular case: compute all three angles directly
 * 3. If gimbal lock: compute coupled angle sum and distribute optimally
 * 4. Apply ±2π shifts to minimize discontinuity from previous angles
 * 
 * GIMBAL LOCK HANDLING:
 * When theta ≈ ±π/2, the ZYX Euler sequence becomes singular and phi/psi
 * are no longer independently observable. We compute their sum and distribute
 * it to minimize discontinuity with the previous trajectory point.
 * 
 * @note This implementation prioritizes trajectory continuity over canonical
 *       angle representation, making it suitable for visualization and analysis
 */
DM quat2euler(const DM& euler_prev, const DM& q) {
    // Extract quaternion components
    double q0 = q(0).scalar();
    double q1 = q(1).scalar();
    double q2 = q(2).scalar();
    double q3 = q(3).scalar();

    // Extract previous Euler angles for continuity
    double phi_prev   = euler_prev(0).scalar();
    double theta_prev = euler_prev(1).scalar();
    double psi_prev   = euler_prev(2).scalar();

    // Compute pitch angle with numerical clamping for safety
    // Formula: θ = asin(2(q0*q2 - q3*q1))
    double sin_theta = 2.0 * (q0*q2 - q3*q1);
    sin_theta = std::max(-1.0, std::min(1.0, sin_theta));  // Clamp to [-1, 1]
    double theta = std::asin(sin_theta);

    double phi = 0.0, psi = 0.0;

    // Check for gimbal lock: |sin(theta)| ≈ 1 means theta ≈ ±90°
    if (std::abs(std::abs(sin_theta) - 1.0) > SINGULARITY_EPSILON) {
        // REGULAR CASE: All three angles are independently observable
        
        // Compute roll and yaw using atan2 for proper quadrant handling
        // Roll:  φ = atan2(2(q0*q1 + q2*q3), 1 - 2(q1² + q2²))
        phi = atan2(2.0*(q0*q1 + q2*q3), 1.0 - 2.0*(q1*q1 + q2*q2));
        
        // Yaw:   ψ = atan2(2(q0*q3 + q1*q2), 1 - 2(q2² + q3²))
        psi = atan2(2.0*(q0*q3 + q1*q2), 1.0 - 2.0*(q2*q2 + q3*q3));

        // Helper lambda: Try ±2π shifts to minimize discontinuity
        auto choose_best_shift = [&](double angle, double prev) -> double {
            double best = angle;
            double best_cost = std::abs(unwrapAngle(angle, prev) - prev);
            
            for (int k = -MAX_2PI_SHIFTS; k <= MAX_2PI_SHIFTS; ++k) {
                double candidate = angle + k * 2.0 * M_PI;
                double unwrapped = unwrapAngle(candidate, prev);
                double cost = std::abs(unwrapped - prev);
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best = candidate;
                }
            }
            return best;
        };

        // Apply optimal ±2π shifts for continuity
        phi = choose_best_shift(phi, phi_prev);
        psi = choose_best_shift(psi, psi_prev);
        
    } else {
        // GIMBAL LOCK CASE: theta ≈ ±π/2
        // In this configuration, only the sum/difference of phi and psi is observable
        
        // Compute the coupled angle (phi + psi or phi - psi depending on sign)
        // Using numerically stable atan2 formulation
        double coupled_angle = atan2(2.0*(q1*q2 + q0*q3), 1.0 - 2.0*(q2*q2 + q3*q3));

        // Strategy A: Keep phi at previous value, compute psi from difference
        double phi_a = phi_prev;
        double psi_a = coupled_angle - phi_a;
        double cost_a = std::abs(unwrapAngle(phi_a, phi_prev) - phi_prev) +
                       std::abs(unwrapAngle(psi_a, psi_prev) - psi_prev);

        // Strategy B: Keep psi at previous value, compute phi from difference
        double psi_b = psi_prev;
        double phi_b = coupled_angle - psi_b;
        double cost_b = std::abs(unwrapAngle(phi_b, phi_prev) - phi_prev) +
                       std::abs(unwrapAngle(psi_b, psi_prev) - psi_prev);

        // Choose strategy with minimum discontinuity
        if (cost_a <= cost_b) {
            phi = phi_a;
            psi = psi_a;
        } else {
            phi = phi_b;
            psi = psi_b;
        }

        // Apply additional ±2π nudges if beneficial
        auto apply_optimal_shift = [&](double& angle, double prev) {
            double best = angle;
            double best_cost = std::abs(unwrapAngle(angle, prev) - prev);
            
            for (int k = -MAX_2PI_SHIFTS; k <= MAX_2PI_SHIFTS; ++k) {
                double candidate = angle + k * 2.0 * M_PI;
                double unwrapped = unwrapAngle(candidate, prev);
                double cost = std::abs(unwrapped - prev);
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best = candidate;
                }
            }
            angle = best;
        };
        
        apply_optimal_shift(phi, phi_prev);
        apply_optimal_shift(psi, psi_prev);
    }

    // Final unwrapping pass to ensure continuity
    phi   = unwrapAngle(phi, phi_prev);
    theta = unwrapAngle(theta, theta_prev);
    psi   = unwrapAngle(psi, psi_prev);

    return DM::vertcat({phi, theta, psi});
}

/**
 * @brief Unwraps angle to maintain continuity with previous value
 * 
 * Adds or subtracts multiples of 2π to minimize the difference between
 * current and previous angles, ensuring smooth trajectories without
 * artificial discontinuities at ±π boundaries.
 * 
 * ALGORITHM:
 * While (current - previous) > π:  subtract 2π from current
 * While (current - previous) < -π: add 2π to current
 */
double unwrapAngle(double current, double previous) {
    double diff = current - previous;

    // Remove positive jumps (diff > π)
    while (diff > M_PI) {
        current -= 2.0 * M_PI;
        diff = current - previous;
    }
    
    // Remove negative jumps (diff < -π)
    while (diff < -M_PI) {
        current += 2.0 * M_PI;
        diff = current - previous;
    }

    return current;
}

/**
 * @brief Normalizes angle to principal range [-π, π]
 * 
 * Uses modular arithmetic to wrap any angle to its equivalent
 * representation in the range [-180°, 180°].
 */
double normalizeAngle(double angle) {
    // Map to [0, 2π] using modulo, then shift to [-π, π]
    angle = fmod(angle + PI, 2.0 * PI);
    if (angle < 0) {
        angle += 2.0 * PI;
    }
    return angle - PI;
}

/**
 * @brief Hamilton quaternion multiplication (CasADi DM version)
 * 
 * Implements the quaternion product using Hamilton's formula:
 *   q = q1 ⊗ q2
 * 
 * GEOMETRIC MEANING:
 * Represents composition of rotations. Applying q1⊗q2 is equivalent
 * to first rotating by q2, then rotating the result by q1.
 * 
 * @note Quaternion multiplication is non-commutative: q1⊗q2 ≠ q2⊗q1
 * @note Product of two unit quaternions is automatically unit (normalized)
 */
DM quat_mul(const DM& a, const DM& b) {
    // Extract components
    double a0 = a(0).scalar(), a1 = a(1).scalar(), a2 = a(2).scalar(), a3 = a(3).scalar();
    double b0 = b(0).scalar(), b1 = b(1).scalar(), b2 = b(2).scalar(), b3 = b(3).scalar();
    
    // Apply Hamilton's quaternion multiplication formula
    double r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3;  // Scalar part
    double r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2;  // Vector x
    double r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1;  // Vector y
    double r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0;  // Vector z
    
    // Construct and normalize result
    DM r = DM::vertcat({r0, r1, r2, r3});
    return r / norm_2(r);
}

/**
 * @brief Computes quaternion conjugate (inverse for unit quaternions)
 * 
 * For quaternion q = [w, x, y, z], conjugate is q* = [w, -x, -y, -z]
 * 
 * PROPERTIES:
 * - For unit quaternions: q* = q^(-1) (conjugate equals inverse)
 * - Represents reverse rotation
 * - Used in frame transformations: v_body = q ⊗ v_inertial ⊗ q*
 */
DM quat_conj(const DM& q) {
    return DM::vertcat({q(0), -q(1), -q(2), -q(3)});
}

/**
 * @brief Vector dot product (std::vector version)
 * 
 * Computes standard Euclidean inner product of two vectors.
 * For quaternions, this measures similarity (dot ≈ 1 means similar orientations).
 */
double dot(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must be the same size for dot product");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Euler to quaternion conversion (std::vector version)
 * 
 * Standard C++ vector implementation identical to euler2quat(DM version).
 * Used by CGPOPS interface which requires std::vector types.
 */
vector<double> Veuler2quat(const double& phi, const double& theta, const double& psi) {
    // Compute quaternion components using half-angle formulas
    double q0 = cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2);
    double q1 = sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2);
    double q2 = cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2);
    double q3 = cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2);
    
    vector<double> q{q0, q1, q2, q3};
    
    // Normalize and enforce positive scalar convention
    q = normalize(q);
    
    if (q[0] < 0) {
        for (double& val : q) {
            val = -val;
        }
    }
    
    return q;
}

/**
 * @brief Quaternion multiplication (std::vector version)
 * 
 * Standard C++ vector implementation of Hamilton quaternion product.
 * Identical mathematics to quat_mul(DM version).
 */
vector<double> Vquat_mul(const vector<double>& a, const vector<double>& b) {
    double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    double b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    
    double r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3;
    double r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2;
    double r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1;
    double r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0;
    
    vector<double> r{r0, r1, r2, r3};
    return normalize(r);
}

/**
 * @brief Normalizes vector to unit length
 * 
 * Essential for maintaining unit quaternion constraint after
 * numerical operations that may introduce small errors.
 */
vector<double> normalize(const vector<double>& v) {
    double norm_squared = 0.0;
    for (double val : v) {
        norm_squared += val * val;
    }
    
    double norm = sqrt(norm_squared);
    
    vector<double> normalized;
    normalized.reserve(v.size());
    for (double val : v) {
        normalized.push_back(val / norm);
    }
    
    return normalized;
}

// =============================================================================
// STATE VECTOR PARSING & CONVERSION
// =============================================================================
// Functions for parsing command-line inputs and converting between Euler
// angles and quaternion state representations.

/**
 * @brief Parses comma-separated state strings into quaternion state vectors
 * 
 * PROCESSING STEPS:
 * 1. Parse CSV strings into numeric values
 * 2. Normalize input Euler angles to [-π, π]
 * 3. Convert Euler angles to quaternions
 * 4. Compute relative quaternion (transformation from initial to final)
 * 5. Convert angular velocities from degrees/s to radians/s
 * 6. Construct 7-element state vectors [q0, q1, q2, q3, ωx, ωy, ωz]
 * 7. Clean up near-zero numerical artifacts
 * 
 * @note Returns relative quaternion representation: initial is identity,
 *       final is the relative rotation needed
 */
std::tuple<DM, DM, DM, DM> parseInput(const std::string& initial_state, 
                                       const std::string& final_state) {
    // Parse CSV strings into numeric vectors
    vector<double> initial_values, final_values;
    istringstream initial_stream(initial_state);
    istringstream final_stream(final_state);
    string token;

    while (getline(initial_stream, token, ',')) {
        initial_values.push_back(stod(token));
    }

    while (getline(final_stream, token, ',')) {
        final_values.push_back(stod(token));
    }

    // Validate input dimensions
    if (initial_values.size() != 6 || final_values.size() != 6) {
        throw invalid_argument("State vector must have exactly 6 elements: [φ, θ, ψ, ωx, ωy, ωz]");
    }

    // Normalize Euler angles to [-π, π] range
    double norm_angles_0[3], norm_angles_f[3];
    for (int i = 0; i < 3; ++i) {
        norm_angles_0[i] = normalizeAngle(initial_values[i] * DEG);  // Convert to radians
        norm_angles_f[i] = normalizeAngle(final_values[i] * DEG);
    }

    // Store normalized angles for output
    DM angles_0 = DM::vertcat({norm_angles_0[0], norm_angles_0[1], norm_angles_0[2]});
    DM angles_f = DM::vertcat({norm_angles_f[0], norm_angles_f[1], norm_angles_f[2]});

    // Convert Euler angles to quaternions
    DM quat_i = euler2quat(norm_angles_0[0], norm_angles_0[1], norm_angles_0[2]);
    DM quat_f = euler2quat(norm_angles_f[0], norm_angles_f[1], norm_angles_f[2]);

    // Convert angular velocities from degrees/s to radians/s
    DM omega_i = DM::vertcat({initial_values[3] * DEG, initial_values[4] * DEG, initial_values[5] * DEG});
    DM omega_f = DM::vertcat({final_values[3] * DEG, final_values[4] * DEG, final_values[5] * DEG});

    // Compute relative quaternion: q_relative = q_initial^(-1) ⊗ q_final
    // This represents the rotation needed to go from initial to final attitude
    DM q_rel_f = quat_mul(quat_conj(quat_i), quat_f);
    DM q_rel_i = DM::vertcat({1.0, 0.0, 0.0, 0.0});  // Identity quaternion
    
    // Resolve quaternion double-cover ambiguity: choose representation closest to identity
    // q and -q represent the same rotation, so we choose the one with positive dot product
    if (dot(q_rel_i, q_rel_f).scalar() < 0) {
        q_rel_f = -q_rel_f;
    }

    // Construct 7-element state vectors
    DM X_0 = DM::vertcat({q_rel_i, omega_i});
    DM X_f = DM::vertcat({q_rel_f, omega_f});

    // Clean up numerical artifacts (values smaller than threshold → 0)
    for (int i = 0; i < n_states; ++i) {
        if (abs(X_0(i).scalar()) < NUMERICAL_ZERO_THRESHOLD) {
            X_0(i) = 0.0;
        }
        if (abs(X_f(i).scalar()) < NUMERICAL_ZERO_THRESHOLD) {
            X_f(i) = 0.0;
        }
    }

    return std::make_tuple(X_0, X_f, angles_0, angles_f);
}

/**
 * @brief State vector parser (std::vector version)
 * 
 * Standard C++ vector implementation of parseInput for CGPOPS interface.
 * Returns only the state vectors without separate angle outputs.
 */
vector<vector<double>> VparseStateVector(const string& initial_state, 
                                         const string& final_state) {
    // Parse CSV strings
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
    
    // Convert Euler angles to quaternions
    vector<double> quat_0 = Veuler2quat(
        values_0[0] * DEG, 
        values_0[1] * DEG, 
        values_0[2] * DEG
    );

    // Compute quaternion conjugate for frame transformation
    vector<double> q_ref = quat_0;
    q_ref[1] = -q_ref[1];  // Negate vector components
    q_ref[2] = -q_ref[2];
    q_ref[3] = -q_ref[3];

    vector<double> quat_f = Veuler2quat(
        values_f[0] * DEG, 
        values_f[1] * DEG, 
        values_f[2] * DEG
    );

    // Compute relative quaternion
    quat_f = Vquat_mul(q_ref, quat_f);
    quat_0 = {1.0, 0.0, 0.0, 0.0};  // Identity

    // Resolve double-cover ambiguity
    if (dot(quat_0, quat_f) < 0) {
        for (double& q : quat_f) {
            q = -q;
        }
    }

    // Build state vectors with angular velocities
    vector<double> state_0, state_f;
    state_0.reserve(n_states);
    state_f.reserve(n_states);

    // Add quaternion components
    for (double q : quat_0) state_0.push_back(q);
    for (double q : quat_f) state_f.push_back(q);
    
    // Add angular velocities (convert to radians/s)
    for (int i = 3; i < 6; ++i) {
        state_0.push_back(values_0[i] * DEG);
        state_f.push_back(values_f[i] * DEG);
    }
    
    // Clean up numerical artifacts
    for (int i = 0; i < n_states; ++i) {
        if (abs(state_0[i]) < NUMERICAL_ZERO_THRESHOLD) state_0[i] = 0.0;
        if (abs(state_f[i]) < NUMERICAL_ZERO_THRESHOLD) state_f[i] = 0.0;
    }

    return {state_0, state_f};
}

// =============================================================================
// TRAJECTORY I/O
// =============================================================================
// Functions for reading and writing trajectory data in various formats.

/**
 * @brief Extracts initial guess from CSV trajectory file
 * 
 * CSV FORMAT EXPECTED:
 *   Section headers: "X", "U", "T", "dt"
 *   X section: 10 rows (3 Euler angles + 7 states)
 *   U section: 3 rows (control components)
 *   dt section: 1 row (time steps)
 * 
 * This function skips the Euler angle rows (first 3 rows of X section)
 * and extracts only the state components (quaternion + angular velocity).
 */
void extractInitialGuess(const std::string& csv_data, DM& X_guess, DM& U_guess, DM& dt_guess) {
    // Open and read entire file
    ifstream file(csv_data);
    if (!file.is_open()) {
        throw runtime_error("Could not open CSV file: " + csv_data);
    }
    
    string csv_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    // Parse CSV with section detection
    istringstream stream(csv_content);
    string line;
    vector<vector<double>> x_data, u_data, dt_data;
    string current_section = "";
    
    while (getline(stream, line)) {
        if (line.empty()) continue;
        
        // Detect section headers
        if (line == CSV_SECTION_STATE || line == CSV_SECTION_CONTROL || 
            line == CSV_SECTION_TIME || line == CSV_SECTION_DT) {
            current_section = line;
            continue;
        }
        
        // Parse numeric data row
        vector<double> row;
        istringstream line_stream(line);
        string cell;
        
        while (getline(line_stream, cell, ',')) {
            row.push_back(stod(cell));
        }
        
        // Store in appropriate section
        if (current_section == CSV_SECTION_STATE) {
            x_data.push_back(row);
        } else if (current_section == CSV_SECTION_CONTROL) {
            u_data.push_back(row);
        } else if (current_section == CSV_SECTION_DT) {
            dt_data.push_back(row);
        }
        // Skip T section (not needed for initial guess)
    }
    
    // Extract state data (skip first 3 rows which are Euler angles, take rows 3-9)
    int n_cols = x_data[0].size();
    X_guess = DM::zeros(n_states, n_cols);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_cols; j++) {
            X_guess(i, j) = x_data[3 + i][j];  // Start from row 3
        }
    }
    
    // Extract control data (all 3 rows, n_cols-1 samples)
    U_guess = DM::zeros(n_controls, n_cols - 1);
    for (int i = 0; i < n_controls; i++) {
        for (int j = 0; j < n_cols - 1; j++) {
            U_guess(i, j) = u_data[i][j];
        }
    }
    
    // Extract time step data (1 row, n_cols-1 samples)
    dt_guess = DM::zeros(1, n_cols - 1);
    for (int j = 0; j < n_cols - 1; j++) {
        dt_guess(0, j) = dt_data[0][j];
    }
}

/**
 * @brief Exports optimal trajectory to CSV file
 * 
 * PROCESSING STEPS:
 * 1. Convert quaternion trajectory to Euler angles for visualization
 * 2. Transform from relative to inertial frame quaternions
 * 3. Write CSV with section headers: X, U, T, dt
 * 4. X section includes both Euler angles (degrees) and full state
 * 
 * CSV OUTPUT FORMAT:
 *   X section: 10 rows (3 Euler + 4 quaternion + 3 angular velocity)
 *   U section: 3 rows (torque components)
 *   T section: scalar (total time)
 *   dt section: row vector (time steps)
 */
void exportTrajectory(DM& X, const DM& U, const DM& T, const DM& dt, 
                     const DM& angles_0, const DM& angles_f, const std::string& filename) {
    // Allocate Euler angle trajectory storage
    DM euler_traj = DM::zeros(3, X.columns());
    euler_traj(Slice(), 0) = angles_0;  // Initialize with initial angles

    // Compute initial inertial frame quaternion from Euler angles
    DM quat_i = euler2quat(angles_0(0).scalar(), angles_0(1).scalar(), angles_0(2).scalar());
    X(Slice(0, 4), 0) = quat_i;
    
    // Convert relative quaternions to inertial frame and extract Euler angles
    for (int i = 1; i < X.columns(); ++i) {
        // Compose quaternions: q_inertial = q_initial ⊗ q_relative
        DM q_inertial = quat_mul(quat_i, X(Slice(0, 4), i));
        X(Slice(0, 4), i) = q_inertial;  // Store inertial quaternion
        
        // Convert to Euler angles with unwrapping for smooth trajectory
        euler_traj(Slice(), i) = quat2euler(euler_traj(Slice(), i - 1), q_inertial);
    }
    
    // Open output file
    ofstream file("../output/" + filename);
    if (!file.is_open()) {
        cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write Euler angles section (converted to degrees for readability)
    file << CSV_SECTION_STATE << "\n";
    for (int i = 0; i < euler_traj.rows(); ++i) {
        for (int j = 0; j < euler_traj.columns(); ++j) {
            file << euler_traj(i, j) * RAD;  // Convert radians to degrees
            if (j < euler_traj.columns() - 1) file << ",";
        }
        file << "\n";
    }

    // Write full state section (quaternion + angular velocity)
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.columns(); ++j) {
            file << X(i, j);
            if (j < X.columns() - 1) file << ",";
        }
        file << "\n";
    }

    // Write control section
    file << "\n" << CSV_SECTION_CONTROL << "\n";
    for (int i = 0; i < U.rows(); ++i) {
        for (int j = 0; j < U.columns(); ++j) {
            file << U(i, j);
            if (j < U.columns() - 1) file << ",";
        }
        file << "\n";
    }

    // Write total time section
    file << "\n" << CSV_SECTION_TIME << "\n" << T << "\n";

    // Write time step section
    file << "\n" << CSV_SECTION_DT << "\n";
    if (dt.size1() == 1) {
        // Single row vector
        file << dt << "\n";
    } else {
        // Column vector - write as row
        for (int j = 0; j < dt.rows(); ++j) {
            file << dt(j);
            if (j < dt.rows() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    cout << "Exported trajectory to " << filename << "\n";
}

/**
 * @brief Parses CGPOPS MATLAB output file
 * 
 * MATLAB FILE FORMAT:
 *   systemXX.phase(1).x(i).point(j) = value;  // States
 *   systemXX.phase(1).u(i).point(j) = value;  // Controls  
 *   systemXX.phase(1).t.point(j) = value;     // Times
 * 
 * Uses regex pattern matching to extract all data points and organize
 * them into properly sized matrices.
 */
void parseMatlab(const int& derivativeSupplier,
                vector<vector<double>>& states,
                vector<vector<double>>& controls,
                vector<double>& dt,
                double& T) {
    string suffix = fileSufix(derivativeSupplier);
    string filename = "../output/cgpopsIPOPTSolution" + suffix + ".m";
    
    // Read entire file
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open MATLAB file: " + filename);
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();
    
    // Data storage: nested maps for indexed data
    map<int, map<int, double>> state_data;
    map<int, map<int, double>> control_data;
    map<int, double> time_data;
    
    // Define regex patterns with suffix
    string state_pattern_str = "system" + suffix + 
        R"(\.phase\(1\)\.x\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    string control_pattern_str = "system" + suffix + 
        R"(\.phase\(1\)\.u\((\d+)\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    string time_pattern_str = "system" + suffix + 
        R"(\.phase\(1\)\.t\.point\((\d+)\)\s*=\s*([-\d.e+-]+);)";
    
    regex state_pattern(state_pattern_str);
    regex control_pattern(control_pattern_str);
    regex time_pattern(time_pattern_str);

    // Parse state data
    auto states_begin = sregex_iterator(content.begin(), content.end(), state_pattern);
    auto states_end = sregex_iterator();
    for (sregex_iterator i = states_begin; i != states_end; ++i) {
        smatch match = *i;
        int state_idx = stoi(match[1].str());
        int point_idx = stoi(match[2].str());
        double value = stod(match[3].str());
        state_data[state_idx][point_idx] = value;
    }
    
    // Parse control data
    auto controls_begin = sregex_iterator(content.begin(), content.end(), control_pattern);
    auto controls_end = sregex_iterator();
    for (sregex_iterator i = controls_begin; i != controls_end; ++i) {
        smatch match = *i;
        int control_idx = stoi(match[1].str());
        int point_idx = stoi(match[2].str());
        double value = stod(match[3].str());
        control_data[control_idx][point_idx] = value;
    }
    
    // Parse time data
    auto time_begin = sregex_iterator(content.begin(), content.end(), time_pattern);
    auto time_end = sregex_iterator();
    for (sregex_iterator i = time_begin; i != time_end; ++i) {
        smatch match = *i;
        int point_idx = stoi(match[1].str());
        double value = stod(match[2].str());
        time_data[point_idx] = value;
    }
    
    // Organize data into output matrices
    int n_points = time_data.size();
    
    // States: n_states rows × n_points columns
    states.resize(n_states);
    for (int i = 1; i <= n_states; ++i) {
        states[i - 1].reserve(n_points);
        for (int j = 1; j <= n_points; ++j) {
            states[i - 1].push_back(state_data[i][j]);
        }
    }
    
    // Controls: n_controls rows × n_points columns
    controls.resize(n_controls);
    for (int i = 1; i <= n_controls; ++i) {
        controls[i - 1].reserve(n_points);
        for (int j = 1; j <= n_points; ++j) {
            controls[i - 1].push_back(control_data[i][j]);
        }
    }
    
    // Time: vector of n_points
    dt.reserve(n_points);
    for (int j = 1; j <= n_points; ++j) {
        dt.push_back(time_data[j]);
    }
    
    T = dt[n_points - 1];  // Total time is last time point
}

/**
 * @brief Loads state samples from CSV for Monte Carlo analysis
 * 
 * CSV FORMAT: Each row contains 14 values:
 *   [q0_i, q1_i, q2_i, q3_i, wx_i, wy_i, wz_i, q0_f, q1_f, q2_f, q3_f, wx_f, wy_f, wz_f]
 * 
 * @return true if load successful, false if file not found or invalid format
 */
bool loadStateSamples(vector<vector<double>>& initial_states,
                     vector<vector<double>>& final_states,
                     const string& filename) {
    ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return false;
    }

    string line;
    getline(csv_file, line);  // Skip header row

    int line_count = 0;
    while (getline(csv_file, line)) {
        line_count++;
        stringstream ss(line);
        string value;
        vector<double> initial(n_states), final(n_states);
        
        try {
            // Parse initial state (7 values)
            for (int i = 0; i < n_states; ++i) {
                if (!getline(ss, value, ',')) {
                    cerr << "Error: Incomplete initial state at line " << line_count << endl;
                    csv_file.close();
                    return false;
                }
                initial[i] = stod(value);
            }
            
            // Parse final state (7 values)
            for (int i = 0; i < n_states; ++i) {
                if (!getline(ss, value, ',')) {
                    cerr << "Error: Incomplete final state at line " << line_count << endl;
                    csv_file.close();
                    return false;
                }
                final[i] = stod(value);
            }
            
            initial_states.push_back(initial);
            final_states.push_back(final);
        }
        catch (const exception& e) {
            cerr << "Error parsing line " << line_count << ": " << e.what() << endl;
            csv_file.close();
            return false;
        }
    }

    csv_file.close();
    
    if (initial_states.empty()) {
        cerr << "Error: No valid samples loaded from " << filename << endl;
        return false;
    }
    
    cout << "Successfully loaded " << initial_states.size() << " samples from " << filename << endl;
    return true;
}

/**
 * @brief Loads PSO parameter samples from CSV file
 * 
 * CSV FORMAT: Each row contains PSO hyperparameters
 * Number of columns determined by n_cols constant (typically 10)
 * 
 * @return true if load successful, false otherwise
 */
bool loadPSOSamples(vector<vector<double>>& params_vector,
                   const string& filename) {
    ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return false;
    }

    constexpr int n_cols = 10;  // PSO parameter count

    string line;
    getline(csv_file, line);  // Skip header

    int line_count = 0;
    while (getline(csv_file, line)) {
        line_count++;
        stringstream ss(line);
        string value;
        vector<double> params(n_cols);
        
        try {
            for (int i = 0; i < n_cols; ++i) {
                if (!getline(ss, value, ',')) {
                    // Allow early termination at column 8
                    if (i == 8) break;
                    
                    cerr << "Error: Incomplete PSO data at line " << line_count << endl;
                    csv_file.close();
                    return false;
                }
                params[i] = stod(value);
            }

            params_vector.push_back(params);
        }
        catch (const exception& e) {
            cerr << "Error parsing line " << line_count << ": " << e.what() << endl;
            csv_file.close();
            return false;
        }
    }

    csv_file.close();

    if (params_vector.empty()) {
        cerr << "Error: No valid samples loaded from " << filename << endl;
        return false;
    }

    cout << "Successfully loaded " << params_vector.size() << " samples from " << filename << endl;
    return true;
}

// =============================================================================
// OPTIMIZATION UTILITIES
// =============================================================================
// Functions for post-processing optimization results and extracting metrics.

/**
 * @brief Processes and visualizes optimization results
 * 
 * WORKFLOW:
 * 1. Extract trajectory data from result dictionary
 * 2. Export to CSV file
 * 3. Generate plots using Python visualization scripts
 * 4. Create 3D animation with spacecraft geometry
 */
void processResults(DMDict& results, const DM& angles_0, const DM& angles_f) {
    DM X = results["X"];
    DM U = results["U"];
    DM T = results["T"];
    DM dt = results["dt"];
    
    // Export trajectory data
    exportTrajectory(X, U, T, dt, angles_0, angles_f, "trajectory.csv");
    
    // Generate 2D plots using Python
    system("python3 ../src/lib/toac/plot_csv_data.py trajectory.csv");
    
    // Generate 3D animation with spacecraft inertia parameters
    string animation_command = "python3 ../src/lib/toac/animation.py trajectory.csv " +
                              to_string(i_x) + " " +
                              to_string(i_y) + " " +
                              to_string(i_z);
    system(animation_command.c_str());
}

/**
 * @brief Extracts objective value from CGPOPS solution file
 * 
 * Reads only the final time (tf) without parsing entire trajectory.
 * Much faster than parseMatlab() when only objective value is needed.
 * 
 * MATLAB FORMAT EXPECTED:
 *   systemXX.phase(1).tf = value;
 */
double getCgpopsSolution(const int& derivativeSupplier) {
    string suffix = fileSufix(derivativeSupplier);
    string filename = "../output/cgpopsIPOPTSolution" + suffix + ".m";
    
    // Read file content
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open MATLAB file: " + filename);
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();
    
    // Extract tf value using regex
    string tf_pattern_str = "system" + suffix + R"(\.phase\(1\)\.tf\s*=\s*([-\d.e+-]+);)";
    regex tf_pattern(tf_pattern_str);
    
    smatch match;
    if (regex_search(content, match, tf_pattern)) {
        return stod(match[1].str());
    } else {
        throw runtime_error("Could not find tf value in file: " + filename);
    }
}

/**
 * @brief Maps derivative provider code to file suffix
 * 
 * MAPPING:
 *   0 → "HD" (HyperDual automatic differentiation)
 *   1 → "BC" (Bicomplex automatic differentiation)
 *   2 → "CD" (Central finite differences)
 *   3 → "CN" (Central naive finite differences)
 */
string fileSufix(const int& derivative_provider) {
    switch (derivative_provider) {
        case 0: return "HD";
        case 1: return "BC";
        case 2: return "CD";
        case 3: return "CN";
        default:
            throw invalid_argument("Invalid derivative provider code: " + to_string(derivative_provider));
    }
}

// =============================================================================
// SOLVER INTERFACE
// =============================================================================
// Functions for loading compiled solvers and managing solver I/O.

/**
 * @brief Loads compiled CasADi solver from shared library
 * 
 * EXPECTED FILE LOCATION:
 *   ../build/lib_solver.so (relative to executable)
 * 
 * The solver must be pre-compiled using CasADi's code generation features.
 */
Function get_solver() {
    // Construct path to compiled solver library
    string build_dir = filesystem::current_path().parent_path().string() + "/build/";
    string lib_path = build_dir + "lib_solver.so";

    // Load external function from shared library
    return external("solver", lib_path);
}

/**
 * @brief Redirects stdout/stderr to log file with comprehensive error handling
 * 
 * CRITICAL SAFETY FEATURES:
 * - Validates file descriptors before all operations
 * - Checks if already redirected (prevents double redirection)
 * - Comprehensive error checking with errno reporting
 * - Atomic operations to maintain consistent state
 * - Graceful failure (warns but doesn't crash)
 * 
 * MECHANISM (POSIX):
 * 1. Check if output is already redirected (use global state)
 * 2. Flush buffers before any fd operations
 * 3. Save original stdout/stderr using dup()
 * 4. Open log file with error checking
 * 5. Redirect using dup2() with validation
 * 6. Mark redirection as active
 * 
 * @param filename Path to log file
 * 
 * @note POSIX-specific implementation (Unix/Linux/macOS)
 * @note Must call restore_output_to_console() to restore normal I/O
 * @note If redirection fails, prints warning and continues (non-fatal)
 */
void redirect_output_to_file(const string& filename) {
    // =========================================================================
    // STATE VALIDATION: Check if already redirected using ONLY global fds
    // =========================================================================
    
    if (log_fd >= 0) {
        // Already redirected - warn and return
        cerr << "Warning: Output already redirected (log_fd=" << log_fd << "). "
             << "Call restore_output_to_console() first." << endl;
        return;
    }
    
    // Defensive check: Verify saved fds are in clean state
    if (original_stdout_fd >= 0 || original_stderr_fd >= 0) {
        cerr << "Warning: File descriptors not in clean state "
             << "(stdout_fd=" << original_stdout_fd 
             << ", stderr_fd=" << original_stderr_fd << "). "
             << "Cleaning up..." << endl;
        restore_output_to_console();
    }
    
    // =========================================================================
    // BUFFER FLUSH: Ensure all pending output is written before redirection
    // =========================================================================
    
    fflush(stdout);
    fflush(stderr);
    
    // =========================================================================
    // DIRECTORY CREATION: Ensure output directory exists
    // =========================================================================
    
    system("mkdir -p ../output");
    
    // =========================================================================
    // SAVE ORIGINAL STDOUT: First operation in atomic sequence
    // =========================================================================
    
    original_stdout_fd = dup(STDOUT_FILENO);
    if (original_stdout_fd == -1) {
        cerr << "Warning: Failed to duplicate stdout (errno=" << errno 
             << "): " << strerror(errno) << endl;
        cerr << "Continuing without output redirection..." << endl;
        
        // Reset to clean state
        original_stdout_fd = -1;
        return;
    }
    
    // =========================================================================
    // SAVE ORIGINAL STDERR: Second operation in atomic sequence
    // =========================================================================
    
    original_stderr_fd = dup(STDERR_FILENO);
    if (original_stderr_fd == -1) {
        cerr << "Warning: Failed to duplicate stderr (errno=" << errno 
             << "): " << strerror(errno) << endl;
        
        // ATOMIC CLEANUP: Undo stdout save
        close(original_stdout_fd);
        original_stdout_fd = -1;
        
        cerr << "Continuing without output redirection..." << endl;
        return;
    }
    
    // =========================================================================
    // OPEN LOG FILE: Third operation in atomic sequence
    // =========================================================================
    
    log_fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (log_fd == -1) {
        cerr << "Warning: Failed to open log file '" << filename 
             << "' (errno=" << errno << "): " << strerror(errno) << endl;
        
        // ATOMIC CLEANUP: Undo all saves
        close(original_stdout_fd);
        close(original_stderr_fd);
        original_stdout_fd = -1;
        original_stderr_fd = -1;
        
        cerr << "Continuing without output redirection..." << endl;
        return;
    }
    
    // =========================================================================
    // REDIRECT STDOUT: Fourth operation in atomic sequence
    // =========================================================================
    
    if (dup2(log_fd, STDOUT_FILENO) == -1) {
        cerr << "Warning: Failed to redirect stdout (errno=" << errno 
             << "): " << strerror(errno) << endl;
        
        // ATOMIC CLEANUP: Undo everything
        close(log_fd);
        close(original_stdout_fd);
        close(original_stderr_fd);
        log_fd = -1;
        original_stdout_fd = -1;
        original_stderr_fd = -1;
        
        cerr << "Continuing without output redirection..." << endl;
        return;
    }
    
    // =========================================================================
    // REDIRECT STDERR: Final operation in atomic sequence
    // =========================================================================
    
    if (dup2(log_fd, STDERR_FILENO) == -1) {
        cerr << "Warning: Failed to redirect stderr (errno=" << errno 
             << "): " << strerror(errno) << endl;
        
        // ATOMIC CLEANUP: Restore stdout and clean up
        if (original_stdout_fd >= 0) {
            dup2(original_stdout_fd, STDOUT_FILENO);
        }
        close(log_fd);
        close(original_stdout_fd);
        close(original_stderr_fd);
        log_fd = -1;
        original_stdout_fd = -1;
        original_stderr_fd = -1;
        
        cerr << "Continuing without output redirection..." << endl;
        return;
    }
}

/**
 * @brief Restores stdout/stderr to console after redirection
 * 
 * CRITICAL SAFETY FEATURES:
 * - Validates file descriptors before restoration
 * - Checks if output is currently redirected
 * - Safe to call multiple times (idempotent)
 * - Comprehensive error checking
 * - Always cleans up resources, even on partial failure
 * 
 * MECHANISM:
 * 1. Check if output is currently redirected
 * 2. Flush buffers before restoring
 * 3. Restore stdout using saved fd
 * 4. Restore stderr using saved fd
 * 5. Close and invalidate all saved fds
 * 6. Mark redirection as inactive
 * 
 * @note Safe to call even if output was not redirected
 * @note Cleans up all file descriptors to prevent leaks
 * @note Always resets global state variables
 */
void restore_output_to_console() {
    // =========================================================================
    // STATE CHECK: Determine if restoration needed using ONLY global fds
    // =========================================================================
    
    // Quick check: If log_fd is -1 and saved fds are -1, nothing to restore
    if (log_fd == -1 && original_stdout_fd == -1 && original_stderr_fd == -1) {
        // Clean state - nothing to do (idempotent operation)
        return;
    }
    
    // If we're here, at least some cleanup is needed
    // Proceed with full restoration/cleanup procedure
    
    // =========================================================================
    // BUFFER FLUSH: Ensure all redirected output is written before restoring
    // =========================================================================
    
    fflush(stdout);
    fflush(stderr);
    
    // =========================================================================
    // RESTORATION: Restore saved file descriptors with error tracking
    // =========================================================================
        
    // Restore stdout if it was saved
    if (original_stdout_fd >= 0) {
        if (dup2(original_stdout_fd, STDOUT_FILENO) == -1) {
            // Can't reliably use cerr here - stderr might be redirected
            const char* msg = "Warning: Failed to restore stdout\n";
            write(STDERR_FILENO, msg, strlen(msg));
        }
        close(original_stdout_fd);
        original_stdout_fd = -1;
    }
    
    // Restore stderr if it was saved
    if (original_stderr_fd >= 0) {
        if (dup2(original_stderr_fd, STDERR_FILENO) == -1) {
            // Can use cerr now if stdout was restored
            cerr << "Warning: Failed to restore stderr (errno=" << errno << ")" << endl;
        }
        close(original_stderr_fd);
        original_stderr_fd = -1;
    }
    
    // Close log file if it was opened
    if (log_fd >= 0) {
        if (close(log_fd) == -1) {
            cerr << "Warning: Failed to close log file (errno=" << errno 
                 << "): " << strerror(errno) << endl;
        }
        log_fd = -1;
    }
}

/**
 * @brief Parses solver log to determine convergence status
 * 
 * Searches log file for IPOPT exit messages to classify solution quality.
 * 
 * RETURN CODES:
 *   0: Optimal solution found
 *   1: Solved to acceptable level
 *  -1: Maximum CPU time exceeded
 *  -2: Restoration failed
 *  -3: Maximum iterations exceeded
 *  -4: Converged to local infeasibility
 *  -5: Other failure or unknown status
 */
int get_solver_status(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open log file: " << filename << endl;
        return -2;
    }
    
    string line, last_line;
    
    // Read all lines, keeping the last non-empty one
    while (getline(file, line)) {
        if (!line.empty()) {
            last_line = line;
        }
    }
    file.close();
    
    if (last_line.empty()) {
        cerr << "Error: Log file is empty" << endl;
        return -3;
    }
    
    // Parse last line for status indicators
    if (last_line.find("success 1") != string::npos) {
        return 0;  // CasADi success indicator
    } else if (last_line.find("EXIT:") != string::npos) {
        // IPOPT exit messages
        if (last_line.find("Optimal Solution Found") != string::npos) {
            return 0;
        } else if (last_line.find("Solved To Acceptable Level") != string::npos) {
            return 1;
        } else if (last_line.find("Restoration Failed!") != string::npos) {
            return -2;
        } else if (last_line.find("Maximum CPU Time Exceeded") != string::npos) {
            return -1;
        } else if (last_line.find("Maximum Number of Iterations Exceeded") != string::npos) {
            return -3;
        } else if (last_line.find("Converged to a point of local infeasibility") != string::npos) {
            return -4;
        }
    }
    
    return -5;  // Unknown status
}

/**
 * @brief Count number of existing result rows in output CSV file
 * 
 * Determines how many simulations have already been completed by counting
 * non-empty lines in the results file (excluding header). This enables
 * crash recovery by resuming from the last completed iteration.
 * 
 * @param filepath Path to results CSV file
 * @param file_exists Output parameter: true if file exists
 * 
 * @return Number of data rows already written (0 if file doesn't exist or is empty)
 * 
 * @note Assumes first line is header (skipped in count)
 * @note Empty lines are ignored
 * @note Returns 0 if file cannot be opened
 */
int countExistingResults(const std::string& filepath, bool& file_exists) {
    file_exists = std::filesystem::exists(filepath);
    if (!file_exists) {
        return 0;
    }
    
    int row_count = 0;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file for reading: " << filepath << std::endl;
        return 0;
    }
    
    std::string line;
    // Skip header line
    if (std::getline(file, line)) {
        // Count non-empty data lines
        while (std::getline(file, line)) {
            if (!line.empty()) {
                ++row_count;
            }
        }
    }
    
    file.close();
    return row_count;
}

/**
 * @brief Initialize results CSV file with header or open for appending
 * 
 * Creates new results file with header if needed, or opens existing file
 * for appending. Handles overwrite mode and crash recovery scenarios.
 * 
 * @param filepath Path to results CSV file
 * @param overwrite If true, create new file (overwrites existing)
 * @param file_exists Input: whether file currently exists
 * 
 * @return FILE* pointer for writing, or nullptr on error
 * 
 * @note Caller is responsible for closing returned FILE*
 * @note Returns nullptr on failure (error message printed)
 * @note Header format: "T, time, status\n"
 */
FILE* initializeResultsFile(const std::string& filepath, bool overwrite, bool file_exists) {
    FILE* file = nullptr;
    bool need_header = false;
    
    if (overwrite) {
        // Create new file, overwriting existing
        file = std::fopen(filepath.c_str(), "w");
        need_header = true;
    } else {
        if (!file_exists || std::filesystem::file_size(filepath) == 0) {
            // Create new file
            file = std::fopen(filepath.c_str(), "w");
            need_header = true;
        } else {
            // Append to existing file
            file = std::fopen(filepath.c_str(), "a");
            need_header = false;
        }
    }
    
    if (!file) {
        std::cerr << "Error: Could not open results file for writing: " << filepath << std::endl;
        return nullptr;
    }
    
    // Write CSV header if creating new file
    if (need_header) {
        std::fprintf(file, "T, time, status\n");
        std::fflush(file);
        fsync(fileno(file));  // Ensure header written to disk
    }
    
    return file;
}

/**
 * @brief Display progress update with time estimates
 * 
 * Prints formatted progress information including:
 * - Current iteration and total count
 * - Completion percentage
 * - Elapsed time
 * - Estimated time remaining
 * 
 * @param current_iteration Current iteration number (0-indexed)
 * @param total_iterations Total number of iterations
 * @param start_time Time point when processing started
 * 
 * @note Output formatted for readability with fixed precision
 * @note Time estimates improve in accuracy as more iterations complete
 */
void reportProgress(int current_iteration, int total_iterations,
                   const std::chrono::high_resolution_clock::time_point& start_time) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        current_time - start_time).count();
    
    // Calculate time statistics
    double progress_fraction = static_cast<double>(current_iteration) / total_iterations;
    double time_remaining = (elapsed_seconds / progress_fraction) * (1.0 - progress_fraction);
    double progress_percent = 100.0 * progress_fraction;
    
    // Format elapsed time
    int elapsed_min = elapsed_seconds / 60;
    int elapsed_sec = elapsed_seconds % 60;
    
    // Format remaining time
    int remaining_min = static_cast<int>(time_remaining) / 60;
    int remaining_sec = static_cast<int>(time_remaining) % 60;
    
    // Display progress
    std::cout << "Progress: " << current_iteration << "/" << total_iterations
              << " (" << std::fixed << std::setprecision(1) << progress_percent << "%)"
              << std::endl;
    std::cout << "Elapsed: " << elapsed_min << " min " << elapsed_sec << " sec, "
              << "Estimated remaining: " << remaining_min << " min " << remaining_sec << " sec"
              << std::endl;
}

// =============================================================================
// NUMERICAL INTEGRATION
// =============================================================================
// Functions for numerical integration and mathematical operators.

/**
 * @brief Constructs 4×4 skew-symmetric matrix for quaternion kinematics
 * 
 * MATRIX STRUCTURE:
 *        ⎡  0   -ωx  -ωy  -ωz ⎤
 *   Ω = ⎢  ωx    0   ωz  -ωy ⎥
 *        ⎢  ωy  -ωz    0   ωx ⎥
 *        ⎣  ωz   ωy  -ωx    0 ⎦
 * 
 * USAGE:
 *   Quaternion rate equation: q̇ = 0.5 * Ω(ω) * q
 * 
 * PROPERTIES:
 *   - Skew-symmetric: Ω^T = -Ω
 *   - Maps 3D angular velocity to 4D quaternion rate
 */
MX skew4(const MX& w) {
    MX S = MX::zeros(4, 4);
    
    // First row: [0, -ωx, -ωy, -ωz]
    S(0, 1) = -w(0);
    S(0, 2) = -w(1);
    S(0, 3) = -w(2);
    
    // Second row: [ωx, 0, ωz, -ωy]
    S(1, 0) =  w(0);
    S(1, 2) =  w(2);
    S(1, 3) = -w(1);
    
    // Third row: [ωy, -ωz, 0, ωx]
    S(2, 0) =  w(1);
    S(2, 1) = -w(2);
    S(2, 3) =  w(0);
    
    // Fourth row: [ωz, ωy, -ωx, 0]
    S(3, 0) =  w(2);
    S(3, 1) =  w(1);
    S(3, 2) = -w(0);
    
    return S;
}

/**
 * @brief Performs one step of 4th-order Runge-Kutta integration
 * 
 * CLASSICAL RK4 ALGORITHM:
 *   k1 = f(x)
 *   k2 = f(x + dt/2 * k1)
 *   k3 = f(x + dt/2 * k2)
 *   k4 = f(x + dt * k3)
 *   x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 * 
 * PROPERTIES:
 *   - 4th order accurate: local error ~ O(dt^5)
 *   - Requires 4 function evaluations per step
 *   - Good balance of accuracy and computational cost
 * 
 * @note For quaternion states, normalize after integration to maintain ||q|| = 1
 */
MX rk4(const MX& x_dot, const MX& x, const MX& dt) {
    // Compute RK4 stages
    MX k1 = x_dot;
    MX k2 = substitute(x_dot, x, x + dt / 2 * k1);
    MX k3 = substitute(x_dot, x, x + dt / 2 * k2);
    MX k4 = substitute(x_dot, x, x + dt * k3);
    
    // Weighted combination of stages
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

// =============================================================================
// SYSTEM UTILITIES
// =============================================================================
// Miscellaneous utility functions for numerical formatting.

/**
 * @brief Rounds floating-point value to specified decimal precision
 * 
 * ALGORITHM:
 *   1. Multiply by 10^precision
 *   2. Round to nearest integer
 *   3. Divide by 10^precision
 * 
 * @note Result is still double type (not truncated string representation)
 * @note Precision > 15 may not be meaningful due to double precision limits
 */
double rnd(double value, int precision) {
    double factor = pow(10.0, precision);
    return round(value * factor) / factor;
}

// =============================================================================
// END OF IMPLEMENTATION
// =============================================================================