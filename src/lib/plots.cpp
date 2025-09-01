#include <toac/plots.h>

using namespace casadi;

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