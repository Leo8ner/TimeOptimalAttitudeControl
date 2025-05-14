#include <toac/plots.h>

using namespace casadi;

void exportTrajectory(const DM& X, const DM& U, const DM& T, const DM& dt, const std::string& filename) {
    
    DM X_expanded = DM::vertcat({DM::zeros(3, X.size2()), X});

    for (int i = 0; i < X.columns(); ++i) {
        X_expanded(Slice(0, 3), i) = quat2euler(X(Slice(0, 4), i));;
    }
    std::ofstream file("../output/" + filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write X
    file << "X\n";
    for (int i = 0; i < X_expanded.rows(); ++i) {
        for (int j = 0; j < X_expanded.columns(); ++j) {
            file << X_expanded(i, j);
            if (j < X_expanded.columns() - 1) file << ",";
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
    file << dt << "\n";

    file.close();
    std::cout << "Exported trajectory to " << filename << "\n";
}

// Converts a quaternion to Euler angles
DM quat2euler(const DM& q) {
    double q0{q(0)};
    double q1{q(1)};
    double q2{q(2)};
    double q3{q(3)};

    double phi{atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))};  // Roll
    double sin_theta = 2 * (q0*q2 - q3*q1);
    sin_theta = std::max(-1.0, std::min(1.0, sin_theta));         // Clamp to the range [-1, 1]
    double theta = std::asin(sin_theta);                          // Pitch
    double psi{atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))};  // Yaw

    return DM::vertcat({phi, theta, psi});
}