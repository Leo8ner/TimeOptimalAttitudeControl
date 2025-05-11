#include <iostream>
#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

// Discretized dynamics of the system X_k+1 = F(X_k,U_k)
MX rk4(const MX& x, const MX& u, const MX& dt) {
    MX k1{f(x, u)};
    MX k2{f(x + dt / 2 * k1, u)};
    MX k3{f(x + dt / 2 * k2, u)};
    MX k4{f(x + dt * k3, u)};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

DM euler2quat(const double& phi, const double& theta, const double& psi) {
    // Convert Euler angles to quaternion
    double q0{cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2)};
    double q1{sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2)};
    double q2{cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2)};
    double q3{cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2)};

    // Normalize the quaternion to eliminate numerical errors
    DM q{DM::vertcat({q0, q1, q2, q3})}; 
    q = q / norm_2(q); 

    return q;
}

int main() {

    // Symbolic variables
    SX X{SX::vertcat({
        SX::sym("q", 4),      // quaternion vector
        SX::sym("w", 3)})};   // angular rates vector
        
    SX U{SX::sym("tau", 3)}; // torque vector

    SX dt{SX::sym("dt")}; // time step

    // Dynamics equations
    SX q{X(Slice(0,4))};      // Quaternion-------------------------, -
    SX w{X(Slice(4))};        // Angular rate-----------------------, rad/s
    SX tau{U};                // Torque-----------------------------, Nm

    SX S{SX::vertcat({
        {0, w(0), -w(1), w(0)},
        {-w(2), 0, w(0), w(1)},
        {w(1), -w(0), 0, w(2)},
        {-w(0), -w(1), -w(2), 0}})}; // Skew-symmetric matrix


}