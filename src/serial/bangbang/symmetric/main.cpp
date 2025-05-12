#include <iostream>
#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

// Discretized dynamics of the system X_k+1 = F(X_k,U_k)
SX rk4(const Function& f, const SX& x, const SX& u, const SX& dt) {
    auto k1{f(SXVector{x, u})[0]};
    auto k2{f(SXVector{x + dt / 2 * k1, u})[0]};
    auto k3{f(SXVector{x + dt / 2 * k2, u})[0]};
    auto k4{f(SXVector{x + dt * k3, u})[0]};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

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

int main() {

    // Symbolic variables
    SX X{SX::vertcat({
        SX::sym("q", 4),      // quaternion vector
        SX::sym("w", 3)})};   // angular rates vector
        
    SX U{SX::sym("tau", 3)};  // torque vector

    SX dt{SX::sym("dt")};     // time step

    auto n_x{X.size1()};       // number of state variables----------, -
    auto n_u{U.size1()};       // number of control inputs-----------, -

    // Dynamics equations
    SX q{X(Slice(0,4))};      // Quaternion-------------------------, -
    SX w{X(Slice(4))};        // Angular rate-----------------------, rad/s
    SX tau{U};                // Torque-----------------------------, Nm

    SX S = SX::zeros(4, 4);                                           // Skew-symmetric matrix
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);

    DM I{DM::diag({i_x, i_y, i_z})};                                  // Inertia matrix
    
    // Dynamics
    SX q_dot = 0.5 * SX::mtimes(S,q);                                 // Quaternion derivative
    SX w_dot = SX::mtimes(inv(I), (tau - cross(w, SX::mtimes(I,w)))); // Angular rate derivative
    SX X_dot = SX::vertcat({q_dot, w_dot});                           // State derivative

    Function f = Function("f", {X, U}, {X_dot});                      // Function for continuous dynamics
    Function F = Function("F", {X, U, dt}, {rk4(f, X, U, dt)});       // Function for discretized dynamics

    // Constraints
    DM q_0{euler2quat(phi_0, theta_0, psi_0)};                        // Initial quaternion
    DM X_0{DM::vertcat({q_0, wx_0, wy_0, wz_0})};                     // Initial state

    DM q_f{euler2quat(phi_f, theta_f, psi_f)};                        // Final quaternion
    DM X_f{DM::vertcat({q_f, wx_f, wy_f, wz_f})};                     // Final state

    DM lb_U{DM::vertcat({-tau_max, -tau_max, -tau_max})};             // Lower bound for torque
    DM ub_U{DM::vertcat({ tau_max,  tau_max,  tau_max})};             // Upper bound for torque

    return 0;
}