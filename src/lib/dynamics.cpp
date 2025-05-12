#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/dynamics.h>

using namespace casadi;

// Constructor implementation
Dynamics::Dynamics() {
    X = MX::vertcat({MX::sym("q", 4), MX::sym("w", 3)});
    U = MX::sym("tau", 3);
    dt = MX::sym("dt");

    MX q = X(Slice(0,4));
    MX w = X(Slice(4));

    MX S = skew4(w);
    MX q_dot = 0.5 * MX::mtimes(S, q);

    MX I = MX::diag(MX::vertcat({i_x, i_y, i_z}));
    MX w_dot = MX::mtimes(inv(I), (U - cross(w, MX::mtimes(I, w))));

    MX X_dot = MX::vertcat({q_dot, w_dot});
    f = Function("f", {X, U}, {X_dot});
    F = Function("F", {X, U, dt}, {rk4(f, X, U, dt)});
}

// Skew-symmetric matrix
MX Dynamics::skew4(const MX& w) {
    MX S = MX::zeros(4, 4);
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);
    return S;
}

// RK4 integrator
MX Dynamics::rk4(const Function& f, const MX& x, const MX& u, const MX& dt) {
    auto k1{f(MXVector{x, u})[0]};
    auto k2{f(MXVector{x + dt / 2 * k1, u})[0]};
    auto k3{f(MXVector{x + dt / 2 * k2, u})[0]};
    auto k4{f(MXVector{x + dt * k3, u})[0]};
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