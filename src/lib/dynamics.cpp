#include <toac/dynamics.h>

using namespace casadi;

// Constructor implementation
Dynamics::Dynamics() {
    X = SX::vertcat({SX::sym("q", 4), SX::sym("w", 3)});
    U = SX::sym("tau", 3);
    dt = SX::sym("dt");

    SX q = X(Slice(0, 4));
    SX w = X(Slice(4, 7));

    SX S = skew4(w);
    SX q_dot = 0.5 * SX::mtimes(S, q);

    SX I = SX::diag(SX::vertcat({i_x, i_y, i_z}));
    SX w_dot = SX::mtimes(inv(I), (U - cross(w, SX::mtimes(I, w))));

    SX X_dot = SX::vertcat({q_dot, w_dot});
    SX X_next = rk4(X_dot, X, dt);
    F = Function("F", {X, U, dt}, {X_next});
//    jac_F  = F.jacobian();
//    jac_jac_F = jac_F.jacobian();

}

// Skew-symmetric matrix
SX Dynamics::skew4(const SX& w) {
    SX S = SX::zeros(4, 4);
    S(0,1) = -w(0); S(0,2) = -w(1); S(0,3) = -w(2);
    S(1,0) =  w(0); S(1,2) =  w(2); S(1,3) = -w(1);
    S(2,0) =  w(1); S(2,1) = -w(2); S(2,3) =  w(0);
    S(3,0) =  w(2); S(3,1) =  w(1); S(3,2) = -w(0);
    return S;
}

// RK4 integrator
SX Dynamics::rk4(const SX& x_dot, const SX& x, const SX& dt) {
    SX k1{x_dot};
    SX k2{substitute(x_dot, x, x + dt / 2 * k1)};
    SX k3{substitute(x_dot, x, x + dt / 2 * k2)};
    SX k4{substitute(x_dot, x, x + dt * k3)};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

// Function getDynamics() {
//         // library prefix and full name
//         std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
//         std::string lib_full_name = prefix_lib + "lib_dynamics.so";

//         // use this function
//         return external("F", lib_full_name);
// }