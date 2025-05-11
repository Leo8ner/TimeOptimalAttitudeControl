#include <iostream>
#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

// Dynamics of the system dX/dt = f(X,U)
MX f(const MX& x, const MX& u) {
    return vertcat(x(1), u-x(1));
  }

// Discretized dynamics of the system X_k+1 = F(X_k,U_k)
MX rk4(const MX& x, const MX& u, const MX& dt) {
    MX k1 {f(x, u)};
    MX k2 {f(x + dt / 2 * k1, u)};
    MX k3 {f(x + dt / 2 * k2, u)};
    MX k4 {f(x + dt * k3, u)};
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

int main() {

    Opti opti {Opti()};                   // Optimization problem
    Slice all;                            // Equivalent to the slice operation in Python

    // ---- decision variables ---------
    MX X {opti.variable(6, n_stp + 1)};   // state trajectory    
    MX U {opti.variable(3, n_stp)};       // control trajectory (torque)
    MX T {opti.variable()};               // final time for time optimal control

    // -------- objective ------------
    opti.minimize(T);

    // ---- dynamic constraints --------
    MX dt = T / n_stp; // time step
    for (int k = 0; k < n_stp; ++k) {
        opti.subject_to(X(all,k+1) == F(X(all,k), U(all,k), dt)); // Enforce the discretized dynamics
    }
}