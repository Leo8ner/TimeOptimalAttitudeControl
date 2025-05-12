#include <iostream>
#include <casadi/casadi.hpp>

using namespace casadi;

/*
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
}*/