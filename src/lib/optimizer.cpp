#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>

using namespace casadi;
/*
class Optimizer {

    int n_X, n_U, n_steps;
    double T_guess;
    Function f, F;
    DM X_0, X_f;
    DM lb_U, ub_U, lb_dt, ub_dt;

public:

    Optimizer(Dynamics dyn, Constraints cons) :
        n_X(dyn.n_X), n_U(dyn.n_U), T_guess(T_0), n_steps(n_stp), f(dyn.f), F(dyn.F), X_0(cons.X_0), X_f(cons.X_f),
        lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {
        // Constructor implementation

        }
    ~Optimizer() = default; // Default destructor
};

    Opti opti {Opti()};                   // Optimization problem
    Slice all;                            // Equivalent to the slice operation in Python

    // ---- decision variables ---------
    MX X {opti.variable(n_X, n_stp + 1)};   // state trajectory    
    MX U {opti.variable(n_U, n_stp)};       // control trajectory (torque)
    MX T {opti.variable()};               // final time for time optimal control


    // -------- objective ------------
    opti.minimize(T);

    // ---- dynamic constraints --------
    SX dt = T / n_stp; // time step
    for (int k = 0; k < n_stp; ++k) {
        opti.subject_to(X(all,k+1) == F(X(all,k), U(all,k), dt)); // Enforce the discretized dynamics
    }
}*/