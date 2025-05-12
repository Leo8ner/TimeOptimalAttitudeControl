#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>

using namespace casadi;

class Optimizer {

    int n_X, n_U;
    Function f, F;
    DM X_0, X_f;
    DM lb_U, ub_U, lb_dt, ub_dt;
    Opti opti {Opti()};                   // Optimization problem
    Slice all;                            // Equivalent to the slice operation in Python

public:

    Optimizer(const Dynamics& dyn, const Constraints& cons) :
        n_X(dyn.n_X), n_U(dyn.n_U), f(dyn.f), F(dyn.F), X_0(cons.X_0), X_f(cons.X_f),
        lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {

        // Decision variables
        MX X {opti.variable(n_X, n_stp + 1)}; // State trajectory    
        MX U {opti.variable(n_U, n_stp)};     // Control trajectory (torque)
        MX T {opti.variable()};               // Time horizon
        MX dt = T / n_stp; // time step
        
        //// Consraints

        // Box constraints
        opti.subject_to(opti.bounded(lb_dt, dt, ub_dt)); // Time step constraints
        opti.subject_to(opti.bounded(lb_U, U, ub_U));     // Control constraints

        // Dyamics constraints
        for (int k = 0; k < n_stp; ++k) {
            opti.subject_to(X(all,k+1) == F({X(all,k), U(all,k), dt})[0]); // Enforce the discretized dynamics
        };

        // Initial and final state constraints
        opti.subject_to(X(all,0) == X_0);     // Initial condition
        opti.subject_to(X(all,n_stp) == X_f); // Final condition

        //// Initial guess
        opti.set_initial(T, T_0); // Initial guess for the state trajectory

        opti.solver("ipopt");         // set numerical backend
        OptiSol sol = opti.solve();   // actual solve

        }






    // -------- objective ------------
    opti.minimize(T);

    // ---- dynamic constraints --------
    MX dt = T / n_stp; // time step

};