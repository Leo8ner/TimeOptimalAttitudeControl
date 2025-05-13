#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Dynamics& dyn, const Constraints& cons) :
    n_X(dyn.n_X), n_U(dyn.n_U), f(dyn.f), F(dyn.F), X_0(cons.X_0), X_f(cons.X_f),
    lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {

    // Decision variables
    X = opti.variable(n_X, n_stp + 1); // State trajectory    
    U = opti.variable(n_U, n_stp);     // Control trajectory (torque)
    T = opti.variable();               // Time horizon
    dt = T / n_stp; // time step
    
    //// Consraints ////

    // Box constraints
    opti.subject_to(opti.bounded(lb_dt, dt, ub_dt)); // Time step constraints
    opti.subject_to(opti.bounded(lb_U, U, ub_U));    // Control constraints

    // Dyamics constraints
    for (int k = 0; k < n_stp; ++k) {
        opti.subject_to(X(all,k+1) == F({X(all,k), U(all,k), dt})[0]); // Enforce the discretized dynamics
    };

    // Initial and final state constraints
    opti.subject_to(X(all,0) == X_0);     // Initial condition
    opti.subject_to(X(all,n_stp) == X_f); // Final condition

    //// Initial guess ////
    opti.set_initial(T, T_0); // Initial guess for the time horizon

    //// Objective ////
    opti.minimize(T);

    ///// Solver ////
    opti.solver("ipopt"); // Set numerical backend
};

std::tuple<DM, DM, DM> Optimizer::solve() {
    auto sol = opti.solve();
    DM X_sol = sol.value(X);
    DM U_sol = sol.value(U);
    DM T_sol = sol.value(T);
    DM dt_sol = T_sol / n_stp;
    std::cout << "dt: " << dt_sol << " s" << std::endl;
    std::cout << "T: " << T_sol << " s" << std::endl;
    return std::make_tuple(X_sol, U_sol, T_sol);
};

