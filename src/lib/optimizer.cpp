#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function& dyn, const Constraints& cons) :
    F(dyn), X_0(cons.X_0), X_f(cons.X_f),
    lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {

    // Decision variables
    X = opti.variable(7, n_stp + 1); // State trajectory    
    U = opti.variable(3, n_stp);     // Control trajectory (torque)
    T = opti.variable();               // Time horizon
    dt = T / n_stp; // time step
    
    //// Consraints ////

    // Box constraints
    opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));    // Time step constraints
    opti.subject_to(opti.bounded(lb_U, U, ub_U));       // Control constraints
    opti.subject_to(X(0, all) >= MX::zeros(1, n_stp + 1)); // Ensure q0 >= 0 to pick a hemisphere

    // Control input constraints
    if (cons.Udot) {
        delta_U = U(Slice(), Slice(1, n_stp)) - U(Slice(), Slice(0, n_stp - 1)); 
        delta_U = reshape(delta_U, 3*(n_stp-1), 1); // Flatten to vector
        delta_U_max = dt * tau_dot_max * MX::ones(3*(n_stp-1), 1); // Also a vector
        opti.subject_to(opti.bounded(-delta_U_max, delta_U, delta_U_max)); // Control input rate of change
    }

    // Dynamics constraints
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

std::tuple<DM, DM, DM, DM> Optimizer::solve() {
    auto sol = opti.solve();
    DM X_sol = sol.value(X);
    DM U_sol = sol.value(U);
    DM T_sol = sol.value(T);
    DM dt_sol = T_sol / n_stp;
    std::cout << "dt: " << dt_sol << " s" << std::endl;
    std::cout << "T: " << T_sol << " s" << std::endl;
    return std::make_tuple(X_sol, U_sol, T_sol, dt_sol);
};

