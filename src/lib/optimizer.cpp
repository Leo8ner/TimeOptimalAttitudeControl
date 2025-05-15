#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function& dyn, const Constraints& cons) :
    F(dyn), X_0(cons.X_0), X_f(cons.X_f),
    lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {

    // // Decision variables
    // X = opti.variable(7, n_stp + 1); // State trajectory    
    // U = opti.variable(3, n_stp);     // Control trajectory (torque)
    // T = opti.variable();               // Time horizon
    // dt = T / n_stp; // time step
    
    // //// Consraints ////

    // // Box constraints
    // opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));    // Time step constraints
    // opti.subject_to(opti.bounded(lb_U, U, ub_U));       // Control constraints
    // opti.subject_to(X(0, all) >= MX::zeros(1, n_stp + 1)); // Ensure q0 >= 0 to pick a hemisphere

    // // Control input constraints
    // if (cons.Udot) {
    //     delta_U = U(Slice(), Slice(1, n_stp)) - U(Slice(), Slice(0, n_stp - 1)); 
    //     delta_U = reshape(delta_U, 3*(n_stp-1), 1); // Flatten to vector
    //     delta_U_max = dt * tau_dot_max * MX::ones(3*(n_stp-1), 1); // Also a vector
    //     opti.subject_to(opti.bounded(-delta_U_max, delta_U, delta_U_max)); // Control input rate of change
    // }

    // // Dynamics constraints
    // for (int k = 0; k < n_stp; ++k) {
    //     opti.subject_to(X(all,k+1) == F({X(all,k), U(all,k), dt})[0]); // Enforce the discretized dynamics
    // };


    // // Initial and final state constraints
    // opti.subject_to(X(all,0) == X_0);     // Initial condition
    // opti.subject_to(X(all,n_stp) == X_f); // Final condition

    // //// Initial guess ////
    // opti.set_initial(T, T_0); // Initial guess for the time horizon

    // //// Objective ////
    // opti.minimize(T);

    // ///// Solver ////
    // opti.solver("ipopt"); // Set numerical backend

    // Variables
    X = MX::sym("X", 7, n_stp + 1);   // States
    U = MX::sym("U", 3, n_stp);       // Controls
    T = MX::sym("T");                 // Final time
    dt = T / n_stp;                   // Time step
    
    // Dynamics constraints
    for (int k = 0; k < n_stp; ++k) {
        MX x_k = X(all, k);
        MX x_next = X(all, k + 1);
        MX u_k = U(all, k);
        MX f_k = F({x_k, u_k, dt})[0];
        g.push_back(x_next - f_k); // x_{k+1} == f(x_k, u_k, dt)
    }

    // Initial and final conditions
    g.push_back(X(all, 0) - X_0);               // initial
    g.push_back(X(all, n_stp) - X_f);           // final

    // Flatten constraints
    MX g_vert = vertcat(g);

    // Bounds on constraints
    args["lbg"] = DM::zeros(g_vert.size1());
    args["ubg"] = DM::zeros(g_vert.size1());

    // Decision variable vector
    MX x = vertcat(vec(X), vec(U), T); // All decision vars

    //// Initial guess ////
    DM x0 = DM::zeros(x.size1());
    x0(10) = T_0; // Initial guess for the time horizon
    args["x0"] = x0;

    //// Box constraints ////
    // Upper bounds
    DM ub_x = DM::inf(x.size1());
    ub_x(Slice(8, 11)) = ub_U;    // Control bounds
    ub_x(10) = n_stp * ub_dt;     // dt < 0.1
    args["ubx"] = DM::inf(x.size1());

    // Lower bounds
    DM lb_x = -DM::inf(x.size1());
    lb_x(0) = 0; // q0 >= 0
    lb_x(Slice(8, 11)) = lb_U; // Control bounds
    lb_x(10) = n_stp * lb_dt; // dt > 0.0001
    args["lbx"] = lb_x;   


    // Objective: minimize time
    MX obj = T;

    // Define NLP problem
    MXDict nlp = {{"x", x}, {"f", obj}, {"g", g_vert}};
    solver = nlpsol("solver", "ipopt", nlp);
};

std::tuple<DM, DM, DM, DM> Optimizer::solve() {
    // auto sol = opti.solve();
    // DM X_sol = sol.value(X);
    // DM U_sol = sol.value(U);
    // DM T_sol = sol.value(T);
    // DM dt_sol = T_sol / n_stp;
    // std::cout << "dt: " << dt_sol << " s" << std::endl;
    // std::cout << "T: " << T_sol << " s" << std::endl;
    // return std::make_tuple(X_sol, U_sol, T_sol, dt_sol);

    // Solve
    DMDict sol = solver(args);

    // Unpack
    DM x_sol = sol.at("x");
    DM X_sol = reshape(x_sol(Slice(0, 7 * (n_stp + 1))), 7, n_stp + 1);
    DM U_sol = reshape(x_sol(Slice(7 * (n_stp + 1), 7 * (n_stp + 1) + 3 * n_stp)), 3, n_stp);
    DM T_sol = x_sol(7 * (n_stp + 1) + 3 * n_stp);
    DM dt_sol = T_sol / n_stp;

    std::cout << "T: " << T_sol << " s\n";
    std::cout << "dt: " << dt_sol << " s\n";

    return std::make_tuple(X_sol, U_sol, T_sol, dt_sol);
};

