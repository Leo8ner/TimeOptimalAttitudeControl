#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function& dyn, const Constraints& cons) :
    F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt) {

    // Decision variables
    X = opti.variable(n_X, n_stp + 1); // State trajectory    
    U = opti.variable(n_U, n_stp);     // Control trajectory (torque)
    dt = opti.variable();               // Time horizon
    
    // Parameters
    p_X0 = opti.parameter(n_X);           // Parameter for initial state
    p_Xf = opti.parameter(n_X);           // Parameter for final state
    
    //// Consraints ////

    // Box constraints
    opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));    // Time step constraints
    opti.subject_to(opti.bounded(lb_U, U, ub_U));       // Control constraints

    // Control input constraints
    if (cons.Udot) {
        delta_U = U(Slice(), Slice(1, n_stp)) - U(Slice(), Slice(0, n_stp - 1)); 
        delta_U = reshape(delta_U, 3*(n_stp-1), 1); // Flatten to vector
        delta_U_max = dt * tau_dot_max * MX::ones(3*(n_stp-1), 1); // Also a vector
        opti.subject_to(opti.bounded(-delta_U_max, delta_U, delta_U_max)); // Control input rate of change
    }

    // Dynamics constraints
    MX X_kp1; // Next state
    for (int k = 0; k < n_stp; ++k) {
        // Integrate dynamics
        X_kp1 = F({X(all,k), U(all,k), dt})[0]; // Call the dynamics function
        opti.subject_to(X(all,k+1) == X_kp1); // Enforce the discretized dynamics
    };


    // Initial and final state constraints
    opti.subject_to(X(all,0) == p_X0);     // Initial condition
    opti.subject_to(X(all,n_stp) == p_Xf); // Final condition

    //// Initial guess ////
    opti.set_initial(dt, dt_0); // Initial guess for the time horizon

    //// Objective ////
    opti.minimize(dt);

    ///// Solver ////

    opti.solver("ipopt", {{"expand", false}}); // Set numerical backend

    solver = opti.to_function("solver",
        {p_X0, p_Xf},                      // Inputs
        {X, U, dt},                         // Outputs
        {"X0", "Xf"},                      // Input names
        {"X", "U", "dt"}                    // Output names
    );
}

// std::tuple<DM, DM, DM, DM> Optimizer::solve() {

//     auto sol = opti.solve();
//     DM X_sol = sol.value(X);
//     DM U_sol = sol.value(U);
//     DM T_sol = sol.value(T);
//     DM dt_sol = T_sol / n_stp;
//     std::cout << "dt: " << dt_sol << " s" << std::endl;
//     std::cout << "T: " << T_sol << " s" << std::endl;

//     return std::make_tuple(X_sol, U_sol, T_sol, dt_sol);
// };

Function get_solver() {
        // library prefix and full name
        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_solver.so";

        // use this function
        return external("solver", lib_full_name);
}

OptiCvodes::OptiCvodes(const Function& dyn, const Constraints& cons) :
    F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
    X_0(cons.X_0), X_f(cons.X_f) {

    // Decision variables
    X = opti.variable(n_X, n_stp + 1); // State trajectory    
    U = opti.variable(n_U, n_stp);     // Control trajectory (torque)
    dt = opti.variable(n_stp);               // Time horizon
    //dt = T / n_stp;               // Time step
    // Parameters
    p_X0 = opti.parameter(n_X);           // Parameter for initial state
    p_Xf = opti.parameter(n_X);           // Parameter for final state
    
    //// Consraints ////

    // Box constraints
    opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));    // Time step constraints
    opti.subject_to(opti.bounded(lb_U, U, ub_U));       // Control constraints
    //opti.subject_to(X(0, all) >= MX::zeros(1, n_stp + 1)); // Ensure q0 >= 0 to pick a hemisphere

    // Dynamics constraints
    MX X_kp1; // Next state
    MX T{0}; // Cost function
    MXDict F_out;
    for (int k = 0; k < n_stp; ++k) {
        // Integrate dynamics
        F_out = F(MXDict{{"x0", X(all,k)}, {"u", U(all,k)}, {"p", dt(k)}});
        X_kp1 = F_out.at("xf");
        T += dt(k);
        opti.subject_to(X(all,k+1) == X_kp1); // Enforce the discretized dynamics
        opti.subject_to(dot(X(Slice(0,4),k+1),X(Slice(0,4),k+1)) == 1); // Ensure |q| = 1 

    };

    // Initial and final state constraints
    opti.subject_to(X(all,0) == p_X0);     // Initial condition
    opti.subject_to(X(all,n_stp) == p_Xf); // Final condition

    //// Initial guess ////
    opti.set_initial(dt, dt_0*DM::ones(n_stp)); // Initial guess for the time horizon
    std::cout << "dt_0: " << dt_0 << std::endl;
    opti.set_initial(X, stateInterpolator(X_0, X_f, n_stp+1)); // Initial guess for the initial state
    opti.set_initial(U, inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp)); // Initial guess for the control input

    //// Objective ////
    opti.minimize(T);

    ///// Solver ////
    opti.solver("ipopt"); // Set numerical backend

    solver = opti.to_function("solver",
        {p_X0, p_Xf},                      // Inputs
        {X, U, T},                        // Outputs
        {"X0", "Xf"},                      // Input names
        {"X", "U", "J"}                   // Output names
    );
}

DM stateInterpolator(const DM& x0, const DM& xf, int n_stp) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_stp < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = 4; 
    DM q(n, n_stp);
    Slice all;

    for (int i = 0; i < n_stp; ++i) {
        double alpha = static_cast<double>(i) / (n_stp - 1);
        q(all, i) = (1.0 - alpha) * x0(Slice(0, n)) + alpha * xf(Slice(0, n));
    }

    DM omega = inputInterpolator(x0(Slice(1, n)), xf(Slice(1, n)), n_stp);

    DM result = DM::vertcat({q, omega});

    return result;
}

DM inputInterpolator(const auto& x0, const auto& xf, int n_stp) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_stp < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = x0.size1(); 
    DM result(n, n_stp);
    Slice all;

    for (int i = 0; i < n_stp/2; ++i) {
        double alpha = static_cast<double>(i) / (n_stp/2 - 1);
        result(all, i) = sign(xf-x0) * alpha;
        result(all, n_stp - 1 - i) = result(all, i);
    }
    return result;
}