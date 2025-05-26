#include <toac/optimizer.h>

using namespace casadi;

Function get_solver() {
        // library prefix and full name
        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_solver.so";

        // use this function
        return external("solver", lib_full_name);
}

Optimizer::Optimizer(const Function& dyn, const Constraints& cons) :
    F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
    X_0(cons.X_0), X_f(cons.X_f) {

    // Decision variables
    X = opti.variable(n_X, n_stp + 1); // State trajectory    
    U = opti.variable(n_U, n_stp);     // Control trajectory (torque)
    dt = opti.variable(n_stp);               // Time horizon

    // Parameters
    p_X0 = opti.parameter(n_X);           // Parameter for initial state
    p_Xf = opti.parameter(n_X);           // Parameter for final state
    
    //// Consraints ////

    // Box constraints
    //opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));    // Time step constraints
    opti.subject_to(dt>0);    // Time step constraints
    opti.subject_to(opti.bounded(lb_U, U, ub_U));       // Control constraints

    // Dynamics constraints
    MX X_kp1; // Next state
    for (int k = 0; k < n_stp; ++k) {
        // Integrate dynamics
        X_kp1 = F(MXDict{{"x0", X(all,k)}, {"u", U(all,k)}, {"p", dt(k)}}).at("xf");
        opti.subject_to(X(all,k+1) == X_kp1); // Enforce the discretized dynamics
        opti.subject_to(dot(X(Slice(0,4),k+1),X(Slice(0,4),k+1)) == 1); // Ensure |q| = 1 

    };

    // Initial and final state constraints
    opti.subject_to(X(all,0) == p_X0);     // Initial condition
    opti.subject_to(X(all,n_stp) == p_Xf); // Final condition

    //// Initial guess ////
    opti.set_initial(dt, dt_0*DM::ones(n_stp)); // Initial guess for the time horizon
    opti.set_initial(X, stateInterpolator(X_0, X_f, n_stp+1)); // Initial guess for the initial state
    opti.set_initial(U, inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp)); // Initial guess for the control input

    //// Objective ////
    MX T = sum(dt); // Objective function
    opti.minimize(T);

    ///// Solver ////
    Dict plugin_opts{}, solver_opts{};
    opti.solver("ipopt", plugin_opts, solver_opts); // Set numerical backend

    solver = opti.to_function("solver",
        {p_X0, p_Xf},                      // Inputs
        {X, U, T, dt},                        // Outputs
        {"X0", "Xf"},                      // Input names
        {"X", "U", "T", "dt"}                   // Output names
    );
}

DM stateInterpolator(const DM& x0, const DM& xf, int n_steps) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_steps < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = 4; 
    DM q(n, n_steps);
    Slice all;

    for (int i = 0; i < n_steps; ++i) {
        double alpha = static_cast<double>(i) / (n_steps - 1);
        q(all, i) = (1.0 - alpha) * x0(Slice(0, n)) + alpha * xf(Slice(0, n));
    }
    DM omega = ratesInterpolator(x0(Slice(1, n)), xf(Slice(1, n)), n_steps);

    DM result = DM::vertcat({q, omega});

    return result;
}

DM ratesInterpolator(const auto& x0, const auto& xf, int n_steps) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_steps < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = x0.size1(); 
    DM result(n, n_steps);
    Slice all;
    auto sgn = sign(xf - x0); // Sign function to handle direction
    for (int i = 0; i < n_steps/2; ++i) {
        double alpha = static_cast<double>(i) / (n_steps/2 - 1);
        result(all, i) = sgn * alpha;
        result(all, n_steps - 1 - i) = result(all, i);
    }

    return result;
}

DM inputInterpolator(const auto& x0, const auto& xf, int n_steps) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_steps < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = x0.size1(); 
    DM result(n, n_steps);
    Slice all;

    result(all, Slice(0, n_steps/2)) = repmat(sign(xf-x0), 1, n_steps/2);
    result(all, Slice(n_steps/2, n_steps)) = repmat(-sign(xf-x0), 1, ceil(n_steps/2.0));

    return result;
}