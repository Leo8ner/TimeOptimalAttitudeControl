#include <toac/cuda_optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function &dyn) : F(dyn), lb_dt(dt_min), ub_dt(dt_max)
                                                                     
{
    lb_U = DM::vertcat({-tau_max, -tau_max, -tau_max});             // Lower bound for torque
    ub_U = DM::vertcat({ tau_max,  tau_max,  tau_max});             // Upper bound for torque
    setupOptimizationProblem();
}

void Optimizer::setupOptimizationProblem()
{
        // Decision variables
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        dt = opti.variable(n_stp);

        // Parameters
        p_X0 = opti.parameter(n_states);
        p_Xf = opti.parameter(n_states);

        // Box constraints
        opti.subject_to(dt > 0);
        opti.subject_to(opti.bounded(lb_U, U, ub_U));

        // Quaternion normalization constraint
        opti.subject_to(sum1(pow(X(Slice(0, 4), Slice()), 2)) == 1);

        // Boundary conditions
        opti.subject_to(X(Slice(), 0) == p_X0);
        opti.subject_to(X(Slice(), n_stp) == p_Xf);

        // CUDA dynamics constraint
        MX X_current = X(Slice(), Slice(0, n_stp));
        MX X_next_computed = F(MXVector{X_current, U, dt})[0];
        MX X_next_expected = X(Slice(), Slice(1, n_stp + 1));
        opti.subject_to(X_next_expected == X_next_computed);
        opti.subject_to(dt(Slice(0, n_stp - 1)) == dt(Slice(1, n_stp)));

        // Objective
        T = sum(dt);
        opti.minimize(T);

        // Solver setup
        Dict plugin_opts{}, solver_opts{};
        solver_opts["print_level"] = 5;
        // solver_opts["max_iter"] = 1000;
        solver_opts["tol"] = 1e-7;            // Main tolerance
        // solver_opts["acceptable_tol"] = 1e-6;  // Acceptable tolerance
        //solver_opts["jacobian_approximation"] = "finite-difference-values"; // Use sparse Jacobian approximation
        solver_opts["hessian_approximation"] = "limited-memory"; // Use limited-memory approximation
        plugin_opts["expand"] = true;


        opti.solver("ipopt", plugin_opts, solver_opts);

        solver = opti.to_function("parsolver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt},
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"});
}


// Constructor implementation
BatchDynamics::BatchDynamics() {
    MX X = MX::vertcat({MX::sym("q", 4, n_stp), MX::sym("w", 3, n_stp)});
    MX U = MX::sym("tau", 3, n_stp);
    MX dt = MX::sym("dt", n_stp);
    
    MX q = X(Slice(0, 4), all);
    MX w = X(Slice(4, 7), all);
    
    // Compute quaternion derivatives for all time steps
    MX q_dot = MX::zeros(4, n_stp);
    for (int i = 0; i < n_stp; i++) {
        MX w_i = w(all, i);
        MX S_i = skew4(w_i);
        q_dot(all, i) = 0.5 * MX::mtimes(S_i, q(all, i));
    }
    
    // Inertia matrices
    MX I = MX::diag(MX::vertcat({i_x, i_y, i_z}));
    MX I_inv = MX::diag(MX::vertcat({1.0/i_x, 1.0/i_y, 1.0/i_z}));
    
    // Compute angular velocity derivatives for all time steps
    MX w_dot = MX::zeros(3, n_stp);
    for (int i = 0; i < n_stp; i++) {
        MX w_i = w(all, i);
        MX U_i = U(all, i);
        MX Iw = MX::mtimes(I, w_i);
        w_dot(all, i) = MX::mtimes(I_inv, (U_i - cross(w_i, Iw)));
    }
    
    MX X_dot = MX::vertcat({q_dot, w_dot});
    
    // Apply RK4 integration for each time step
    MX X_next = MX::zeros(7, n_stp);
    for (int i = 0; i < n_stp; i++) {
        MX X_i = X(all, i);
        MX X_dot_i = X_dot(all, i);
        MX dt_i = dt(i);
        X_next(all, i) = rk4(X_dot_i, X_i, dt_i);
    }
    
    F = Function("F", {X, U, dt}, {X_next});
}