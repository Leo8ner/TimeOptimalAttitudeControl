#include <toac/cuda_optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function &dyn, const Constraints &cons) : F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
                                                                     X_0(cons.X_0), X_f(cons.X_f)
{

        setupOptimizationProblem();
}

void Optimizer::setupOptimizationProblem()
{
        // Decision variables (same as your original)
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
        
        // Initial guess
        DM X_guess = stateInterpolator(X_0, X_f, n_stp + 1);
        DM U_guess = inputInterpolator(X_0(Slice(1, 4)), X_f(Slice(1, 4)), n_stp);
        opti.set_initial(dt, dt_0 * DM::ones(n_stp));
        opti.set_initial(X, X_guess);
        opti.set_initial(U, U_guess);

        // Objective
        T = sum(dt);
        opti.minimize(T);

        // Solver setup
        Dict plugin_opts{}, solver_opts{};
        solver_opts["print_level"] = 5;
        //solver_opts["max_iter"] = 1000;
        solver_opts["tol"] = 1e-6;            // Main tolerance
        solver_opts["acceptable_tol"] = 1e-6;  // Acceptable tolerance
        solver_opts["constr_viol_tol"] = 1e-6; // Constraint violation tolerance
        //solver_opts["jacobian_approximation"] = "finite-difference-values"; // Use sparse Jacobian approximation
        solver_opts["hessian_approximation"] = "limited-memory"; // Use limited-memory approximation
        //plugin_opts["expand"] = true;


        opti.solver("ipopt", plugin_opts, solver_opts);

        solver = opti.to_function("parsolver",
                                  {p_X0, p_Xf},
                                  {X, U, T, dt},
                                  {"X0", "Xf"},
                                  {"X", "U", "T", "dt"});
}

// Constructor implementation
BatchDynamics::BatchDynamics() {
    SX X = SX::vertcat({SX::sym("q", 4, n_stp), SX::sym("w", 3, n_stp)});
    SX U = SX::sym("tau", 3, n_stp);
    SX dt = SX::sym("dt", n_stp);
    
    SX q = X(Slice(0, 4), all);
    SX w = X(Slice(4, 7), all);
    
    // Compute quaternion derivatives for all time steps
    SX q_dot = SX::zeros(4, n_stp);
    for (int i = 0; i < n_stp; i++) {
        SX w_i = w(all, i);
        SX S_i = skew4(w_i);
        q_dot(all, i) = 0.5 * SX::mtimes(S_i, q(all, i));
    }
    
    // Inertia matrices
    SX I = SX::diag(SX::vertcat({i_x, i_y, i_z}));
    SX I_inv = SX::diag(SX::vertcat({1.0/i_x, 1.0/i_y, 1.0/i_z}));
    
    // Compute angular velocity derivatives for all time steps
    SX w_dot = SX::zeros(3, n_stp);
    for (int i = 0; i < n_stp; i++) {
        SX w_i = w(all, i);
        SX U_i = U(all, i);
        SX Iw = SX::mtimes(I, w_i);
        w_dot(all, i) = SX::mtimes(I_inv, (U_i - cross(w_i, Iw)));
    }
    
    SX X_dot = SX::vertcat({q_dot, w_dot});
    
    // Apply RK4 integration for each time step
    SX X_next = SX::zeros(7, n_stp);
    for (int i = 0; i < n_stp; i++) {
        SX X_i = X(all, i);
        SX X_dot_i = X_dot(all, i);
        SX dt_i = dt(i);
        X_next(all, i) = rk4(X_dot_i, X_i, dt_i);
    }
    
    F = Function("F", {X, U, dt}, {X_next});
}