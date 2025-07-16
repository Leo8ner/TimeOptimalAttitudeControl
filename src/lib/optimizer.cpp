#include <toac/optimizer.h>

using namespace casadi;

Function get_solver() {
        // library prefix and full name
        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_solver.so";

        // use this function
        return external("solver", lib_full_name);
}

Optimizer::Optimizer(const Function& dyn, const Constraints& cons, const std::string& plugin, 
                     bool fixed_step) :
    F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
    X_0(cons.X_0), X_f(cons.X_f) {

    // Parameters
    p_X0 = opti.parameter(n_states);           // Parameter for initial state
    p_Xf = opti.parameter(n_states);           // Parameter for final state

    Dict plugin_opts{}, solver_opts{};

    // Initial guesses
    DM X_guess = stateInterpolator(X_0, X_f, n_stp + 1);
    DM U_guess = inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp);
    
    if (plugin == "fatrop") {

        // Reconstruct matrices for function interface compatibility
        X = MX::zeros(n_states, n_stp + 1);
        U = MX::zeros(n_controls, n_stp);
        T = MX::zeros(1);
        dt = MX::zeros(n_stp);
        
        // Create variables.
        std::vector<MX> x(n_stp + 1);
        std::vector<MX> u(n_stp);
        std::vector<MX> delta_t(n_stp + 1);
        
        for (int k = 0; k < n_stp; ++k) {
            x[k] = opti.variable(n_states);
            delta_t[k] = opti.variable();
            u[k] = opti.variable(n_controls);
        }
        x[n_stp] = opti.variable(n_states);  // Final state variable
        delta_t[n_stp] = opti.variable();  // Final time step variable

        for (int k = 0; k < n_stp; ++k) {
            X(all, k) = x[k];
            U(all, k) = u[k];
            dt(k) = delta_t[k];
        }
        X(all, n_stp) = x[n_stp];

        for (int k = 0; k < n_stp; ++k) {
            // Dynamics
            //MX X_kp1 = F(MXDict{{"x0", x[k]}, {"u", u[k](Slice(0, n_controls))}, {"p", u[k](n_controls)}}).at("xf");
            MX X_kp1 = F({x[k],u[k],delta_t[k]})[0];
            opti.subject_to(x[k+1] == X_kp1);  // State at next time step
            if (fixed_step) {
                opti.subject_to(delta_t[k+1] == delta_t[k]);  // Fixed time step
            }
            // Path constraints
            if (k == 0) {
                opti.subject_to(x[k] == p_X0);  // Initial state
            } else {
                opti.subject_to(sum1(pow(x[k](Slice(0,4)), 2)) == 1);  // Quaternion norm
            }
            opti.subject_to(lb_U <= u[k](Slice(0, n_controls)) <= ub_U);  // Control upper bounds
            opti.subject_to(0 < delta_t[k] );  // dt > 0
        }
        
        // Final time step path constraints
        opti.subject_to(x[n_stp] == p_Xf);  // Final state
        
        // Objective function
        for (int k = 0; k < n_stp; ++k) {
            T += delta_t[k];
        }
        
        // Initial guesses
        for (int k = 0; k < n_stp; ++k) {
            opti.set_initial(x[k], X_guess(all, k));
            opti.set_initial(delta_t[k], dt_0);  
            opti.set_initial(u[k], U_guess(all, k));
        }
        opti.set_initial(x[n_stp], X_guess(all, n_stp));
        
        // Solver configuration
        plugin_opts = {
            {"expand", true},
            {"structure_detection", "auto"},
            {"debug", true}
        };
        solver_opts = {
            {"print_level", 0},
            {"tol", 1e-16},              // Main tolerance
            {"constr_viol_tol", 1e-16}, // Constraint violation tolerance
            {"mu_init", 1e-1},           // Larger initial barrier parameter
        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf},
            {X, U, T, dt},
            {"X0", "Xf"},
            {"X", "U", "T", "dt"}
        );
        
        
    } else if (plugin == "ipopt") {

        // Define variables
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        if (fixed_step) {
            T = opti.variable(1);  // Total time
            dt = MX::repmat(T/n_stp, n_stp, 1);
        } else {
            dt = opti.variable(n_stp);
            T = sum(dt);  // Total time        
        }

        MX X_kp1 = F(MXDict{{"x0", X(all,Slice(0, n_stp))}, {"u", U}, {"p", dt}}).at("xf");
        opti.subject_to(X(all,Slice(1, n_stp+1)) == X_kp1);
        opti.subject_to(sum1(pow(X(Slice(0,4),all), 2)) == 1);
        
        opti.subject_to(X(all,0) == p_X0);
        opti.subject_to(X(all,n_stp) == p_Xf);

        if (fixed_step) {
            opti.subject_to(T > 0);  // Fixed step size
        } else {
            //opti.subject_to(dt >= lb_dt);
            //opti.subject_to(dt <= ub_dt);
            opti.subject_to(dt>0);
        }
        opti.subject_to(opti.bounded(lb_U, U, ub_U));
        
        opti.set_initial(dt, dt_0*DM::ones(n_stp));
        opti.set_initial(X, X_guess);
        opti.set_initial(U, U_guess);

        solver_opts = {
            {"tol", 1e-10},              // Main tolerance
            {"acceptable_tol", 1e-7},    // Acceptable tolerance
            {"constr_viol_tol", 1e-6}, // Constraint violation tolerance

        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf},
            {X, U, T, dt},
            {"X0", "Xf"},
            {"X", "U", "T", "dt"}
        );
                
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }
}

DM stateInterpolator(const DM& x0, const DM& xf, int n_steps) {
    if (x0.size1() != xf.size1() || x0.size2() != xf.size2()) {
        throw std::invalid_argument("x0 and xf must have the same dimensions.");
    }
    if (n_steps < 2) {
        throw std::invalid_argument("Number of steps must be at least 2.");
    }

    int n = 4; 
    DM q = quaternionSlerp(x0(Slice(0, n)), xf(Slice(0, n)), n_steps);

    DM omega = ratesInterpolator(x0(Slice(1, n)), xf(Slice(1, n)), n_steps);

    DM result = DM::vertcat({q, omega});

    return result;
}

// Discretized quaternion spherical interpolation
DM quaternionSlerp(const auto& q1, const auto& q2, int n_steps) {
    if (n_steps <= 0) {
        throw std::invalid_argument("n_steps must be positive");
    }
    if (n_steps == 1) {
        DM result = DM::zeros(4, 1);
        result(Slice(), 0) = q1 / sqrt(sumsqr(q1));
        return result;
    }
    
    DM result = DM::zeros(4, n_steps);
    
    // Normalize input quaternions
    DM q1_norm = q1 / sqrt(sumsqr(q1));
    DM q2_norm = q2 / sqrt(sumsqr(q2));
    
    // Compute dot product
    double dot = static_cast<double>(mtimes(q1_norm.T(), q2_norm));
    
    // Choose shorter path
    if (dot < 0.0) {
        q2_norm = -q2_norm;
        dot = -dot;
    }    
    
    // Generate interpolated quaternions
    for (int i = 0; i < n_steps; ++i) {
        double t = static_cast<double>(i) / (n_steps - 1);
        
        // Initialize q_interp to avoid undefined behavior
        DM q_interp = q1_norm;  // Safe default initialization
        
        // Linear interpolation with normalization
        q_interp = (1.0 - t) * q1_norm + t * q2_norm;
        q_interp /= sqrt(sumsqr(q_interp));
        result(Slice(), i) = q_interp;
    }
    
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