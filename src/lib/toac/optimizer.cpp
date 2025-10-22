#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Dynamics& dyn, bool fixed_step) :
    F(dyn.F), lb_dt(dt_min), ub_dt(dt_max), plugin(dyn.plugin), method(dyn.method) {

    lb_U = DM::vertcat({tau_min, tau_min, tau_min});             // Lower bound for torque
    ub_U = DM::vertcat({tau_max, tau_max, tau_max});             // Upper bound for torque

    lb_X = DM::vertcat({q_min, q_min, q_min, q_min,
                        w_min, w_min, w_min});      // Lower bound for states
    ub_X = DM::vertcat({q_max, q_max, q_max, q_max,
                        w_max, w_max, w_max});      // Upper bound for states

    // Parameters
    p_X0 = opti.parameter(n_states);           // Parameter for initial state
    p_Xf = opti.parameter(n_states);           // Parameter for final state

    Dict plugin_opts{}, solver_opts{};
    
    if (plugin == "fatrop") {
        
        // Create variables.
        std::vector<MX> x(n_stp + 1);
        std::vector<MX> u(n_stp);
        std::vector<MX> delta_t(n_stp);
        
        for (int k = 0; k < n_stp; ++k) {
            x[k] = opti.variable(n_states);
            delta_t[k] = opti.variable();
            u[k] = opti.variable(n_controls);
        }
        x[n_stp] = opti.variable(n_states);  // Final state variable

        // Concatenate variables for input and output
        X = x[0];
        U = u[0];
        dt = delta_t[0];
        for (int k = 1; k < n_stp; ++k) {
            X = MX::horzcat({X, x[k]});
            U = MX::horzcat({U, u[k]});
            dt = MX::vertcat({dt, delta_t[k]});
        }
        X = MX::horzcat({X, x[n_stp]});

        for (int k = 0; k < n_stp; ++k) {
            // Dynamics
            if (method == "shooting") {
                MX X_kp1 = F({x[k],u[k],delta_t[k]})[0];
                opti.subject_to(x[k+1] == X_kp1);  // State at next time step

            // } else if (method == "collocation") {
            //     // Evaluate dynamics at start of interval
            //     MX f_k = F(MXVector{x[k], u[k]})[0];
                
            //     // For endpoint dynamics, handle last interval specially
            //     MX u_kp1;
            //     if (k < n_stp - 1) {
            //         u_kp1 = u[k+1];  // Use next control point
            //     } else {
            //         u_kp1 = u[k];    // Last interval: extrapolate (piecewise constant)
            //     }
            //     MX f_k1 = F(MXVector{x[k+1], u_kp1})[0];
                
            //     // Compute midpoint state and control
            //     MX x_c = 0.5*(x[k] + x[k+1]) + delta_t[k]/8.0*(f_k - f_k1);
            //     MX u_c = 0.5*(u[k] + u_kp1);
                
            //     // Evaluate dynamics at midpoint
            //     MX f_c = F(MXVector{x_c, u_c})[0];
                
            //     // Simpson's rule defect constraint
            //     MX defect = x[k+1] - x[k] - delta_t[k]/6.0*(f_k + 4*f_c + f_k1);
            //     opti.subject_to(defect == 0);

            } else {
                throw std::invalid_argument("Unsupported method: " + method);
            }

            if (fixed_step && k < n_stp - 1) {
                opti.subject_to(delta_t[k+1] == delta_t[k]);  // Fixed time step
            }
            // Path constraints
            if (k == 0) {
                opti.subject_to(x[k] == p_X0);  // Initial state
            } else {
                opti.subject_to(sum1(pow(x[k](Slice(0,4)), 2)) == 1);  // Quaternion norm
            }
            opti.subject_to(lb_X <= x[k] <= ub_X);  // State bounds
            opti.subject_to(lb_U <= u[k] <= ub_U);  // Control upper bounds
            opti.subject_to(lb_dt <= delta_t[k] <= ub_dt);  // dt > 0
        }
        
        // Final time step path constraints
        opti.subject_to(x[n_stp] == p_Xf);  // Final state
        
        // Objective function
        T = MX::zeros(1);
        for (int k = 0; k < n_stp; ++k) {
            T += delta_t[k];
        }

        // Solver configuration
        plugin_opts = {
            {"expand", true},
            {"structure_detection", "auto"},
            //{"debug", true}
        };
        solver_opts = {
            {"print_level", 5},
            {"tol", 1e-7},              // Main tolerance
            //{"constr_viol_tol", 1e-7}, // Constraint violation tolerance
            //{"acceptable_tol", 1e-5},    // Acceptable tolerance
            //{"mu_init", 1e-1},           // Larger initial barrier parameter
        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);
        

        solver = opti.to_function("solver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt},
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"} // Add output names
        );
        
        
    } else if (plugin == "ipopt" || plugin == "snopt") {

        // Define variables
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        dt = opti.variable(n_stp);
        T = sum(dt);  // Total time        

        // MX X_kp1 = F(MXDict{{"x0", X(all,Slice(0, n_stp))}, {"u", U}, {"p", dt}}).at("xf");
        if (method == "shooting") {
            MX X_kp1 = F({X(all,Slice(0, n_stp)), U, dt})[0];
            opti.subject_to(X(all,Slice(1, n_stp+1)) == X_kp1);
        } else if (method == "collocation") {
            
            // Slice states and controls
            MX x_k = X(all, Slice(0, n_stp));
            MX x_kp1 = X(all, Slice(1, 1 + n_stp));
            MX u_k = U;
            
            // Handle control at k+1 (repeat last control for final interval)
            MX u_kp1 = MX::horzcat({U(all, Slice(1, n_stp)), U(all, n_stp-1)});
            
            // Evaluate dynamics at endpoints (vectorized)
            MX f_k = F(MXVector{x_k, u_k})[0];
            MX f_k1 = F(MXVector{x_kp1, u_kp1})[0];
            
            // Compute midpoint state with proper broadcasting
            // Need to broadcast dt across columns: each column scaled by corresponding dt[k]
            MX dt_broadcast = repmat(dt.T(), n_states, 1);  // Shape: (n_states x n_stp)
            MX x_c = 0.5*(x_k + x_kp1) + (dt_broadcast/8.0) * (f_k - f_k1);
            MX u_c = 0.5*(u_k + u_kp1);
            
            // Evaluate dynamics at midpoint (vectorized)
            MX f_c = F(MXVector{x_c, u_c})[0];
            
            // Simpson's rule defect constraint
            MX defect = x_kp1 - x_k - (dt_broadcast/6.0) * (f_k + 4*f_c + f_k1);
            opti.subject_to(defect == 0);

        } else {
            throw std::invalid_argument("Unsupported method: " + method);
        }

        opti.subject_to(sum1(pow(X(Slice(0,4),all), 2)) == 1);
        
        opti.subject_to(X(all,0) == p_X0);
        opti.subject_to(X(all,n_stp) == p_Xf);

        if (fixed_step) {
            opti.subject_to(dt(Slice(0, n_stp - 1)) == dt(Slice(1, n_stp)));
        }
        opti.subject_to(opti.bounded(lb_X, X, ub_X));
        opti.subject_to(opti.bounded(lb_U, U, ub_U));
        opti.subject_to(opti.bounded(lb_dt, dt, ub_dt));

        // Solver configuration
        plugin_opts = {
            {"expand", true},
        };
        solver_opts = {
            {"print_level", 5},
            {"warm_start_init_point", "yes"},
            {"max_iter", 1000},
            //{"linear_solver", "ma57"},
            {"mu_strategy", "adaptive"},
            {"tol", 1e-7},              // Main tolerance
            //{"acceptable_tol", 1e-8},    // Acceptable tolerance
            //{"constr_viol_tol", 1e-6}, // Constraint violation tolerance
            //{"hessian_approximation", "limited-memory"}, // Use limited-memory approximation

        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt}, // Add stats
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"} // Add output names
        );
         
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }
}

// void Optimizer::SetDynamicConstraints() {
//     // This function can be used to set or update dynamic constraints if needed
//     if (plugin != "fatrop" && method == "shooting") {
//         // Implement shooting method constraints
//     } else if (plugin == "fatrop" && method == "shooting") {
//         MX X_kp1 = F({x[k],u[k],delta_t[k]})[0];
//         opti.subject_to(x[k+1] == X_kp1);  // State at next time step
//     } else if (plugin == "fatrop" && method == "collocation") {
//         // Implement collocation constraints if needed
//     } else if (plugin != "ipopt" && method == "collocation") {
//         // Implement collocation constraints if needed
//     } else {
//         throw std::invalid_argument("Unsupported method: " + method);
// }