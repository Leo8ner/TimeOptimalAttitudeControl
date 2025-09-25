#include <toac/optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function& dyn, const std::string& plugin, 
                     bool fixed_step) :
    F(dyn), lb_dt(dt_min), ub_dt(dt_max) {

    lb_U = DM::vertcat({-tau_max, -tau_max, -tau_max});             // Lower bound for torque
    ub_U = DM::vertcat({ tau_max,  tau_max,  tau_max});             // Upper bound for torque

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
        MX X = x[0];
        MX U = u[0];
        MX dt = delta_t[0];
        for (int k = 1; k < n_stp; ++k) {
            X = MX::horzcat({X, x[k]});
            U = MX::horzcat({U, u[k]});
            dt = MX::vertcat({dt, delta_t[k]});
        }
        X = MX::horzcat({X, x[n_stp]});

        for (int k = 0; k < n_stp; ++k) {
            // Dynamics
            MX X_kp1 = F({x[k],u[k],delta_t[k]})[0];
            opti.subject_to(x[k+1] == X_kp1);  // State at next time step
            if (fixed_step && k < n_stp - 1) {
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
            {"print_level", 0},
            //{"tol", 1e-16},              // Main tolerance
            //{"constr_viol_tol", 1e-16}, // Constraint violation tolerance
            //{"mu_init", 1e-1},           // Larger initial barrier parameter
        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt},
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"}
        );
        
        
    } else if (plugin == "ipopt") {

        // Define variables
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        dt = opti.variable(n_stp);
        T = sum(dt);  // Total time        

        // MX X_kp1 = F(MXDict{{"x0", X(all,Slice(0, n_stp))}, {"u", U}, {"p", dt}}).at("xf");
        MX X_kp1 = F({X(all,Slice(0, n_stp)), U, dt})[0];
        opti.subject_to(X(all,Slice(1, n_stp+1)) == X_kp1);
        opti.subject_to(sum1(pow(X(Slice(0,4),all), 2)) == 1);
        
        opti.subject_to(X(all,0) == p_X0);
        opti.subject_to(X(all,n_stp) == p_Xf);

        if (fixed_step) {
            opti.subject_to(dt(Slice(0, n_stp - 1)) == dt(Slice(1, n_stp)));
        }
        opti.subject_to(opti.bounded(lb_U, U, ub_U));
        opti.subject_to(dt > 0);  

        // Solver configuration
        plugin_opts = {
            {"expand", true},
        };
        solver_opts = {
            {"print_level", 0},
            //{"tol", 1e-10},              // Main tolerance
            //{"acceptable_tol", 1e-8},    // Acceptable tolerance
            //{"constr_viol_tol", 1e-6}, // Constraint violation tolerance
            //{"hessian_approximation", "limited-memory"}, // Use limited-memory approximation

        };

        // Set the objective function
        opti.minimize(T);
        opti.solver(plugin, plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt},
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"}
        );

    } else if (plugin == "qpoases") {

        // Define variables (similar to IPOPT approach)
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        dt = opti.variable(n_stp);
        T = sum(dt);  // Total time        

        // Dynamics constraints
        MX X_kp1 = F({X(all,Slice(0, n_stp)), U, dt})[0];
        opti.subject_to(X(all,Slice(1, n_stp+1)) == X_kp1);
        
        // Quaternion norm constraints
        opti.subject_to(sum1(pow(X(Slice(0,4),all), 2)) == 1);
        
        // Boundary conditions
        opti.subject_to(X(all,0) == p_X0);     // Initial state
        opti.subject_to(X(all,n_stp) == p_Xf); // Final state

        // Fixed time step constraint (if specified)
        if (fixed_step) {
            opti.subject_to(dt(Slice(0, n_stp - 1)) == dt(Slice(1, n_stp)));
        }
        
        // Control and time constraints
        opti.subject_to(opti.bounded(lb_U, U, ub_U));
        opti.subject_to(dt > 0);  

    Dict qpsol_opts{
        {"solver", "ipopt"},
    };
        // qpOASES-specific plugin options
   plugin_opts = {
        {"qpsol", "nlpsol"},                  // QP solver
        {"qpsol_options", qpsol_opts},       // Use qpOASES
        //{"structure_detection", "auto"},
    };

    solver_opts = {
        //{"nlpsol", "fatrop"},                  // QP solver
        // {"codegen", true},                   // Enable code generation

    };
        // Set the objective function
        opti.minimize(T);
        opti.solver("sqpmethod", plugin_opts, solver_opts);

        solver = opti.to_function("solver",
            {p_X0, p_Xf, X, U, dt},
            {X, U, T, dt},
            {"X0", "Xf", "X_guess", "U_guess", "dt_guess"},
            {"X", "U", "T", "dt"}
        );
                
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }
}