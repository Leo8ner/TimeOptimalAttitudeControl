#include <toac/optimizer.h>

using namespace casadi;

Function get_solver() {
        // library prefix and full name
        std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
        std::string lib_full_name = prefix_lib + "lib_solver.so";

        // use this function
        return external("solver", lib_full_name);
}

Optimizer::Optimizer(const Function& dyn, const Constraints& cons, const std::string& plugin) :
    F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
    X_0(cons.X_0), X_f(cons.X_f) {

    // Parameters
    p_X0 = opti.parameter(n_states);           // Parameter for initial state
    p_Xf = opti.parameter(n_states);           // Parameter for final state

    Dict plugin_opts{}, solver_opts{};
    
    if (plugin == "fatrop") {
        // Define structure arrays for Fatrop
        std::vector<int> nx(n_stp + 1, n_states);  // 12 states at each time step
        std::vector<int> nu(n_stp + 1, 0);         // Control + dt variables
        std::vector<int> ng(n_stp + 1, 0);         // Path constraints
        
        for (int k = 0; k < n_stp; ++k) {
            nu[k] = n_controls + 1;  // 4 controls + 1 dt variable
            ng[k] = 1;  // Quaternion normalization constraint
        }
        nu[n_stp] = 0;  // No controls at final time step
        ng[n_stp] = 1;  // Quaternion normalization constraint at final step
        
        // Create decision variables for each time step
        std::vector<MX> x(n_stp + 1);
        std::vector<MX> u(n_stp + 1);
        
        for (int k = 0; k <= n_stp; ++k) {
            x[k] = opti.variable(nx[k]);
            u[k] = opti.variable(nu[k]);
        }
        
        // Extract variables for compatibility with existing code
        X = MX::zeros(n_states, n_stp + 1);
        U = MX::zeros(n_controls, n_stp);
        dt = MX::zeros(n_stp);
        
        for (int k = 0; k <= n_stp; ++k) {
            X(all, k) = x[k];
            if (k < n_stp) {
                U(all, k) = u[k](Slice(0, n_controls));
                dt(k) = u[k](n_controls);
            }
        }
        
        // Constraints in required order: dynamics, path constraints
        for (int k = 0; k < n_stp; ++k) {
            // Discrete dynamics constraint
            MX X_kp1 = F(MXDict{{"x0", x[k]}, {"u", u[k](Slice(0, n_controls))}, {"p", u[k](n_controls)}}).at("xf");
            opti.subject_to(x[k+1] == X_kp1);
            
            // Path constraints
            opti.subject_to(sum1(pow(x[k](Slice(0,4)), 2)) == 1);  // Quaternion normalization
            opti.subject_to(u[k](n_controls) > 0);  // Time step > 0
            opti.subject_to(opti.bounded(lb_U, u[k](Slice(0, n_controls)), ub_U));  // Control bounds
        }
        
        // Path constraint at final time step
        opti.subject_to(sum1(pow(x[n_stp](Slice(0,4)), 2)) == 1);  // Quaternion normalization
        
        // Initial and final conditions
        opti.subject_to(x[0] == p_X0);
        opti.subject_to(x[n_stp] == p_Xf);
        
        // Objective function
        MX T = MX::zeros(1);
        for (int k = 0; k < n_stp; ++k) {
            T += u[k](n_controls);
        }
        opti.minimize(T);
        
        // Initial guess
        DM X_guess = stateInterpolator(X_0, X_f, n_stp+1);
        DM U_guess = inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp);
        
        for (int k = 0; k <= n_stp; ++k) {
            opti.set_initial(x[k], X_guess(all, k));
            if (k < n_stp) {
                DM u_init = DM::zeros(n_controls + 1);
                u_init(Slice(0, n_controls)) = U_guess(all, k);
                u_init(n_controls) = dt_0;
                opti.set_initial(u[k], u_init);
            }
        }
        
        // Solver configuration
        plugin_opts = {
            {"expand", false},
            {"structure_detection", "manual"},
            {"nx", nx},
            {"nu", nu}, 
            {"ng", ng},
            {"N", n_stp},
            {"debug", true}
        };
        solver_opts = {       
            {"print_level", 4}
        };
        
        
    } else if (plugin == "ipopt") {
        // Keep original IPOPT implementation
        X = opti.variable(n_states, n_stp + 1);
        U = opti.variable(n_controls, n_stp);
        dt = opti.variable(n_stp);
        
        MX T = sum(dt);
        opti.minimize(T);
        
        DM X_guess = stateInterpolator(X_0, X_f, n_stp+1);
        DM U_guess = inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp);
        
        MX X_kp1 = F(MXDict{{"x0", X(all,Slice(0, n_stp))}, {"u", U}, {"p", dt}}).at("xf");
        opti.subject_to(X(all,Slice(1, n_stp+1)) == X_kp1);
        opti.subject_to(sum1(pow(X(Slice(0,4),all), 2)) == 1);
        
        opti.subject_to(X(all,0) == p_X0);
        opti.subject_to(X(all,n_stp) == p_Xf);
        
        opti.subject_to(dt>0);
        opti.subject_to(opti.bounded(lb_U, U, ub_U));
        
        opti.set_initial(dt, dt_0*DM::ones(n_stp));
        opti.set_initial(X, X_guess);
        opti.set_initial(U, U_guess);
                
    } else {
        throw std::invalid_argument("Unsupported solver type: " + plugin);
    }

    opti.solver(plugin, plugin_opts, solver_opts);

    
    solver = opti.to_function("solver",
        {p_X0, p_Xf},
        {X, U, T, dt},
        {"X0", "Xf"},
        {"X", "U", "T", "dt"}
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