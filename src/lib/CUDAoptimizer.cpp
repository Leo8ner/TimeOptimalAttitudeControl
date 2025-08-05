#include <toac/cuda_optimizer.h>

using namespace casadi;

Optimizer::Optimizer(const Function &dyn, const Constraints &cons, const std::string& csv_data) : F(dyn), lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
                                                                     csv_file(csv_data) 
{

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
        
        // Initial guess
        if (!csv_file.empty()) {
            extractInitialGuess();
            opti.set_initial(X, X_guess);
            opti.set_initial(U, U_guess);
        } else {
            // Default initial guess
            // X_guess = stateInterpolator(X_0, X_f, n_stp + 1);
            // U_guess = inputInterpolator(X_0(Slice(1, 4)), X_f(Slice(1, 4)), n_stp);
            dt_guess = DM::repmat(dt_0, 1, n_stp);
        }
        opti.set_initial(dt, dt_guess);

        // Objective
        T = sum(dt);
        opti.minimize(T);

        // Solver setup
        Dict plugin_opts{}, solver_opts{};
        solver_opts["print_level"] = 5;
        //solver_opts["max_iter"] = 1000;
        solver_opts["tol"] = 1e-10;            // Main tolerance
        solver_opts["acceptable_tol"] = 1e-6;  // Acceptable tolerance
        solver_opts["constr_viol_tol"] = 1e-6; // Constraint violation tolerance
        //solver_opts["jacobian_approximation"] = "finite-difference-values"; // Use sparse Jacobian approximation
        solver_opts["hessian_approximation"] = "limited-memory"; // Use limited-memory approximation
        plugin_opts["expand"] = true;


        opti.solver("ipopt", plugin_opts, solver_opts);

        solver = opti.to_function("parsolver",
                                  {p_X0, p_Xf},
                                  {X, U, T, dt},
                                  {"X0", "Xf"},
                                  {"X", "U", "T", "dt"});
}

void Optimizer::extractInitialGuess() {
    // Read CSV file
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + csv_file);
    }
    
    // Load file content
    std::string csv_content((std::istreambuf_iterator<char>(file)), 
                            std::istreambuf_iterator<char>());
    file.close();
    
    // Parse CSV data with header detection
    std::istringstream stream(csv_content);
    std::string line;
    std::vector<std::vector<double>> x_data, u_data, dt_data;
    std::string current_section = "";
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        // Check if line is a header
        if (line == "X" || line == "U" || line == "T" || line == "dt") {
            current_section = line;
            continue;
        }
        
        // Parse numeric data
        std::vector<double> row;
        std::istringstream line_stream(line);
        std::string cell;
        
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        
        // Store data in appropriate section
        if (current_section == "X") {
            x_data.push_back(row);
        } else if (current_section == "U") {
            u_data.push_back(row);
        } else if (current_section == "dt") {
            dt_data.push_back(row);
        }
        // Skip "T" section as it's not needed
    }
    
    // Extract X (rows 3-9, which are the last 7 rows)
    int n_cols = x_data[0].size();
    X_guess = DM::zeros(7, n_cols);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < n_cols; j++) {
            X_guess(i, j) = x_data[3 + i][j];  // Start from row 3
        }
    }
    
    // Extract U (all 3 rows)
    U_guess = DM::zeros(3, n_cols-1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < n_cols-1; j++) {
            U_guess(i, j) = u_data[i][j];
        }
    }
    
    // Extract dt (1 row)
    dt_guess = DM::zeros(1, n_cols-1);
    for (int j = 0; j < n_cols-1; j++) {
        dt_guess(0, j) = dt_data[0][j];
    }
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