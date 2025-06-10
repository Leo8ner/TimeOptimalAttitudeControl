#include <toac/optimizer.h>

using namespace casadi;

CUDAOptimizer::CUDAOptimizer(const Constraints& cons) :
        lb_U(cons.lb_U), ub_U(cons.ub_U), lb_dt(cons.lb_dt), ub_dt(cons.ub_dt),
        X_0(cons.X_0), X_f(cons.X_f) {
        
        // Create CUDA dynamics function
        cuda_dynamics = CUDADynamicsCallback::create_function(
            "cuda_dynamics", n_X, n_U);

        setupOptimizationProblem();
}
    
void CUDAOptimizer::setupOptimizationProblem() {
    // Decision variables (same as your original)
    X = opti.variable(n_X, n_stp + 1);
    U = opti.variable(n_U, n_stp);
    dt = opti.variable(n_stp);
    
    // Parameters
    p_X0 = opti.parameter(n_X);
    p_Xf = opti.parameter(n_X);
    
    // Box constraints (same as original)
    opti.subject_to(dt > 0);
    opti.subject_to(opti.bounded(lb_U, U, ub_U));
    
    // Quaternion normalization constraint
    opti.subject_to(sum1(pow(X(Slice(0,4), Slice()), 2)) == 1);
    
    // Boundary conditions
    opti.subject_to(X(Slice(), 0) == p_X0);
    opti.subject_to(X(Slice(), n_stp) == p_Xf);
    
    // CUDA-accelerated dynamics constraints
    setupCUDADynamicsConstraints();
    
    // Initial guess
    opti.set_initial(dt, dt_0 * DM::ones(n_stp));
    opti.set_initial(X, stateInterpolator(X_0, X_f, n_stp + 1));
    opti.set_initial(U, inputInterpolator(X_0(Slice(1,4)), X_f(Slice(1,4)), n_stp));
    
    // Objective
    MX T = sum(dt);
    opti.minimize(T);
    
    // Solver setup
    Dict plugin_opts{}, solver_opts{};
    solver_opts["print_level"] = 3;
    solver_opts["max_iter"] = 1000;
    
    opti.solver("ipopt", plugin_opts, solver_opts);
    
    solver = opti.to_function("cuda_solver",
                            {p_X0, p_Xf},
                            {X, U, T, dt},
                            {"X0", "Xf"},
                            {"X", "U", "T", "dt"});
}

void CUDAOptimizer::setupCUDADynamicsConstraints() {
    
    // Apply dynamics constraint using CUDA function
    MX X_next = cuda_dynamics({X(Slice(), Slice(0, n_stp)), U, dt})[0];
    opti.subject_to(X(Slice(), Slice(1, n_stp + 1)) == X_next);
}
