#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>
#include <iostream>
#include <sundials/sundials_types.h>
#include <cvode/cvode.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <cuda_runtime.h>
#include <memory>

using namespace casadi;

const int n_X{7};                     // Number of states
const int n_U{3};                     // Number of controls

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters

public:

    Function solver; // Solver function
    Optimizer(const Function& dyn, const Constraints& cons);
};

// Modified Optimizer class with CUDA integration
class CUDAOptimizer {
    Opti opti;
    MX X, U, dt;
    MX p_X0, p_Xf;
    
    // CUDA integrator
    Function cuda_dynamics; // Dynamics functions
    
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    
public:
    Function solver;
    CUDAOptimizer(const Constraints& cons);
    
private:
    void setupOptimizationProblem();
    
    void setupCUDADynamicsConstraints();
};

Function get_solver();

DM stateInterpolator(const DM& x0, const DM& xf, int n_stp);
DM inputInterpolator(const auto& x0, const auto& xf, int n_stp);
DM ratesInterpolator(const auto& x0, const auto& xf, int n_stp);

#endif // OPTIMIZER_H