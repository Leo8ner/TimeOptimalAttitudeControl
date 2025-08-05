#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/cuda_dynamics.h>
#include <toac/constraints.h>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters
    DM X_guess, U_guess, dt_guess;     // Initial guesses
    std::string csv_file;              // CSV file for initial guess

    void setupOptimizationProblem();
    void extractInitialGuess();


public:

    Function solver; // Solver function
    Optimizer(const Function& dyn, const Constraints& cons, const std::string& csv_data = "");

};

Function get_solver();

// DM stateInterpolator(const DM& x0, const DM& xf, int n_stp);
// DM inputInterpolator(const auto& x0, const auto& xf, int n_stp);
// DM ratesInterpolator(const auto& x0, const auto& xf, int n_stp);
// DM quaternionSlerp(const auto& q1, const auto& q2, int n_steps);


class BatchDynamics {

    Slice all;

public:
    BatchDynamics();
    Function F;
};

// Takes a 3D vector w and returns a 4x4 skew-symmetric matrix
SX skew4(const SX& w);

// RK4 integrator
SX rk4(const SX& x_dot, const SX& x, const SX& dt);



#endif // OPTIMIZER_H