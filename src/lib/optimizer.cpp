#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
//#include <toac/optimizer.h>

using namespace casadi;
/*
class Optimizer() {

private:
    int n_X, n_U, n_stp;
    double t_0;
    Function f, F;
    DM x_0, x_f;
    DM lb_U, ub_U, lb_dt, ub_dt;

public:
    Optimizer(const Optimizer&) = delete; // Disable copy constructor
    Optimizer& operator=(const Optimizer&) = delete; // Disable copy assignment
    Optimizer(Optimizer&&) = default; // Enable move constructor
    Optimizer& operator=(Optimizer&&) = default; // Enable move assignment
    Optimizer(int n_X, int n_U, double T_0, int n_stp, Function f, Function F, DM X_0, DM X_f, DM lb_U, DM ub_U, DM lb_dt, DM ub_dt) :
        n_X(n_X), n_U(N_U), t_0(T_0), n_stp(n_stp), f(f), F(F), x_0(X_0), x_f(X_f),
        lb_U(lb_U), ub_U(ub_U), lb_dt(lb_dt), ub_dt(ub_dt) {
        // Constructor implementation

        }
    ~Optimizer() = default; // Default destructor
};

    Opti opti {Opti()};                   // Optimization problem
    Slice all;                            // Equivalent to the slice operation in Python

    // ---- decision variables ---------
    MX X {opti.variable(n_X, n_stp + 1)};   // state trajectory    
    MX U {opti.variable(n_U, n_stp)};       // control trajectory (torque)
    MX T {opti.variable()};               // final time for time optimal control


    // -------- objective ------------
    opti.minimize(T);

    // ---- dynamic constraints --------
    SX dt = T / n_stp; // time step
    for (int k = 0; k < n_stp; ++k) {
        opti.subject_to(X(all,k+1) == F(X(all,k), U(all,k), dt)); // Enforce the discretized dynamics
    }
}*/