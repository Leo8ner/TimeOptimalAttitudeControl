#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/helper_functions.h>

using namespace casadi;

struct Constraints {
    // DM X_0, X_f;
    DM lb_U, ub_U;
    DM lb_dt, ub_dt;
    bool Udot {false};                    // Control input constraints

    Constraints();
    void setUdot();                      // Set the constraints for the control input
};


#endif // CONSTRAINTS_H