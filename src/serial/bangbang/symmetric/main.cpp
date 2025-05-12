#include <casadi/casadi.hpp>
//#include <toac/optimizer.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>

using namespace casadi;

int main() {

    // Dynamics
    Dynamics dyn; // Create an instance of the Dynamics class

    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    //Optimizer opti{dyn, cons};          // Create an instance of the Optimizer class

    return 0;
}