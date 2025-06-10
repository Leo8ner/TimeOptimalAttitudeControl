#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <toac/symmetric_spacecraft.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/cuda.h>

using namespace casadi;

class ExplicitDynamics {

    SX X, U, dt;

public:
    Function F; 

    ExplicitDynamics();
};

struct ImplicitDynamics {
    
    Function F; 

    ImplicitDynamics();
};

// Takes a 3D vector w and returns a 4x4 skew-symmetric matrix
SX skew4(const SX& w);

// RK4 integrator
SX rk4(const SX& x_dot, const SX& x, const SX& dt);

using namespace casadi;



class CUDADynamicsCallback : public Callback {
    
    std::unique_ptr<CUDADynamics> cuda_integrator;
    int n_X;        // Number of states per shooting node
    int n_U;        // Number of controls per shooting node

public:
    // Constructor
    CUDADynamicsCallback(const std::string& name, int n_states, int n_controls);
    
    // Destructor
    ~CUDADynamicsCallback() = default;
    
    // Required: Get number of inputs
    casadi_int get_n_in() override;
    
    // Required: Get number of outputs  
    casadi_int get_n_out() override;

    // Required: Main numerical evaluation function
    std::vector<DM> eval(const std::vector<DM>& arg) const override;

    // Required: Main symbolic evaluation function
    std::vector<MX> eval(const std::vector<MX>& arg) const;
    
    // Factory method to create the callback
    static Function create_function(const std::string& name, int n_states, int n_controls);
};

#endif // DYNAMICS_H