#pragma once
#include <casadi/casadi.hpp>
#include <toac/cuda_optimizer.h>
#include <toac/cuda_dynamics.h>

using namespace casadi;

class CUDACallback : public Callback {
private:
    std::unique_ptr<OptimizedDynamicsIntegrator> integrator;
    
public:
    CUDACallback(const std::string& name);

    // Input/output dimensions
    casadi_int get_n_in() override;
    casadi_int get_n_out() override;
    
    std::string get_name_in(casadi_int i) override;
    
    
    std::string get_name_out(casadi_int i) override;
    
    Sparsity get_sparsity_in(casadi_int i) override;
    
    Sparsity get_sparsity_out(casadi_int i) override;

    // Main evaluation function - calls batch CUDA integrator
    DMVector eval(const DMVector& arg) const override;

};