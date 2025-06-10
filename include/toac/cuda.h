#ifndef TOAC_CUDA_H
#define TOAC_CUDA_H

#include <toac/symmetric_spacecraft.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <sundials/sundials_types.h>
#include <sundials/sundials_context.h>
#include <cvode/cvode.h>
#include <nvector/nvector_cuda.h>
#include <cuda_runtime.h>
#include <memory>

// CUDA-accelerated dynamics integrator class
class CUDADynamics {

    void* cvode_mem;
    N_Vector y;
    sunrealtype *d_control_input;
    sunrealtype *d_dt_input;
    SUNContext sunctx;
    
    // SUNDIALS RHS function
    static int attitude_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);
    
public:
    CUDADynamics();
    
    ~CUDADynamics();
    
    // Integrate multiple shooting nodes in parallel
    int integrate_parallel(const std::vector<sunrealtype>& initial_states,
                         const std::vector<sunrealtype>& controls,
                         const std::vector<sunrealtype>& dt_values,
                         std::vector<sunrealtype>& final_states);
};

// CUDA kernel for attitude dynamics
__global__ void attitude_dynamics_kernel(sunrealtype* y_dot, const sunrealtype* y, 
                                         const sunrealtype* u, const sunrealtype* dt_vec, 
                                         int n_states);

#endif // TOAC_CUDA_H