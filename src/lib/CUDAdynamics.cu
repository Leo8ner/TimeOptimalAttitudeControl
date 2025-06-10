#include <toac/cuda.h>
    
// CUDA kernel for attitude dynamics
__global__ void attitude_dynamics_kernel(sunrealtype* y_dot, const sunrealtype* y, 
                                         const sunrealtype* u, const sunrealtype* dt_vec, 
                                         int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_states/7) {
        int base_idx = idx * 7;
        
        // Extract state variables (quaternion + angular velocity)
        sunrealtype q0 = y[base_idx + 0];  // Scalar part of quaternion
        sunrealtype q1 = y[base_idx + 1];  // i component
        sunrealtype q2 = y[base_idx + 2];  // j component
        sunrealtype q3 = y[base_idx + 3];  // k component
        sunrealtype wx = y[base_idx + 4];  // Angular velocity x
        sunrealtype wy = y[base_idx + 5];  // Angular velocity y
        sunrealtype wz = y[base_idx + 6];  // Angular velocity z
        
        // Control inputs (torques) for this shooting node
        sunrealtype tau_x = u[idx * 3 + 0];
        sunrealtype tau_y = u[idx * 3 + 1];
        sunrealtype tau_z = u[idx * 3 + 2];
        
        // Time step for this shooting node
        sunrealtype dt = dt_vec[idx];
        
        // Quaternion dynamics: q_dot = 0.5 * Omega(w) * q
        // Where Omega(w) is the skew-symmetric matrix of angular velocity
        y_dot[base_idx + 0] = dt * 0.5 * (-q1*wx - q2*wy - q3*wz);
        y_dot[base_idx + 1] = dt * 0.5 * ( q0*wx - q3*wy + q2*wz);
        y_dot[base_idx + 2] = dt * 0.5 * ( q3*wx + q0*wy - q1*wz);
        y_dot[base_idx + 3] = dt * 0.5 * (-q2*wx + q1*wy + q0*wz);
        
        // Angular velocity dynamics: I*w_dot = tau - w x (I*w)
        const sunrealtype I_X = i_x, I_Y = i_y, I_Z = i_z; 
        const sunrealtype I_X_INV = 1.0/I_X, I_Y_INV = 1.0/I_Y, I_Z_INV = 1.0/I_Z;
        
        // Calculate I*w (moment of inertia times angular velocity)
        sunrealtype Iw_x = I_X * wx;
        sunrealtype Iw_y = I_Y * wy; 
        sunrealtype Iw_z = I_Z * wz;
        
        // Cross product: w x (I*w) - gyroscopic effects
        sunrealtype cross_x = wy * Iw_z - wz * Iw_y;
        sunrealtype cross_y = wz * Iw_x - wx * Iw_z;
        sunrealtype cross_z = wx * Iw_y - wy * Iw_x;
        
        // Apply Euler's rotational equation: w_dot = I^(-1) * (tau - w x (I*w))
        y_dot[base_idx + 4] = dt * I_X_INV * (tau_x - cross_x);
        y_dot[base_idx + 5] = dt * I_Y_INV * (tau_y - cross_y);
        y_dot[base_idx + 6] = dt * I_Z_INV * (tau_z - cross_z);
    }
}

// SUNDIALS RHS function - called by the integrator
int CUDADynamics::attitude_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    // Extract user data containing control inputs and time steps
    auto* data = static_cast<std::pair<sunrealtype*, sunrealtype*>*>(user_data);
    sunrealtype *u = data->first;      // Control inputs
    sunrealtype *dt_vec = data->second; // Time steps
    
    // Get device pointers to SUNDIALS vectors
    sunrealtype *y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype *ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
    
    // Calculate problem size
    sunindextype N = N_VGetLength_Cuda(y);
    int n_states = N / 7;  // Each state has 7 components (4 quaternion + 3 angular velocity)
    
    // Configure CUDA kernel launch parameters
    int blockSize = 256;
    int numBlocks = (n_states + blockSize - 1) / blockSize;
    
    // Launch CUDA kernel to compute derivatives
    attitude_dynamics_kernel<<<numBlocks, blockSize>>>(
        ydot_data, y_data, u, dt_vec, N
    );
    
    // Check for CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }
    
    // Synchronize to ensure kernel completion (optional for debugging)
    // cudaDeviceSynchronize();
    
    return 0;  // Success
}

// Constructor - initializes CUDA memory and SUNDIALS solver
CUDADynamics::CUDADynamics() : sunctx(nullptr), y(nullptr), 
                                         d_control_input(nullptr), d_dt_input(nullptr), cvode_mem(nullptr) {
    // Initialize SUNDIALS context (required for v6.0+)
    int flag = SUNContext_Create(NULL, &sunctx);
    if (flag != SUN_SUCCESS) {
        throw std::runtime_error("Failed to create SUNDIALS context");
    }
    
    // Set CUDA device
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to set CUDA device");
    }
    
    // Create SUNDIALS CUDA vector (7 states per shooting node)
    sunindextype N = 7 * n_stp;
    y = N_VNew_Cuda(N, sunctx);
    if (y == nullptr) {
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to create SUNDIALS CUDA vector");
    }
    
    // Allocate GPU memory for control inputs and time steps
    cuda_status = cudaMalloc(&d_control_input, 3 * n_stp * sizeof(sunrealtype));
    if (cuda_status != cudaSuccess) {
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to allocate GPU memory for control inputs");
    }
    
    cuda_status = cudaMalloc(&d_dt_input, n_stp * sizeof(sunrealtype));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_control_input);
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to allocate GPU memory for time steps");
    }
    
    // Initialize CVODE solver with context
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (cvode_mem == nullptr) {
        cudaFree(d_dt_input);
        cudaFree(d_control_input);
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to create CVODE solver");
    }
    
    // Initialize the solver with RHS function
    flag = CVodeInit(cvode_mem, attitude_rhs, 0.0, y);
    if (flag != CV_SUCCESS) {
        CVodeFree(&cvode_mem);
        cudaFree(d_dt_input);
        cudaFree(d_control_input);
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to initialize CVODE");
    }
    
    // Set tolerances (relative and absolute)
    flag = CVodeSStolerances(cvode_mem, 1e-8, 1e-10);
    if (flag != CV_SUCCESS) {
        CVodeFree(&cvode_mem);
        cudaFree(d_dt_input);
        cudaFree(d_control_input);
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        throw std::runtime_error("Failed to set CVODE tolerances");
    }
    
    // Set user data as pair of control and dt pointers
    static std::pair<sunrealtype*, sunrealtype*> user_data_pair;
    user_data_pair = {d_control_input, d_dt_input};
    CVodeSetUserData(cvode_mem, &user_data_pair);
}

// Destructor - cleanup memory
CUDADynamics::~CUDADynamics() {
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (y) N_VDestroy(y);
    if (d_control_input) cudaFree(d_control_input);
    if (d_dt_input) cudaFree(d_dt_input);
    if (sunctx) SUNContext_Free(&sunctx);  // Free SUNDIALS context last
}

// Main integration function - integrates multiple shooting nodes in parallel
int CUDADynamics::integrate_parallel(const std::vector<sunrealtype>& initial_states,
                                    const std::vector<sunrealtype>& controls,
                                    const std::vector<sunrealtype>& dt_values,
                                    std::vector<sunrealtype>& final_states) {
    
    // Validate input sizes
    if (initial_states.size() != 7 * n_stp) {
        std::cerr << "Error: Initial states size mismatch" << std::endl;
        return -1;
    }
    if (controls.size() != 3 * n_stp) {
        std::cerr << "Error: Controls size mismatch" << std::endl;
        return -1;
    }
    if (dt_values.size() != n_stp) {
        std::cerr << "Error: Time steps size mismatch" << std::endl;
        return -1;
    }
    
    // Copy initial states to SUNDIALS vector
    sunrealtype *y_data = N_VGetHostArrayPointer_Cuda(y);
    for (size_t i = 0; i < initial_states.size(); i++) {
        y_data[i] = initial_states[i];
    }
    N_VCopyToDevice_Cuda(y);  // Transfer to GPU
    
    // Copy control inputs and time steps to GPU
    cudaError_t cuda_status;
    cuda_status = cudaMemcpy(d_control_input, controls.data(), 
                            controls.size() * sizeof(sunrealtype), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error copying controls to GPU" << std::endl;
        return -1;
    }
    
    cuda_status = cudaMemcpy(d_dt_input, dt_values.data(),
                            dt_values.size() * sizeof(sunrealtype), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error copying time steps to GPU" << std::endl;
        return -1;
    }
    
    // Reinitialize solver for new integration
    int flag = CVodeReInit(cvode_mem, 0.0, y);
    if (flag != CV_SUCCESS) {
        std::cerr << "Error reinitializing CVODE" << std::endl;
        return flag;
    }
    
    // Integrate from t=0 to t=1 (dt is built into the dynamics)
    sunrealtype t = 0.0;
    flag = CVode(cvode_mem, 1.0, y, &t, CV_NORMAL);
    if (flag < 0) {
        std::cerr << "Integration failed with flag: " << flag << std::endl;
        return flag;
    }
    
    // Copy results back to host
    N_VCopyFromDevice_Cuda(y);
    final_states.resize(7 * n_stp);
    for (size_t i = 0; i < final_states.size(); i++) {
        final_states[i] = y_data[i];
    }
    
    return 0;  // Success
}