#include "test.h"

using namespace std;

//==============================================================================
// CUDA CONSTANT MEMORY FOR SPACECRAFT PARAMETERS
//==============================================================================

/**
 * @brief Device constant memory for spacecraft inertia and common constants
 * 
 * Constant memory is cached and provides fast access across all GPU threads.
 * Layout: [Ix, Iy, Iz, 0.5, 1/Ix, 1/Iy, 1/Iz, Iz-Iy, Ix-Iz, Iy-Ix, -0.5, 0.0]
 */
__device__ __constant__ sunrealtype d_inertia_constants[12];

//==============================================================================
// GPU KERNELS FOR SPACECRAFT DYNAMICS
//==============================================================================

/**
 * @brief GPU kernel for computing spacecraft dynamics right-hand side
 * 
 * Computes time derivatives for quaternion kinematics and angular velocity dynamics
 * using Euler's equations for rigid body rotation. Each thread processes one complete
 * spacecraft system.
 * 
 * Mathematical Model:
 * - Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
 * - Euler's equations: İω̇ = τ - ω × (İω)
 * - Time scaling: All derivatives multiplied by dt for unit-time integration
 * 
 * Memory Access Pattern:
 * - Coalesced reads from global memory for state variables
 * - Cached access to constant memory for inertia parameters
 * - Coalesced writes to global memory for derivatives
 * 
 * @param n_total Total number of state variables across all spacecraft systems
 * @param y Input state vector [q0,q1,q2,q3,wx,wy,wz] per system (device memory)
 * @param ydot Output time derivative vector (device memory)
 * @param torque_params Applied torques [τx,τy,τz] per system (device memory)
 * @param dt_params Integration time steps per system (device memory)
 */
__global__ void dynamicsRHS(int n_total, sunrealtype* y, sunrealtype* ydot, 
                            sunrealtype* torque_params, sunrealtype* dt_params) {
    // Calculate system index from thread and block indices
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys < 0 || sys >= n_stp) return;

    // Load precomputed inertia constants from cached constant memory
    const sunrealtype Ix = d_inertia_constants[0];      // Moment of inertia about x-axis
    const sunrealtype Iy = d_inertia_constants[1];      // Moment of inertia about y-axis
    const sunrealtype Iz = d_inertia_constants[2];      // Moment of inertia about z-axis
    const sunrealtype half = d_inertia_constants[3];    // Constant 0.5 for quaternion kinematics
    const sunrealtype Ix_inv = d_inertia_constants[4];  // 1/Ix for efficient division
    const sunrealtype Iy_inv = d_inertia_constants[5];  // 1/Iy for efficient division
    const sunrealtype Iz_inv = d_inertia_constants[6];  // 1/Iz for efficient division

    // Calculate base index for this system's state variables
    int base_idx = sys * n_states;
    if (base_idx + n_states > n_total) return;
    
    // Extract individual state components for readability
    sunrealtype q0 = y[base_idx + 0], q1 = y[base_idx + 1];  // Quaternion components
    sunrealtype q2 = y[base_idx + 2], q3 = y[base_idx + 3];
    sunrealtype wx = y[base_idx + 4], wy = y[base_idx + 5], wz = y[base_idx + 6];  // Angular velocities

    // Load control parameters for this system
    sunrealtype tau_x = torque_params[sys * n_controls + 0];  // Torque about x-axis
    sunrealtype tau_y = torque_params[sys * n_controls + 1];  // Torque about y-axis
    sunrealtype tau_z = torque_params[sys * n_controls + 2];  // Torque about z-axis
    sunrealtype dt    = dt_params[sys];                       // Time step for this system
    
    // Quaternion kinematic equations: q̇ = 0.5 * Ω(ω) * q
    // where Ω(ω) is the skew-symmetric matrix of angular velocity
    ydot[base_idx + 0] = half * (-q1*wx - q2*wy - q3*wz) * dt;  // q0_dot
    ydot[base_idx + 1] = half * ( q0*wx - q3*wy + q2*wz) * dt;  // q1_dot
    ydot[base_idx + 2] = half * ( q3*wx + q0*wy - q1*wz) * dt;  // q2_dot
    ydot[base_idx + 3] = half * (-q2*wx + q1*wy + q0*wz) * dt;  // q3_dot

    // Euler's equations: İω̇ = τ - ω × (İω)
    // Pre-compute angular momentum components for efficiency
    sunrealtype Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;
    
    // Compute angular acceleration including gyroscopic terms
    ydot[base_idx + 4] = Ix_inv * (tau_x - (wy * Iw_z - wz * Iw_y)) * dt;  // wx_dot
    ydot[base_idx + 5] = Iy_inv * (tau_y - (wz * Iw_x - wx * Iw_z)) * dt;  // wy_dot
    ydot[base_idx + 6] = Iz_inv * (tau_z - (wx * Iw_y - wy * Iw_x)) * dt;  // wz_dot
}

/**
 * @brief GPU kernel for computing sparse Jacobian matrix blocks
 * 
 * Computes analytical Jacobian ∂f/∂y for each spacecraft system using the exact
 * derivatives of the dynamics equations. The Jacobian has a specific sparsity
 * pattern due to the coupling structure in spacecraft dynamics.
 * 
 * Sparsity Pattern (per 7×7 block):
 * - Quaternion rows (0-3): Couple to all variables (full rows)
 * - Angular velocity rows (4-6): Only couple to angular velocities (gyroscopic terms)
 * 
 * Storage: Block-diagonal CSR format with exactly 37 non-zero entries per block
 * 
 * @param n_blocks Number of spacecraft systems (matrix blocks)
 * @param block_data Sparse Jacobian matrix data array (device memory)
 * @param y Current state vector for Jacobian evaluation (device memory)
 * @param dt_params Time step parameters for each system (device memory)
 */
__global__ void sparseJacobian(int n_blocks, sunrealtype* block_data, 
                               sunrealtype* y, sunrealtype* dt_params) {
    // One thread block per spacecraft system
    int block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    
    // Calculate pointer to this block's Jacobian data
    sunrealtype* block_jac = block_data + block_id * nnz;
    int base_state_idx = block_id * n_states;
    
    // Load current state variables for Jacobian evaluation
    sunrealtype q0 = y[base_state_idx + 0], q1 = y[base_state_idx + 1];
    sunrealtype q2 = y[base_state_idx + 2], q3 = y[base_state_idx + 3];
    sunrealtype wx = y[base_state_idx + 4], wy = y[base_state_idx + 5], wz = y[base_state_idx + 6];
    
    // Load time step for this system
    sunrealtype dt = dt_params[block_id];

    // Load constants from constant memory
    const sunrealtype half = d_inertia_constants[3];           // 0.5
    const sunrealtype Ix_inv = d_inertia_constants[4];         // 1/Ix
    const sunrealtype Iy_inv = d_inertia_constants[5];         // 1/Iy
    const sunrealtype Iz_inv = d_inertia_constants[6];         // 1/Iz
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];    // Iz - Iy
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];    // Ix - Iz
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];    // Iy - Ix
    const sunrealtype minus_half = d_inertia_constants[10];    // -0.5
    const sunrealtype zero = d_inertia_constants[11];          // 0.0
    
    // Fill Jacobian entries in CSR order (row-by-row, within each row by column)
    int idx = 0;
    
    // Quaternion kinematic Jacobian: ∂q̇/∂[q,ω] 
    // Each quaternion derivative couples to all state variables
    
    // Row 0: ∂q0_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = zero;                      // ∂q0_dot/∂q0 = 0
    block_jac[idx++] = minus_half * wx * dt;      // ∂q0_dot/∂q1 = -0.5*wx*dt
    block_jac[idx++] = minus_half * wy * dt;      // ∂q0_dot/∂q2 = -0.5*wy*dt
    block_jac[idx++] = minus_half * wz * dt;      // ∂q0_dot/∂q3 = -0.5*wz*dt
    block_jac[idx++] = minus_half * q1 * dt;      // ∂q0_dot/∂wx = -0.5*q1*dt
    block_jac[idx++] = minus_half * q2 * dt;      // ∂q0_dot/∂wy = -0.5*q2*dt
    block_jac[idx++] = minus_half * q3 * dt;      // ∂q0_dot/∂wz = -0.5*q3*dt
    
    // Row 1: ∂q1_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wx * dt;            // ∂q1_dot/∂q0 = 0.5*wx*dt
    block_jac[idx++] = zero;                      // ∂q1_dot/∂q1 = 0
    block_jac[idx++] = half * wz * dt;            // ∂q1_dot/∂q2 = 0.5*wz*dt
    block_jac[idx++] = minus_half * wy * dt;      // ∂q1_dot/∂q3 = -0.5*wy*dt
    block_jac[idx++] = half * q0 * dt;            // ∂q1_dot/∂wx = 0.5*q0*dt
    block_jac[idx++] = minus_half * q3 * dt;      // ∂q1_dot/∂wy = -0.5*q3*dt
    block_jac[idx++] = half * q2 * dt;            // ∂q1_dot/∂wz = 0.5*q2*dt
    
    // Row 2: ∂q2_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wy * dt;            // ∂q2_dot/∂q0 = 0.5*wy*dt
    block_jac[idx++] = minus_half * wz * dt;      // ∂q2_dot/∂q1 = -0.5*wz*dt
    block_jac[idx++] = zero;                      // ∂q2_dot/∂q2 = 0
    block_jac[idx++] = half * wx * dt;            // ∂q2_dot/∂q3 = 0.5*wx*dt
    block_jac[idx++] = half * q3 * dt;            // ∂q2_dot/∂wx = 0.5*q3*dt
    block_jac[idx++] = half * q0 * dt;            // ∂q2_dot/∂wy = 0.5*q0*dt
    block_jac[idx++] = minus_half * q1 * dt;      // ∂q2_dot/∂wz = -0.5*q1*dt
    
    // Row 3: ∂q3_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wz * dt;            // ∂q3_dot/∂q0 = 0.5*wz*dt
    block_jac[idx++] = half * wy * dt;            // ∂q3_dot/∂q1 = 0.5*wy*dt
    block_jac[idx++] = minus_half * wx * dt;      // ∂q3_dot/∂q2 = -0.5*wx*dt
    block_jac[idx++] = zero;                      // ∂q3_dot/∂q3 = 0
    block_jac[idx++] = minus_half * q2 * dt;      // ∂q3_dot/∂wx = -0.5*q2*dt
    block_jac[idx++] = half * q1 * dt;            // ∂q3_dot/∂wy = 0.5*q1*dt
    block_jac[idx++] = half * q0 * dt;            // ∂q3_dot/∂wz = 0.5*q0*dt
    
    // Angular velocity dynamics Jacobian: ∂ω̇/∂ω (gyroscopic coupling only)
    // Quaternions don't directly affect angular acceleration, so ∂ω̇/∂q = 0
    
    // Row 4: ∂wx_dot/∂[wx,wy,wz] (only non-zero angular velocity derivatives)
    block_jac[idx++] = zero;                                    // ∂wx_dot/∂wx = 0
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wz * dt;       // ∂wx_dot/∂wy (gyroscopic)
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy) * wy * dt;       // ∂wx_dot/∂wz (gyroscopic)
    
    // Row 5: ∂wy_dot/∂[wx,wy,wz]
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wz * dt;       // ∂wy_dot/∂wx (gyroscopic)
    block_jac[idx++] = zero;                                    // ∂wy_dot/∂wy = 0
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz) * wx * dt;       // ∂wy_dot/∂wz (gyroscopic)
    
    // Row 6: ∂wz_dot/∂[wx,wy,wz]
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wy * dt;       // ∂wz_dot/∂wx (gyroscopic)
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix) * wx * dt;       // ∂wz_dot/∂wy (gyroscopic)
    block_jac[idx++] = zero;                                    // ∂wz_dot/∂wz = 0
}

/**
 * @brief GPU kernel for computing forward sensitivity equations
 * 
 * Computes sensitivity time derivatives using the forward sensitivity method:
 * ṡ = ∂f/∂y * s + ∂f/∂p, where s = ∂y/∂p is the sensitivity vector.
 * 
 * This kernel uses shared memory to efficiently compute Jacobian-vector products
 * for all parameters of a given spacecraft system simultaneously.
 * 
 * Thread Organization:
 * - One thread block per spacecraft system
 * - Each thread handles one parameter type (initial condition, control, or time)
 * - Shared memory for broadcasting system state to all threads
 * 
 * @param Ns Total number of sensitivity parameters across all systems
 * @param y Current state vector (device memory)
 * @param yS_data_array Array of pointers to sensitivity vectors (device memory)
 * @param ySdot_data_array Array of pointers to sensitivity derivatives (device memory)
 * @param dt_params Time step parameters for each system (device memory)
 */
__global__ void sensitivityRHS(int Ns, sunrealtype* y, sunrealtype** yS_data_array, 
                              sunrealtype** ySdot_data_array, sunrealtype* dt_params) {
    int sys = blockIdx.x;        // One thread block per spacecraft system
    int param_type = threadIdx.x; // Parameter type within this system
    
    if (sys >= n_stp) return;
    
    // Shared memory for efficient broadcasting of system state to all threads
    __shared__ sunrealtype shared_state[n_states];
    __shared__ sunrealtype shared_dt;
    
    // Load system state into shared memory using first n_states threads
    if (param_type < n_states) {
        shared_state[param_type] = y[sys * n_states + param_type];
    }
    if (param_type == 0) {
        shared_dt = dt_params[sys];
    }
    __syncthreads();  // Ensure all threads see the loaded state
    
    // Decode global parameter index based on parameter type and system
    int global_param_idx;
    if (param_type < n_states) {
        // Initial condition parameter: ∂y/∂y₀
        global_param_idx = sys * n_states + param_type;
    } else if (param_type < n_states + n_controls) {
        // Control parameter: ∂y/∂u  
        global_param_idx = n_stp * n_states + sys * n_controls + (param_type - n_states);
    } else if (param_type == n_states + n_controls) {
        // Time step parameter: ∂y/∂dt
        global_param_idx = n_stp * (n_states + n_controls) + sys;
    } else {
        return; // Invalid parameter index
    }
    
    if (global_param_idx >= Ns) return;
    
    // Access sensitivity vectors for this parameter
    sunrealtype* yS_data = yS_data_array[global_param_idx];
    sunrealtype* ySdot_data = ySdot_data_array[global_param_idx];
    
    // Load current sensitivity state for this system
    sunrealtype s_state[n_states];
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        s_state[i] = yS_data[sys * n_states + i];
    }
    
    // Extract system state components from shared memory
    sunrealtype q0 = shared_state[0], q1 = shared_state[1];
    sunrealtype q2 = shared_state[2], q3 = shared_state[3];
    sunrealtype wx = shared_state[4], wy = shared_state[5], wz = shared_state[6];
    
    // Load constants for Jacobian computation
    const sunrealtype half = d_inertia_constants[3];           // 0.5
    const sunrealtype Ix_inv = d_inertia_constants[4];         // 1/Ix
    const sunrealtype Iy_inv = d_inertia_constants[5];         // 1/Iy
    const sunrealtype Iz_inv = d_inertia_constants[6];         // 1/Iz
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7];    // Iz - Iy
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8];    // Ix - Iz
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9];    // Iy - Ix

    // Compute Jacobian-vector product: ∂f/∂y * s
    sunrealtype Js[n_states];
    
    // Quaternion sensitivity derivatives (couple to all variables)
    Js[0] = half * (-wx*s_state[1] - wy*s_state[2] - wz*s_state[3] - q1*s_state[4] - q2*s_state[5] - q3*s_state[6]);
    Js[1] = half * (wx*s_state[0] + wz*s_state[2] - wy*s_state[3] + q0*s_state[4] - q3*s_state[5] + q2*s_state[6]);
    Js[2] = half * (wy*s_state[0] - wz*s_state[1] + wx*s_state[3] + q3*s_state[4] + q0*s_state[5] - q1*s_state[6]);
    Js[3] = half * (wz*s_state[0] + wy*s_state[1] - wx*s_state[2] - q2*s_state[4] + q1*s_state[5] + q0*s_state[6]);
    
    // Angular velocity sensitivity derivatives (gyroscopic coupling only)
    Js[4] = -Ix_inv * Iz_minus_Iy * wz * s_state[5] - Ix_inv * Iz_minus_Iy * wy * s_state[6];
    Js[5] = -Iy_inv * Ix_minus_Iz * wz * s_state[4] - Iy_inv * Ix_minus_Iz * wx * s_state[6];
    Js[6] = -Iz_inv * Iy_minus_Ix * wy * s_state[4] - Iz_inv * Iy_minus_Ix * wx * s_state[5];
    
    // Add direct parameter dependencies: ∂f/∂p terms
    if (param_type >= n_states && param_type < n_states + n_controls) {
        // Control parameter affects corresponding angular acceleration directly
        int torque_idx = param_type - n_states;
        if (torque_idx == 0) Js[4] += Ix_inv;      // ∂wx_dot/∂tau_x = 1/Ix
        else if (torque_idx == 1) Js[5] += Iy_inv; // ∂wy_dot/∂tau_y = 1/Iy
        else if (torque_idx == 2) Js[6] += Iz_inv; // ∂wz_dot/∂tau_z = 1/Iz
    } else if (param_type == n_states + n_controls) {
        // Time step parameter: ∂f/∂dt = f(y,u) (the original dynamics)
        Js[0] += half * (-wx*q1 - wy*q2 - wz*q3);
        Js[1] += half * (wx*q0 + wz*q2 - wy*q3);
        Js[2] += half * (wy*q0 - wz*q1 + wx*q3);
        Js[3] += half * (wz*q0 + wy*q1 - wx*q2);
        Js[4] += Ix_inv * Iz_minus_Iy * wy * wz;
        Js[5] += Iy_inv * Ix_minus_Iz * wx * wz;
        Js[6] += Iz_inv * Iy_minus_Ix * wx * wy;
    }
    
    // Store sensitivity derivatives for this system (scaled by time step)
    #pragma unroll
    for (int i = 0; i < n_states; i++) {
        ySdot_data[sys * n_states + i] = shared_dt * Js[i];
    }
}

//==============================================================================
// DYNAMICS INTEGRATOR IMPLEMENTATION
//==============================================================================


/**
 * @brief Constructor: Comprehensive GPU and SUNDIALS initialization
 * 
 * Performs all expensive one-time setup operations including:
 * - CUDA context and library handle creation
 * - Memory allocation for pinned host and device arrays
 * - SUNDIALS CVODES integrator configuration
 * - Sparse matrix structure setup
 * - Linear solver initialization
 * - Sensitivity analysis preparation (if enabled)
 * 
 * This constructor is designed for expensive setup with efficient repeated use.
 * 
 * @param enable_sensitivity Enable forward sensitivity analysis capability
 * @throws std::runtime_error on any initialization failure
 */
DynamicsIntegrator::DynamicsIntegrator(bool enable_sensitivity) : 
    n_total(n_states * n_stp), setup_time(0), solve_time(0),
    d_yS_ptrs(nullptr), d_ySdot_ptrs(nullptr), yS(nullptr), Ns(0), 
    sensitivity_enabled(enable_sensitivity), sens_was_setup(false),
    y0_row_idx(nullptr), y0_nnz(0), u_row_idx(nullptr), u_nnz(0), 
    dt_row_idx(nullptr), dt_nnz(0) {
    
    // Create CUDA events for precise timing of setup operations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    try {
        //======================================================================
        // Initialize CUDA constant memory with spacecraft parameters
        //======================================================================
        
        sunrealtype h_constants[12] = {
            i_x, i_y, i_z, 0.5,                    // [0-3]: Principal moments and 0.5
            1.0/i_x, 1.0/i_y, 1.0/i_z,             // [4-6]: Inverse inertias for efficiency
            (i_z - i_y), (i_x - i_z), (i_y - i_x), // [7-9]: Inertia differences (Euler terms)
            -0.5, 0.0                              // [10-11]: Additional constants
        };
        
        cudaError_t cuda_err = cudaMemcpyToSymbol(d_inertia_constants, h_constants, 
                                    12 * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error copying spacecraft parameters to constant memory: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Allocate device memory for control parameters and time steps
        //======================================================================
        
        cuda_err = cudaMalloc(&d_torque_params_ptr, 
                             n_stp * n_controls * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating device memory for torque parameters: " + 
                               string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMalloc(&d_dt_params_ptr, n_stp * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating device memory for time step parameters: " + 
                               string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Initialize SUNDIALS context and library handles
        //======================================================================
        
        if (SUNContext_Create(NULL, &sunctx) != 0) {
            throw runtime_error("Error creating SUNDIALS execution context");
        }
        
        // Create cuSPARSE handle for sparse matrix operations
        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            throw runtime_error("Error creating cuSPARSE handle: " + 
                                    to_string(cusparse_status));
        }
        
        // Create cuSOLVER handle for linear system solving
        cusolverStatus_t cusolver_status = cusolverSpCreate(&cusolver_handle);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            throw runtime_error("Error creating cuSOLVER handle: " + 
                                    to_string(cusolver_status));
        }

        //======================================================================
        // Create SUNDIALS vectors and allocate pinned host memory
        //======================================================================
        
        y = N_VNew_Cuda(n_total, sunctx);
        if (!y) {
            throw runtime_error("Error creating CUDA state vector");
        }
        
        // Allocate pinned host memory for fast GPU transfers
        cuda_err = cudaMallocHost(&h_y_pinned, n_total * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating pinned memory for states: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMallocHost(&h_tau_pinned, n_stp * n_controls * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating pinned memory for controls: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMallocHost(&h_dt_pinned, n_stp * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess) {
            throw runtime_error("Error allocating pinned memory for time steps: " + 
                                    string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Initialize CVODES integrator
        //======================================================================
        
        cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
        if (!cvode_mem) {
            throw runtime_error("Error creating CVODES integrator instance");
        }

        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != 0) {
            throw runtime_error("Error initializing CVODES: " + to_string(retval));
        }

        //======================================================================
        // Setup sparse Jacobian matrix and linear solver
        //======================================================================
        
        Jac = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, n_states, nnz, 
                                            cusparse_handle, sunctx);
        if (!Jac) {
            throw runtime_error("Error creating cuSPARSE block-diagonal matrix");
        }
        
        setupJacobianStructure();
        
        // Create batch QR linear solver optimized for block-diagonal systems
        LS = SUNLinSol_cuSolverSp_batchQR(y, Jac, cusolver_handle, sunctx);
        if (!LS) {
            throw runtime_error("Error creating cuSOLVER batch QR linear solver");
        }

        //======================================================================
        // Configure CVODES with tolerances and solver
        //======================================================================
        
        retval = CVodeSetUserData(cvode_mem, this);
        if (retval != 0) {
            throw runtime_error("Error setting CVODES user data: " + to_string(retval));
        }
        
        retval = CVodeSStolerances(cvode_mem, DEFAULT_RTOL, DEFAULT_ATOL);
        if (retval != 0) {
            throw runtime_error("Error setting CVODES tolerances: " + to_string(retval));
        }
        
        retval = CVodeSetLinearSolver(cvode_mem, LS, Jac);
        if (retval != 0) {
            throw runtime_error("Error attaching linear solver to CVODES: " + to_string(retval));
        }
        
        retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
        if (retval != 0) {
            throw runtime_error("Error setting Jacobian function: " + to_string(retval));
        }
        
        retval = CVodeSetMaxNumSteps(cvode_mem, MAX_CVODE_STEPS);
        if (retval != 0) {
            throw runtime_error("Error setting maximum integration steps: " + to_string(retval));
        }

        //======================================================================
        // Initialize sensitivity analysis if requested
        //======================================================================
        
        if (enable_sensitivity) {
            retval = setupSensitivityAnalysis();
            if (retval != 0) {
                throw runtime_error("Error setting up sensitivity analysis: " + to_string(retval));
            }
        }
        
    } catch (const runtime_error& e) {
        cleanup(); // Ensure proper cleanup before re-throwing
        throw;     
    }

    // Record setup timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setup_time, start, stop);
    
    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Destructor: Clean up all allocated GPU and SUNDIALS resources
 */
DynamicsIntegrator::~DynamicsIntegrator() {
    cleanup();
}

/**
 * @brief Comprehensive cleanup of all allocated resources
 * 
 * Safely releases all GPU memory, SUNDIALS contexts, and library handles.
 * Called by destructor and in error conditions.
 */
void DynamicsIntegrator::cleanup() {

    // SUNDIALS cleanup
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (Jac) SUNMatDestroy(Jac);
    if (LS) SUNLinSolFree(LS);
    if (y) N_VDestroy(y);
    if (sunctx) SUNContext_Free(&sunctx);

    // CUDA library handles
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cusolver_handle) cusolverSpDestroy(cusolver_handle);

    // Device memory
    if (d_torque_params_ptr) {
        cudaFree(d_torque_params_ptr); 
        d_torque_params_ptr = nullptr;
    }
    if (d_dt_params_ptr) {
        cudaFree(d_dt_params_ptr); 
        d_dt_params_ptr = nullptr;
    }

    // Pinned host memory
    if (h_y_pinned) {
        cudaFreeHost(h_y_pinned); 
        h_y_pinned = nullptr;
    }
    if (h_tau_pinned) {
        cudaFreeHost(h_tau_pinned); 
        h_tau_pinned = nullptr;
    }
    if (h_dt_pinned) {
        cudaFreeHost(h_dt_pinned); 
        h_dt_pinned = nullptr;
    }

    // Sensitivity analysis resources
    if (yS) N_VDestroyVectorArray(yS, Ns);
    if (d_yS_ptrs) {
        cudaFree(d_yS_ptrs); 
        d_yS_ptrs = nullptr;
    }
    if (d_ySdot_ptrs) {
        cudaFree(d_ySdot_ptrs); 
        d_ySdot_ptrs = nullptr;
    }
    if (compute_stream) {
        cudaStreamDestroy(compute_stream); 
        compute_stream = nullptr;
    }

    // Internal sparsity storage
    if (y0_row_idx) {
        free(y0_row_idx);
        y0_row_idx = nullptr;
    }
    if (u_row_idx) {
        free(u_row_idx);
        u_row_idx = nullptr;
    }
    if (dt_row_idx) {
        free(dt_row_idx);
        dt_row_idx = nullptr;
    }
}

/**
 * @brief Setup sparse Jacobian structure for SUNDIALS integration
 * 
 * Configures the Compressed Sparse Row (CSR) structure for the system Jacobian
 * matrix. The structure is block-diagonal with each block representing one
 * spacecraft system's coupling pattern.
 * 
 * Block structure (per 7×7 system):
 * - Quaternion rows (0-3): Full coupling to all 7 variables
 * - Angular velocity rows (4-6): Coupling only to angular velocities (4-6)
 * 
 * Total: 4×7 + 3×3 = 37 non-zero entries per block
 */
void DynamicsIntegrator::setupJacobianStructure() {
    sunindextype h_rowptrs[n_states + 1];
    sunindextype h_colvals[nnz];
    
    int nnz_count = 0;
    
    // Quaternion rows: each couples to all 7 state variables
    for (int row = 0; row < 4; row++) {
        h_rowptrs[row] = nnz_count;
        for (int j = 0; j < n_states; j++) {
            h_colvals[nnz_count++] = j;  // Columns 0,1,2,3,4,5,6
        }
    }
    
    // Angular velocity rows: each couples only to angular velocities
    for (int row = 4; row < n_states; row++) {
        h_rowptrs[row] = nnz_count;
        for (int j = 4; j < n_states; j++) {
            h_colvals[nnz_count++] = j;  // Columns 4,5,6 only
        }
    }
    
    h_rowptrs[n_states] = nnz_count;
        
    // Transfer sparsity pattern to device memory
    sunindextype* d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype* d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);
    
    CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs,
                         (n_states + 1) * sizeof(sunindextype), 
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_colvals, h_colvals,
                         nnz * sizeof(sunindextype),
                         cudaMemcpyHostToDevice));
    
    // Inform SUNDIALS that sparsity pattern is fixed
    SUNMatrix_cuSparse_SetFixedPattern(Jac, SUNTRUE);
}

/**
 * @brief Initialize forward sensitivity analysis system
 * 
 * Sets up sensitivity vectors and CVODES sensitivity module for computing
 * derivatives ∂y/∂p with respect to initial conditions, controls, and time steps.
 * 
 * Parameter organization:
 * - Parameters 0 to (n_stp×n_states-1): Initial condition sensitivities
 * - Parameters (n_stp×n_states) to (n_stp×(n_states+n_controls)-1): Control sensitivities
 * - Parameters (n_stp×(n_states+n_controls)) to (n_stp×(n_states+n_controls+1)-1): Time sensitivities
 * 
 * @return 0 on success, negative error code on failure
 */
int DynamicsIntegrator::setupSensitivityAnalysis() {
    // Calculate total number of sensitivity parameters
    Ns = n_stp * (n_states + n_controls + 1);
    
    // Create array of sensitivity vectors (one per parameter)
    yS = N_VCloneVectorArray(Ns, y);
    if (!yS) {
        cerr << "Error creating sensitivity vector array" << endl;
        return -1;
    }
    
    // Initialize sensitivity vectors with appropriate initial conditions
    initializeSensitivityVectors();
        
    // Initialize CVODES sensitivity module
    SUNDIALS_CHECK(
        CVodeSensInit(cvode_mem, Ns, CV_STAGGERED, sensitivityRHSFunction, yS),
        "Error initializing CVODES sensitivity analysis"
    );
    
    // Set sensitivity-specific tolerances (typically more relaxed)
    sunrealtype* abstol_S = (sunrealtype*)malloc(Ns * sizeof(sunrealtype));
    for (int i = 0; i < Ns; i++) {
        abstol_S[i] = SENSITIVITY_ATOL;
    }
    
    SUNDIALS_CHECK(
        CVodeSensSStolerances(cvode_mem, SENSITIVITY_RTOL, abstol_S),
        "Error setting sensitivity tolerances"
    );
    
    free(abstol_S);
    
    // Enable sensitivity error control for robust integration
    SUNDIALS_CHECK(
        CVodeSetSensErrCon(cvode_mem, SUNTRUE),
        "Error enabling sensitivity error control"
    );

    // Allocate device memory for sensitivity vector pointer arrays
    CUDA_CHECK(cudaMalloc(&d_yS_ptrs, Ns * sizeof(sunrealtype*)));
    CUDA_CHECK(cudaMalloc(&d_ySdot_ptrs, Ns * sizeof(sunrealtype*)));

    // Create dedicated stream for sensitivity computations
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    // Pre-compute sparsity patterns for efficient Jacobian extraction
    computeSparsities();

    sens_was_setup = true;
    sensitivity_enabled = true;
    
    return 0;
}

/**
 * @brief Initialize sensitivity vectors with identity structure at t=0
 * 
 * Sets up initial conditions for sensitivity differential equations:
 * - ∂y(t=0)/∂y₀ = I (identity matrix)
 * - ∂y(t=0)/∂u = 0 (zero matrix)
 * - ∂y(t=0)/∂dt = 0 (zero vector)
 */
void DynamicsIntegrator::initializeSensitivityVectors() {
    sunrealtype one = 1.0;
    
    // Zero all sensitivity vectors
    for (int is = 0; is < Ns; is++) {
        sunrealtype* yS_data = N_VGetDeviceArrayPointer_Cuda(yS[is]);
        CUDA_CHECK(cudaMemset(yS_data, 0, n_total * sizeof(sunrealtype)));
        
        if (is < n_stp * n_states) {
            // Initial condition sensitivity: set diagonal elements to 1
            int sys = is / n_states;
            int state_idx = is % n_states;
            int state_idx_global = sys * n_states + state_idx;
            CUDA_CHECK(cudaMemcpy(&yS_data[state_idx_global], &one, 
                                 sizeof(sunrealtype), cudaMemcpyHostToDevice));
        }
        // Control and time sensitivities start at zero (already set by memset)
    }
}

/**
 * @brief Pre-compute sparse Jacobian patterns and allocate internal storage
 * 
 * Calculates and stores the sparsity structure for ∂y/∂y₀, ∂y/∂u, and ∂y/∂dt
 * matrices. This is done once to avoid repeated computation during sensitivity extraction.
 * 
 * Sparsity patterns reflect the block-diagonal structure where each spacecraft
 * system only affects its own final state.
 */
void DynamicsIntegrator::computeSparsities() {
    if (sens_was_setup) return;  // Already computed
    
    //==========================================================================
    // ∂y/∂y₀ sparsity pattern (state-to-state sensitivities)
    //==========================================================================
    
    // Count non-zeros first
    y0_nnz = nnz * n_stp;  // Each system has 37 non-zero entries

    // Allocate internal storage
    y0_row_idx = (long long int*)malloc(y0_nnz * sizeof(long long int));
    if (!y0_row_idx) {
        cerr << "Error allocating memory for Y0 sparsity" << endl;
        return;
    }
    
    // Fill sparsity pattern
    int idx = 0;
    for (int sys = 0; sys < n_stp; sys++) {
        for (int state = 0; state < n_states; state++) {
            if (state < 4) {
                // Quaternion column: affects only quaternions of same system
                for (int row = 0; row < 4; row++) {
                    y0_row_idx[idx++] = sys * n_states + row;
                }
            } else {
                // Angular velocity column: affects all states of same system
                for (int row = 0; row < n_states; row++) {
                    y0_row_idx[idx++] = sys * n_states + row;
                }
            }
        }
    }
    
    //==========================================================================
    // ∂y/∂u sparsity pattern (control-to-state sensitivities)
    //==========================================================================
    
    u_nnz = n_stp * n_controls * n_states;  // Dense within each system
    u_row_idx = (long long int*)malloc(u_nnz * sizeof(long long int));
    if (!u_row_idx) {
        cerr << "Error allocating memory for U sparsity" << endl;
        return;
    }
    
    idx = 0;
    for (int sys = 0; sys < n_stp; sys++) {
        for (int ctrl = 0; ctrl < n_controls; ctrl++) {
            // Each control affects all states of the same system
            for (int row = 0; row < n_states; row++) {
                u_row_idx[idx++] = sys * n_states + row;
            }
        }
    }
    
    //==========================================================================
    // ∂y/∂dt sparsity pattern (time-to-state sensitivities)
    //==========================================================================
    
    dt_nnz = n_stp * n_states;  // Dense within each system
    dt_row_idx = (long long int*)malloc(dt_nnz * sizeof(long long int));
    if (!dt_row_idx) {
        cerr << "Error allocating memory for dt sparsity" << endl;
        return;
    }
    
    idx = 0;
    for (int sys = 0; sys < n_stp; sys++) {
        // Each time step affects all states of the same system
        for (int row = 0; row < n_states; row++) {
            dt_row_idx[idx++] = sys * n_states + row;
        }
    }
}

//==============================================================================
// STATIC SUNDIALS CALLBACK FUNCTIONS
//==============================================================================

/**
 * @brief SUNDIALS RHS function callback - computes time derivatives
 * 
 * Called by CVODES during integration to evaluate f(t,y). Launches GPU kernel
 * for batch computation of spacecraft dynamics.
 */
int DynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    DynamicsIntegrator* integrator = static_cast<DynamicsIntegrator*>(user_data);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype* ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);
        
    // Launch RHS kernel with optimal thread configuration
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dynamicsRHS, 0, 0);
    int gridSize = (n_stp + blockSize - 1) / blockSize;
    
    dynamicsRHS<<<gridSize, blockSize>>>(N_VGetLength(y), y_data, ydot_data, 
                                        integrator->d_torque_params_ptr, 
                                        integrator->d_dt_params_ptr);
    
    CUDA_CHECK_KERNEL();
    
    return 0;
}

/**
 * @brief SUNDIALS Jacobian function callback - computes system Jacobian
 * 
 * Called by CVODES for Newton iteration setup. Launches GPU kernel for
 * sparse Jacobian matrix computation.
 */
int DynamicsIntegrator::jacobianFunction(sunrealtype t, N_Vector y, N_Vector fy, 
                           SUNMatrix Jac, void* user_data, 
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {

    DynamicsIntegrator* integrator = static_cast<DynamicsIntegrator*>(user_data);
    int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
    sunrealtype* data = SUNMatrix_cuSparse_Data(Jac);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
        
    // Launch Jacobian kernel with one thread block per spacecraft system
    dim3 blockSize(32);
    dim3 gridSize(n_blocks);
    
    sparseJacobian<<<gridSize, blockSize>>>(n_blocks, data, y_data, 
                                           integrator->d_dt_params_ptr);
        
    CUDA_CHECK_KERNEL();
        
    return 0;
}

/**
 * @brief SUNDIALS sensitivity RHS callback - computes sensitivity derivatives
 * 
 * Called by CVODES during sensitivity integration. Launches GPU kernel for
 * batch sensitivity derivative computation using Jacobian-vector products.
 */
int DynamicsIntegrator::sensitivityRHSFunction(int Ns, sunrealtype t, N_Vector y, N_Vector ydot,
                                             N_Vector* yS, N_Vector* ySdot, void* user_data,
                                             N_Vector tmp1, N_Vector tmp2) {

    DynamicsIntegrator* integrator = static_cast<DynamicsIntegrator*>(user_data);
    sunrealtype* y_data = N_VGetDeviceArrayPointer_Cuda(y);
    
    // Prepare array of device pointers for kernel access
    sunrealtype** h_yS_ptrs = (sunrealtype**)malloc(Ns * sizeof(sunrealtype*));
    sunrealtype** h_ySdot_ptrs = (sunrealtype**)malloc(Ns * sizeof(sunrealtype*));
    
    for (int i = 0; i < Ns; i++) {
        h_yS_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(yS[i]);
        h_ySdot_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(ySdot[i]);
    }
    
    // Transfer pointer arrays to device memory
    CUDA_CHECK(cudaMemcpyAsync(integrator->d_yS_ptrs, h_yS_ptrs, 
                            Ns * sizeof(sunrealtype*), cudaMemcpyHostToDevice, 
                            integrator->compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(integrator->d_ySdot_ptrs, h_ySdot_ptrs, 
                            Ns * sizeof(sunrealtype*), cudaMemcpyHostToDevice, 
                            integrator->compute_stream));
    
    // Launch sensitivity kernel with one block per system
    dim3 sensBlockSize(n_states + n_controls + 1);  // One thread per parameter type
    dim3 sensGridSize(n_stp);                       // One block per system
    sensitivityRHS<<<sensGridSize, sensBlockSize, 0, integrator->compute_stream>>>(
        Ns, y_data, integrator->d_yS_ptrs, integrator->d_ySdot_ptrs, 
        integrator->d_dt_params_ptr);

    // Ensure all operations complete before returning to CVODES
    CUDA_CHECK(cudaStreamSynchronize(integrator->compute_stream));
    
    // Cleanup temporary arrays
    free(h_yS_ptrs);
    free(h_ySdot_ptrs);
    
    return 0;
}

//==============================================================================
// MAIN INTERFACE METHODS
//==============================================================================

/**
 * @brief Main integration function - solve spacecraft dynamics problem
 * 
 * Performs batch integration of multiple spacecraft systems from t=0 to t=dt
 * with optional forward sensitivity analysis for optimization applications.
 * 
 * Integration process:
 * 1. Transfer problem data to GPU memory
 * 2. Setup/reinitialize sensitivity analysis (if requested)
 * 3. Reset CVODES integrator state
 * 4. Perform adaptive integration with Newton iterations
 * 5. Extract final states and sensitivities
 * 
 * @param initial_states Caller-allocated array of initial quaternions and angular velocities (n_stp × n_states)
 * @param torque_params Caller-allocated array of applied torques for each system (n_stp × n_controls)
 * @param delta_t Caller-allocated array of integration time steps for each system (n_stp)
 * @param enable_sensitivity Compute forward sensitivities for optimization
 * @return 0 on success, negative error code on failure
 */
int DynamicsIntegrator::solve(const sunrealtype* initial_states, 
                             const sunrealtype* torque_params,
                             const sunrealtype* delta_t, 
                             bool enable_sensitivity) {
    
    // Create CUDA events for precise timing of solve operations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
    // Transfer initial conditions and parameters to GPU memory
    setInitialConditions(initial_states, torque_params, delta_t);

    //==========================================================================
    // Setup sensitivity analysis if requested
    //==========================================================================
    
    if (enable_sensitivity) {
        if (!sens_was_setup) {
            // First time setup - expensive operation
            SUNDIALS_CHECK(setupSensitivityAnalysis(), 
                          "Error setting up sensitivity analysis");
        } else {
            // Subsequent times: just reinitialize vectors
            initializeSensitivityVectors();
            SUNDIALS_CHECK(CVodeSensReInit(cvode_mem, CV_STAGGERED, yS), 
                          "Error reinitializing sensitivity analysis");
        }
        sensitivity_enabled = true;
    } else if (sensitivity_enabled) {
        // Disable sensitivity analysis for this solve
        SUNDIALS_CHECK(CVodeSensToggleOff(cvode_mem), 
                      "Error disabling sensitivity analysis");
        sensitivity_enabled = false;
    }

    //==========================================================================
    // Perform integration from t=0 to t=1 (unit time with dt scaling)
    //==========================================================================
    
    SUNDIALS_CHECK(CVodeReInit(cvode_mem, 0.0, y), "Error reinitializing CVODE");
    
    sunrealtype t = 0.0;
    SUNDIALS_CHECK(CVode(cvode_mem, 1.0, y, &t, CV_NORMAL), "CVode integration error");

    // Ensure all GPU operations complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract sensitivity results if computed
    if (sensitivity_enabled) {
        SUNDIALS_CHECK(CVodeGetSens(cvode_mem, &t, yS), 
                      "Error getting sensitivity derivatives");
    }

    // Record solve timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&solve_time, start, stop);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/**
 * @brief Transfer initial conditions and parameters to GPU memory
 * 
 * Copies problem data from host to device using pinned memory for optimal
 * transfer performance. This is called for every solve() operation.
 */
void DynamicsIntegrator::setInitialConditions(const sunrealtype* initial_states, 
                                              const sunrealtype* torque_params,
                                              const sunrealtype* delta_t) {
    
    // Copy to pinned memory for fast GPU transfer
    memcpy(h_y_pinned, initial_states, n_total * sizeof(sunrealtype));
    memcpy(h_tau_pinned, torque_params, n_stp * n_controls * sizeof(sunrealtype));
    memcpy(h_dt_pinned, delta_t, n_stp * sizeof(sunrealtype));

    // Transfer to GPU memory
    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_total * sizeof(sunrealtype), 
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, h_tau_pinned, 
                             n_stp * n_controls * sizeof(sunrealtype), 
                             cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_dt_params_ptr, h_dt_pinned, 
                             n_stp * sizeof(sunrealtype), 
                             cudaMemcpyHostToDevice));
}

//==============================================================================
// RESULT ACCESS METHODS
//==============================================================================

/**
 * @brief Copy final integrated states from GPU to caller-allocated buffer
 * 
 * Copies the final quaternions and angular velocities from GPU to host memory
 * in the same layout as the input.
 */
int DynamicsIntegrator::getSolution(sunrealtype* next_state) const {

    if (!next_state) {
        return -1;
    }

    sunrealtype* d_y = N_VGetDeviceArrayPointer_Cuda(y);

    CUDA_CHECK(cudaMemcpy(next_state, d_y, n_total * sizeof(sunrealtype),
                         cudaMemcpyDeviceToHost));
    return 0;
}

//==============================================================================
// SPARSE SENSITIVITY ACCESS METHODS
//==============================================================================

/**
 * @brief Extract ∂y_final/∂y_initial sensitivities to caller-allocated array
 * 
 * Extracts sensitivity values corresponding to the pre-computed sparsity pattern
 * for efficient use with sparse matrix libraries.
 */
int DynamicsIntegrator::getSensitivitiesY0(sunrealtype* values) const {
    if (!sensitivity_enabled) {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!sens_was_setup || !values) {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }
    
    sunrealtype* sens_data = (sunrealtype*)malloc(n_total * sizeof(sunrealtype));
    if (!sens_data) {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }
    
    int values_idx = 0;
    int sparsity_idx = 0;
    
    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_stp * n_states; col++) {
        int param_idx = col;  // Initial condition parameters come first
        
        // Copy sensitivity data from GPU for this column
        sunrealtype* d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_total * sizeof(sunrealtype), 
                             cudaMemcpyDeviceToHost));
        
        // Determine number of entries for this column based on state type
        int entries_this_col;
        int state_type = col % n_states;
        if (state_type < 4) {
            entries_this_col = 4;  // Quaternion affects 4 quaternions
        } else {
            entries_this_col = n_states;  // Angular velocity affects all states
        }
        
        // Extract values for non-zero entries in this column
        for (int entry = 0; entry < entries_this_col; entry++) {
            int row = y0_row_idx[sparsity_idx++];
            values[values_idx++] = sens_data[row];
        }
    }
    
    free(sens_data);
    return 0;
}

/**
 * @brief Extract ∂y_final/∂u sensitivities to caller-allocated array
 */
int DynamicsIntegrator::getSensitivitiesU(sunrealtype* values) const {
    if (!sensitivity_enabled) {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!sens_was_setup || !values) {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }
    
    sunrealtype* sens_data = (sunrealtype*)malloc(n_total * sizeof(sunrealtype));
    if (!sens_data) {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }
    
    int values_idx = 0;
    int sparsity_idx = 0;
    
    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_stp * n_controls; col++) {
        int param_idx = n_stp * n_states + col;  // Control parameters start after initial conditions
        
        sunrealtype* d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_total * sizeof(sunrealtype), 
                             cudaMemcpyDeviceToHost));
        
        // Each control affects all states of its system
        for (int entry = 0; entry < n_states; entry++) {
            int row = u_row_idx[sparsity_idx++];
            values[values_idx++] = sens_data[row];
        }
    }
    
    free(sens_data);
    return 0;
}

/**
 * @brief Extract ∂y_final/∂dt sensitivities to caller-allocated array
 */
int DynamicsIntegrator::getSensitivitiesDt(sunrealtype* values) const {
    if (!sensitivity_enabled) {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!sens_was_setup || !values) {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }
    
    sunrealtype* sens_data = (sunrealtype*)malloc(n_total * sizeof(sunrealtype));
    if (!sens_data) {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }
    
    int values_idx = 0;
    int sparsity_idx = 0;
    
    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_stp; col++) {
        int param_idx = n_stp * (n_states + n_controls) + col;  // dt parameters come last
        
        sunrealtype* d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_total * sizeof(sunrealtype), 
                             cudaMemcpyDeviceToHost));
        
        // Each dt affects all states of its system
        for (int entry = 0; entry < n_states; entry++) {
            int row = dt_row_idx[sparsity_idx++];
            values[values_idx++] = sens_data[row];
        }
    }
    
    free(sens_data);
    return 0;
}

//==============================================================================
// SPARSITY PATTERN ACCESS METHODS
//==============================================================================

/**
 * @brief Get sparsity pattern for ∂y_final/∂y_initial Jacobian
 */
int DynamicsIntegrator::getSparsityY0(long long int* row_indices, long long int* col_pointers, int* nnz_out) const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    if (!row_indices || !col_pointers || !nnz_out) {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }
    
    // Copy row indices
    memcpy(row_indices, y0_row_idx, y0_nnz * sizeof(long long int));
    
    // Build column pointers
    int col_ptr_idx = 0;
    int sparsity_idx = 0;
    col_pointers[col_ptr_idx++] = 0;
    
    for (int col = 0; col < n_stp * n_states; col++) {
        int state_type = col % n_states;
        int entries_this_col = (state_type < 4) ? 4 : n_states;
        sparsity_idx += entries_this_col;
        col_pointers[col_ptr_idx++] = sparsity_idx;
    }
    
    *nnz_out = y0_nnz;
    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂u Jacobian
 */
int DynamicsIntegrator::getSparsityU(long long int* row_indices, long long int* col_pointers, int* nnz_out) const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    if (!row_indices || !col_pointers || !nnz_out) {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }
    
    // Copy row indices
    memcpy(row_indices, u_row_idx, u_nnz * sizeof(long long int));
    
    // Build column pointers
    col_pointers[0] = 0;
    for (int col = 0; col < n_stp * n_controls; col++) {
        col_pointers[col + 1] = col_pointers[col] + n_states;
    }
    
    *nnz_out = u_nnz;
    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂dt Jacobian
 */
int DynamicsIntegrator::getSparsityDt(long long int* row_indices, long long int* col_pointers, int* nnz_out) const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    if (!row_indices || !col_pointers || !nnz_out) {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }
    
    // Copy row indices
    memcpy(row_indices, dt_row_idx, dt_nnz * sizeof(long long int));
    
    // Build column pointers
    col_pointers[0] = 0;
    for (int col = 0; col < n_stp; col++) {
        col_pointers[col + 1] = col_pointers[col] + n_states;
    }
    
    *nnz_out = dt_nnz;
    return 0;
}

//==============================================================================
// SPARSITY SIZE QUERY METHODS
//==============================================================================

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂y_initial Jacobian
 */
int DynamicsIntegrator::getSparsitySizeY0() const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    return y0_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂u Jacobian
 */
int DynamicsIntegrator::getSparsitySizeU() const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    return u_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂dt Jacobian
 */
int DynamicsIntegrator::getSparsitySizeDt() const {
    if (!sens_was_setup) {
        cerr << "Sparsity not computed. Enable sensitivity analysis first." << endl;
        return -1;
    }
    return dt_nnz;
}

//==============================================================================
// UPDATED TEST AND VALIDATION METHODS
//==============================================================================

/**
 * @brief Generate a random unit quaternion using rejection sampling
 */
void DynamicsIntegrator::generateRandomQuaternion(sunrealtype* quaternion) {
    sunrealtype norm_sq;
    
    do {
        quaternion[0] = 2.0 * ((sunrealtype)rand() / RAND_MAX) - 1.0;
        quaternion[1] = 2.0 * ((sunrealtype)rand() / RAND_MAX) - 1.0;
        quaternion[2] = 2.0 * ((sunrealtype)rand() / RAND_MAX) - 1.0;
        quaternion[3] = 2.0 * ((sunrealtype)rand() / RAND_MAX) - 1.0;
        
        norm_sq = quaternion[0]*quaternion[0] + quaternion[1]*quaternion[1] + 
                  quaternion[2]*quaternion[2] + quaternion[3]*quaternion[3];
    } while (norm_sq > 1.0 || norm_sq < 1e-6);
    
    // Normalize to unit quaternion
    sunrealtype norm = sqrt(norm_sq);
    quaternion[0] /= norm;
    quaternion[1] /= norm;
    quaternion[2] /= norm;
    quaternion[3] /= norm;
}

/**
 * @brief Generate random angular velocity in range [-π/2, π/2]
 */
void DynamicsIntegrator::generateRandomAngularVelocity(sunrealtype* omega) {
    const sunrealtype pi_half = PI / 2.0;
    
    omega[0] = 2.0 * pi_half * ((sunrealtype)rand() / RAND_MAX) - pi_half; // wx
    omega[1] = 2.0 * pi_half * ((sunrealtype)rand() / RAND_MAX) - pi_half; // wy
    omega[2] = 2.0 * pi_half * ((sunrealtype)rand() / RAND_MAX) - pi_half; // wz
}

/**
 * @brief Generate random torque within actuator limits
 */
void DynamicsIntegrator::generateRandomTorque(sunrealtype* torque) {
    torque[0] = 2.0 * tau_max * ((sunrealtype)rand() / RAND_MAX) - tau_max; // tau_x
    torque[1] = 2.0 * tau_max * ((sunrealtype)rand() / RAND_MAX) - tau_max; // tau_y
    torque[2] = 2.0 * tau_max * ((sunrealtype)rand() / RAND_MAX) - tau_max; // tau_z
}

/**
 * @brief Generate batch of random initial conditions and torques
 */
void DynamicsIntegrator::generateBatchInputs(int batch_size, 
                        sunrealtype* initial_states,
                        sunrealtype* torque_params,
                        sunrealtype* dt_params) {
    
    sunrealtype quat[4], omega[3], torque[3];
    
    for (int i = 0; i < batch_size; i++) {
        // Generate random quaternion and angular velocity
        generateRandomQuaternion(quat);
        generateRandomAngularVelocity(omega);
        generateRandomTorque(torque);

        // Fill initial states array
        int state_base = i * n_states;
        for (int j = 0; j < 4; j++) {
            initial_states[state_base + j] = quat[j];
        }
        for (int j = 0; j < 3; j++) {
            initial_states[state_base + 4 + j] = omega[j];
        }

        // Fill torque params array
        int torque_base = i * n_controls;
        for (int j = 0; j < n_controls; j++) {
            torque_params[torque_base + j] = torque[j];
        }

        // Random time step between 0.01 and 1.0 seconds
        dt_params[i] = 0.01 + 0.99 * ((sunrealtype)rand() / RAND_MAX);
    }

}

/**
 * @brief Compute quaternion norms to check integration accuracy
 */
void DynamicsIntegrator::getQuaternionNorms(const sunrealtype* states, sunrealtype* norms) const {
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        norms[i] = sqrt(states[base_idx+0]*states[base_idx+0] + 
                       states[base_idx+1]*states[base_idx+1] + 
                       states[base_idx+2]*states[base_idx+2] + 
                       states[base_idx+3]*states[base_idx+3]);
    }
}

/**
 * @brief Compute angular momentum vector for each system: L = I * omega
 */
void DynamicsIntegrator::computeAngularMomentum(const sunrealtype* states, sunrealtype* angular_momentum) const {
    for (int i = 0; i < n_stp; i++) {
        int state_base = i * n_states;
        int momentum_base = i * n_controls;
        
        // L = I * omega for diagonal inertia matrix
        angular_momentum[momentum_base + 0] = i_x * states[state_base + 4];  // Lx
        angular_momentum[momentum_base + 1] = i_y * states[state_base + 5];  // Ly
        angular_momentum[momentum_base + 2] = i_z * states[state_base + 6];  // Lz 
    }
}

/**
 * @brief Compute rotational energy for each system: E = 0.5 * omega^T * I * omega
 */
void DynamicsIntegrator::computeRotationalEnergy(const sunrealtype* states, sunrealtype* energy) const{
    for (int i = 0; i < n_stp; i++) {
        int base_idx = i * n_states;
        
        // E = 0.5 * omega^T * I * omega for diagonal inertia matrix
        energy[i] = 0.5 * (i_x * states[base_idx + 4] * states[base_idx + 4] + 
                          i_y * states[base_idx + 5] * states[base_idx + 5] + 
                          i_z * states[base_idx + 6] * states[base_idx + 6]);
    }
}

/**
 * @brief Validate sensitivity analysis using finite difference approximation
 */
void DynamicsIntegrator::validateSensitivityAnalysis(const sunrealtype* initial_states,
                                                   const sunrealtype* torque_params,
                                                   const sunrealtype* integration_time,
                                                   int num_systems_to_test) {
    
    cout << "\n=== Sensitivity Analysis Validation ===" << endl;
    
    const sunrealtype epsilon = 1e-3;
    const sunrealtype tolerance = 1e-5;  // Relaxed for numerical differences
    
    int systems_to_test = min(num_systems_to_test, n_stp);
    
    cout << "Testing " << systems_to_test << " systems with ε = " 
              << scientific << epsilon << endl;
    
    // Allocate arrays for baseline solution
    sunrealtype baseline_solution[n_total];
    sunrealtype perturbed_solution[n_total];
    sunrealtype perturbed_initial[n_total];
    sunrealtype perturbed_torques[n_stp * n_controls];

    try {
        // Get baseline solution with sensitivity
        int result = solve(initial_states, torque_params, integration_time, true);
        if (result != 0) {
            cout << "ERROR: Baseline integration failed" << endl;
            return;
        }
        getSolution(baseline_solution);
        
        // Get sensitivity sizes and allocate arrays
        int y0_nnz = getSparsitySizeY0();
        int u_nnz = getSparsitySizeU();
        int dt_nnz = getSparsitySizeDt();
        
        if (y0_nnz < 0 || u_nnz < 0 || dt_nnz < 0) {
            cout << "ERROR: Failed to get sensitivity sizes" << endl;
            return;
        }

        sunrealtype y0_values[y0_nnz];
        sunrealtype u_values[u_nnz];
        sunrealtype dt_values[dt_nnz];

        try {
            // Get sensitivity values
            getSensitivitiesY0(y0_values);
            getSensitivitiesU(u_values);
            getSensitivitiesDt(dt_values);
            
            vector<sunrealtype> gradient_errors;
            int total_tests = 0;
            int passed_tests = 0;
            
            vector<int> ic_params_to_test = {1, 5, 6}; // q1, wy, wz indices
            
            for (int sys = 0; sys < systems_to_test; sys++) {
                cout << "\nTesting System " << sys << ":" << endl;
                
                for (int state_idx : ic_params_to_test) {
                    int param_idx = sys * n_states + state_idx;
                    
                    if (param_idx >= n_total) continue;
                    
                    // Copy initial states and perturb parameter
                    memcpy(perturbed_initial, initial_states, n_total * sizeof(sunrealtype));
                    perturbed_initial[param_idx] += epsilon;
                    
                    cout << "  Perturbing parameter " << param_idx 
                              << " (" << (state_idx == 1 ? "q1" : state_idx == 5 ? "wy" : "wz") << ")" << endl;
                    
                    result = solve(perturbed_initial, torque_params, integration_time, false);
                    if (result != 0) {
                        cout << "  WARNING: Perturbed integration failed" << endl;
                        continue;
                    }
                    getSolution(perturbed_solution);
                    
                    // Compare only the same system response
                    sunrealtype fd_gradient[n_states];
                    for (int i = 0; i < n_states; i++) {
                        fd_gradient[i] = (perturbed_solution[sys * n_states + i] - baseline_solution[sys * n_states + i]) / epsilon;
                    }
                    
                    // Extract analytical gradients for the same system
                    sunrealtype analytical_gradient[n_states];
                    if (state_idx < 4) {
                        // Quaternion components
                        for (int i = 0; i < 4; i++) {
                            analytical_gradient[i] = y0_values[sys * nnz + state_idx * 4 + i];
                        }
                        for (int i = 0; i < 3; i++) {
                            analytical_gradient[i + 4] = 0.0;
                        }
                    } else {
                        // Angular velocity components
                        for (int i = 0; i < n_states; i++) {
                            analytical_gradient[i] = y0_values[sys * nnz + 16 + (state_idx-4) * n_states + i];
                        }
                    }


                    cout << "  Analytical: [" << scientific << setprecision(3);
                    for (int i = 0; i < n_states; i++) cout << analytical_gradient[i] << " ";
                    cout << "]" << endl;
                    
                    cout << "  Finite Diff: [";
                    for (int i = 0; i < n_states; i++) cout << fd_gradient[i] << " ";
                    cout << "]" << endl;
                    
                    // Compute errors and statistics
                    sunrealtype max_error = 0.0;
                    for (int state_comp = 0; state_comp < n_states; state_comp++) {
                        sunrealtype error = abs(analytical_gradient[state_comp] - fd_gradient[state_comp]);
                        max_error = max(max_error, error);
                        gradient_errors.push_back(error);
                        total_tests++;
                        if (error < tolerance) passed_tests++;
                    }
                    
                    cout << "  Max error: " << scientific << max_error 
                            << (max_error < tolerance ? " PASS" : " FAIL") << endl;
                }
                
                // Test one torque parameter
                int torque_param_idx = n_stp * n_states + sys * n_controls; // tau_x
                
                if (torque_param_idx < n_stp * (n_states + n_controls)) {
                    // Copy initial states and perturb parameter
                    memcpy(perturbed_torques, torque_params, n_stp*n_controls * sizeof(sunrealtype));
                    perturbed_torques[sys*n_controls] += epsilon;
                    
                    int result = solve(initial_states, perturbed_torques, integration_time, false);
                    if (result == 0) {
                        getSolution(perturbed_solution);
                        
                        sunrealtype fd_gradient[n_states];
                        for (int i = 0; i < n_states; i++) {
                            fd_gradient[i] = (perturbed_solution[sys*n_states + i] - baseline_solution[sys*n_states + i]) / epsilon;
                        }
                        
                        sunrealtype analytical_gradient[n_states];
                        for (int state_comp = 0; state_comp < n_states; state_comp++) {
                            analytical_gradient[state_comp] = u_values[sys * n_controls * n_states + state_comp];
                        }
                        
                        sunrealtype max_error = 0.0;
                        for (int state_comp = 0; state_comp < n_states; state_comp++) {
                            sunrealtype error = abs(analytical_gradient[state_comp] - fd_gradient[state_comp]);
                            max_error = max(max_error, error);
                            gradient_errors.push_back(error);
                            total_tests++;
                            if (error < tolerance) passed_tests++;
                        }

                        cout << "  Perturbing parameter " << n_total + sys * n_controls << " (tau_x)" << endl;

                        cout << "  Analytical: [" << scientific << setprecision(3);
                        for (int i = 0; i < n_states; i++) cout << analytical_gradient[i] << " ";
                        cout << "]" << endl;
                        
                        cout << "  Finite Diff: [";
                        for (int i = 0; i < n_states; i++) cout << fd_gradient[i] << " ";
                        cout << "]" << endl;
                        cout << "  ∂y/∂tau_x max error: " << scientific << max_error 
                                << (max_error < tolerance ? " PASS" : " FAIL") << endl;
                    }
                }
            }
            
            // Overall statistics
            if (!gradient_errors.empty()) {
                auto max_overall_error = *max_element(gradient_errors.begin(), gradient_errors.end());
                auto avg_error = accumulate(gradient_errors.begin(), gradient_errors.end(), 0.0) / gradient_errors.size();
                
                cout << "\n=== Sensitivity Validation Summary ===" << endl;
                cout << "Total tests: " << total_tests << endl;
                cout << "Passed tests: " << passed_tests << " (" << fixed << setprecision(1) 
                        << (100.0 * passed_tests / total_tests) << "%)" << endl;
                cout << "Max error: " << scientific << max_overall_error << endl;
                cout << "Avg error: " << scientific << avg_error << endl;
                cout << "Overall result: " << (max_overall_error < tolerance ? "PASS" : "FAIL") << endl;
            }

        }
        catch (const std::exception& e) {
            cerr << "Error during sensitivity validation: " << e.what() << endl;
            return;
        }
    }
    catch (...) {
        cerr << "Unexpected error during sensitivity validation" << endl;
        return;
    }
}

void DynamicsIntegrator::measureSensitivityCost(int n_iterations) {
    if (!sensitivity_enabled) {
        cout << "Sensitivity analysis not enabled, skipping cost measurement." << endl;
        return;
    }
    
    cout << "\n=== Sensitivity Analysis Cost Measurement ===" << endl;
    
    
    // Run a single sensitivity analysis to measure cost
    sunrealtype initial_states[n_stp * n_states];
    sunrealtype torque_params[n_stp * n_controls];
    sunrealtype delta_t[n_stp];
    generateBatchInputs(n_stp, initial_states, torque_params, delta_t);
    float solve_time = 0.0;
    float sumtime = 0.0;
    int result = 1;
    cout << "=========================" << endl;
    cout << "With sensitivity analysis" << endl;
    cout << "=========================" << endl;
    for (int i = 0; i < n_iterations; i++) {
        result = solve(initial_states, torque_params, delta_t, true);
        if (result != 0) {
            cerr << "Integration failed during sensitivity cost measurement" << endl;
            return;
        }
        solve_time = getSolveTime();
        sumtime += solve_time;

        cout << "Iteration " << i + 1 << ": Solve time = " << scientific 
                << setprecision(3) << solve_time << " ms" << endl;

    }

    float sens_time = sumtime / n_iterations;
    result = solve(initial_states, torque_params, delta_t, false);
    if (result != 0) {
        cerr << "Integration failed during sensitivity cost measurement" << endl;
        return;
    }
    sumtime = 0.0;
    cout << "============================" << endl;
    cout << "Without sensitivity analysis" << endl;
    cout << "============================" << endl;
    for (int i = 0; i < n_iterations; i++) {
        result = solve(initial_states, torque_params, delta_t, false);
        if (result != 0) {
            cerr << "Integration failed during sensitivity cost measurement" << endl;
            return;
        }
        solve_time = getSolveTime();
        sumtime += solve_time;

        cout << "Iteration " << i + 1 << ": Solve time = " << scientific 
                << setprecision(3) << solve_time << " ms" << endl;
    }
    float no_sens_time = sumtime / n_iterations;
    float extra_cost = sens_time - no_sens_time;
    cout << "Sensitivity analysis cost per iteration: " << scientific 
            << setprecision(3) << sens_time << " ms" << endl;
    cout << "Without sensitivity analysis cost per iteration: " << scientific 
            << setprecision(3) << no_sens_time << " ms" << endl;
    cout << "Sensitivity analysis extra cost per iteration: " << scientific 
            << setprecision(3) << extra_cost << " ms" << endl;
}

/**
 * @brief Profile integration performance with detailed timing
 */
void DynamicsIntegrator::profileIntegration(int batch_size, int num_iterations) {
    cout << "\n=== Performance Profiling ===" << endl;
    cout << "Batch size: " << batch_size << ", Iterations: " << num_iterations << endl;
    
    vector<float> solve_times;
    solve_times.reserve(num_iterations);
    float t_setup = getSetupTime();

    // Allocate arrays for test data
    sunrealtype initial_states[batch_size * n_states];
    sunrealtype torque_params[batch_size * n_controls];
    sunrealtype delta_t[batch_size];
    
    try {
        for (int iter = 0; iter < num_iterations; iter++) {
            cout << "\n--- Iteration " << iter + 1 << " ---" << endl;
            
            // Generate random inputs
            generateBatchInputs(batch_size, initial_states, torque_params, delta_t);

            int result = solve(initial_states, torque_params, delta_t, false);
            
            if (result != 0) {
                cerr << "Integration failed on iteration " << iter << endl;
                continue;
            }
            solve_times.push_back(getSolveTime());
        }

        cout << "\n-----------------------";
        cout << "\n--- Average Results ---";
        cout << "\n-----------------------" << endl;

        // Statistical analysis
        float solve_min = *min_element(solve_times.begin(), solve_times.end());
        float solve_max = *max_element(solve_times.begin(), solve_times.end());
        float solve_avg = accumulate(solve_times.begin(), solve_times.end(), 0.0f) / solve_times.size();
        
        // Report timing results
        cout << fixed << setprecision(3);
        cout << "Setup Time (ms):  " << t_setup << endl;
        cout << "Solve Time (ms):  Min=" << solve_min << ", Max=" << solve_max << ", Avg=" << solve_avg << endl;
        
        // Performance metrics
        float avg_time_per_system = solve_avg / batch_size;
        float systems_per_second = (batch_size * 1000.0f) / solve_avg;
        
        cout << "Performance Metrics:" << endl;
        cout << "  Time per system: " << avg_time_per_system << " ms" << endl;
        cout << "  Systems per second: " << (int)systems_per_second << endl;
        cout << endl;
        
    } catch (...) {
        throw;
    }
}

/**
 * @brief Main test driver - runs comprehensive validation suite
 */
void DynamicsIntegrator::runComprehensiveTest(const vector<int>& batch_sizes, int iterations) {
    cout << "Starting Comprehensive Dynamics Integrator Test Suite..." << endl;
    cout << "Random seed: " << time(nullptr) << endl;
    
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Generate performance report
    profileIntegration(n_stp, iterations);
    
    // Allocate arrays for physics tests
    sunrealtype initial_states[n_total];
    sunrealtype torque_params[n_stp * n_controls];
    sunrealtype delta_t[n_stp];
    sunrealtype final_states[n_total];
    
    try {
        // Run physics tests
        generateBatchInputs(n_stp, initial_states, torque_params, delta_t);
        cout << "Running physics verification for initial random states..." << endl;
        
        int result = solve(initial_states, torque_params, delta_t, false);
        if (result != 0) {
            cerr << "Integration failed" << endl;
            return;
        }
        
        getSolution(final_states);

        validatePhysics(initial_states, final_states, torque_params, delta_t);

        cout << "Physics verification for initial random states completed." << endl;
        
        // Validate sensitivity analysis
        validateSensitivityAnalysis(initial_states, torque_params, delta_t, 1);
        
    } catch (...) {

        throw;
    }
}

/**
 * @brief Compute power input from torques: P = tau · omega
 */
void DynamicsIntegrator::computePowerInput(const sunrealtype* states,
                                           const sunrealtype* torques, 
                                           sunrealtype* power) const {
    for (int i = 0; i < n_stp; i++) {
        int state_base = i * n_states;
        int torque_base = i * n_controls;
        
        // P = tau · omega (dot product of torque and angular velocity)
        power[i] = torques[torque_base + 0] * states[state_base + 4] + 
                  torques[torque_base + 1] * states[state_base + 5] + 
                  torques[torque_base + 2] * states[state_base + 6];
    }
}

/**
 * @brief Verify physics laws with external torques applied
 */
void DynamicsIntegrator::verifyPhysics(const sunrealtype* initial_states,
                                       const sunrealtype* final_states,
                                       const sunrealtype* torque_params,
                                       const sunrealtype* integration_time) const {

    cout << "\n=== Physics Verification ===" << endl;
    
    // Allocate arrays for physics quantities
    sunrealtype L_initial[n_stp * n_controls];
    sunrealtype L_final[n_stp * n_controls];
    sunrealtype E_initial[n_stp];
    sunrealtype E_final[n_stp];
    sunrealtype P_initial[n_stp];
    sunrealtype P_final[n_stp];
    sunrealtype quat_norms[n_stp];
    
    try {
        // Compute initial and final quantities
        computeAngularMomentum(initial_states, L_initial);
        computeAngularMomentum(final_states, L_final);
        computeRotationalEnergy(initial_states, E_initial);
        computeRotationalEnergy(final_states, E_final);
        computePowerInput(initial_states, torque_params, P_initial);
        computePowerInput(final_states, torque_params, P_final);
        getQuaternionNorms(final_states, quat_norms);
        
        vector<sunrealtype> L_errors, E_errors, norm_errors;
        L_errors.reserve(n_stp);
        E_errors.reserve(n_stp);
        norm_errors.reserve(n_stp);
        
        for (int sys = 0; sys < n_stp; sys++) {
            // 1. Angular Momentum Rate Check: dL/dt = tau_applied
            // Expected change: Delta_L = tau * Delta_t
            int momentum_base = sys * n_controls;
            int torque_base = sys * n_controls;
            
            sunrealtype expected_dL[3];
            expected_dL[0] = torque_params[torque_base + 0] * integration_time[sys];
            expected_dL[1] = torque_params[torque_base + 1] * integration_time[sys];
            expected_dL[2] = torque_params[torque_base + 2] * integration_time[sys];
            
            sunrealtype actual_dL[3];
            actual_dL[0] = L_final[momentum_base + 0] - L_initial[momentum_base + 0];
            actual_dL[1] = L_final[momentum_base + 1] - L_initial[momentum_base + 1];
            actual_dL[2] = L_final[momentum_base + 2] - L_initial[momentum_base + 2];
            
            // Error in angular momentum change (vector magnitude)
            sunrealtype dL_error_sq = 0.0;
            for (int j = 0; j < n_controls; j++) {
                sunrealtype diff = actual_dL[j] - expected_dL[j];
                dL_error_sq += diff * diff;
            }
            L_errors.push_back(sqrt(dL_error_sq));
            
            // 2. Energy Rate Check: dE/dt = tau · omega
            // Use average power for better approximation: Delta_E ≈ P_avg * Delta_t
            sunrealtype P_avg = 0.5 * (P_initial[sys] + P_final[sys]);
            sunrealtype expected_dE = P_avg * integration_time[sys];
            sunrealtype actual_dE = E_final[sys] - E_initial[sys];
            E_errors.push_back(abs(actual_dE - expected_dE));
            
            // 3. Quaternion Norm Preservation (should always be conserved)
            norm_errors.push_back(abs(quat_norms[sys] - 1.0));
        }
        
        // Statistical analysis
        auto L_max = *max_element(L_errors.begin(), L_errors.end());
        auto L_avg = accumulate(L_errors.begin(), L_errors.end(), 0.0) / L_errors.size();
        
        auto E_max = *max_element(E_errors.begin(), E_errors.end());
        auto E_avg = accumulate(E_errors.begin(), E_errors.end(), 0.0) / E_errors.size();
        
        auto norm_max = *max_element(norm_errors.begin(), norm_errors.end());
        auto norm_avg = accumulate(norm_errors.begin(), norm_errors.end(), 0.0) / norm_errors.size();
        
        // Report results
        cout << "Angular Momentum Rate Verification (dL/dt = τ):" << endl;
        cout << "  Max error: " << scientific << L_max << " kg⋅m²/s" << endl;
        cout << "  Avg error: " << scientific << L_avg << " kg⋅m²/s" << endl;
        
        cout << "Energy Rate Verification (dE/dt = τ⋅ω):" << endl;
        cout << "  Max error: " << scientific << E_max << " J" << endl;
        cout << "  Avg error: " << scientific << E_avg << " J" << endl;
        
        cout << "Quaternion Norm Preservation:" << endl;
        cout << "  Max error: " << scientific << norm_max << endl;
        cout << "  Avg error: " << scientific << norm_avg << endl;
        
        // Pass/fail assessment (adjusted tolerances for physics verification)
        bool L_pass = L_max < 1e-8;  // Relaxed for torque integration
        bool E_pass = E_max < 1e-8;  // Relaxed for power integration
        bool norm_pass = norm_max < QUAT_NORM_TOLERANCE;  // Strict for quaternion norm
        
        cout << "Physics Verification: " 
                  << (L_pass ? "τ-PASS " : "τ-FAIL ")
                  << (E_pass ? "P-PASS " : "P-FAIL ")
                  << (norm_pass ? "N-PASS" : "N-FAIL") << endl;
        
    } catch (...) {
        throw;
    }
}

/**
 * @brief Validate physics with torque considerations
 */
void DynamicsIntegrator::validatePhysics(const sunrealtype* initial_states,
                                         const sunrealtype* final_states,
                                         const sunrealtype* torque_params,
                                         const sunrealtype* integration_time) const{
    // Check if torques are negligible (all components < 1e-8)
    bool torques_negligible = true;
    for (int i = 0; i < n_stp * n_controls; i++) {
        if (abs(torque_params[i]) > 1e-8) {
            torques_negligible = false;
            break;
        }
    }
    
    if (torques_negligible) {
        // Use original conservation law tests for torque-free cases
        cout << "\n=== Torque-Free Conservation Analysis ===" << endl;

        sunrealtype L_initial[n_stp * n_controls];
        sunrealtype L_final[n_stp * n_controls];
        sunrealtype E_initial[n_stp];
        sunrealtype E_final[n_stp];

        try {
            computeAngularMomentum(initial_states, L_initial);
            computeAngularMomentum(final_states, L_final);
            computeRotationalEnergy(initial_states, E_initial);
            computeRotationalEnergy(final_states, E_final);
            
            // Check conservation with strict tolerances
            vector<sunrealtype> L_errors, E_errors;
            for (int sys = 0; sys < n_stp; sys++) {
                int momentum_base = sys * n_controls;
                
                sunrealtype L_total_final = sqrt(L_final[momentum_base + 0] * L_final[momentum_base + 0] +
                                                L_final[momentum_base + 1] * L_final[momentum_base + 1] +
                                                L_final[momentum_base + 2] * L_final[momentum_base + 2]);
                sunrealtype L_total_initial = sqrt(L_initial[momentum_base + 0] * L_initial[momentum_base + 0] +
                                                  L_initial[momentum_base + 1] * L_initial[momentum_base + 1] +
                                                  L_initial[momentum_base + 2] * L_initial[momentum_base + 2]);
                
                L_errors.push_back(abs(L_total_final - L_total_initial));
                E_errors.push_back(abs(E_final[sys] - E_initial[sys]));
            }
            
            auto L_max = *max_element(L_errors.begin(), L_errors.end());
            auto E_max = *max_element(E_errors.begin(), E_errors.end());
            
            cout << "Conservation errors - L: " << scientific << L_max 
                      << ", E: " << E_max << endl;
            
            bool conserved = (L_max < 1e-8 && E_max < 1e-8);
            cout << "Conservation test: " << (conserved ? "PASS" : "FAIL") << endl;
            
        } catch (...) {
            throw;
        }
        
    } else {
        // Use physics verification for cases with external torques
        verifyPhysics(initial_states, final_states, torque_params, integration_time);
    }
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

int main() {
    try {
        // Initialize integrator with sensitivity analysis enabled
        DynamicsIntegrator integrator(true);
        
        // // Define test parameters
        // vector<int> batch_sizes = {n_stp};
        // int iterations_per_test = 1;
        
        // cout << "CUDA Dynamics Integrator Test Suite" << endl;
        // cout << "===================================" << endl;
        
        // // Run comprehensive test suite
        // integrator.runComprehensiveTest(batch_sizes, iterations_per_test);
        
        // cout << "\nTest suite completed successfully!" << endl;

        integrator.measureSensitivityCost(10);
        
    } catch (const exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Test failed with unknown exception" << endl;
        return 1;
    }
    
    return 0;
}