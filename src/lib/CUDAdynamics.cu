#include <toac/cuda_dynamics.h>

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
__global__ void dynamicsRHS(int n_total, sunrealtype *y, sunrealtype *ydot,
                            sunrealtype *torque_params, sunrealtype *dt_params)
{
    // Calculate system index from thread and block indices
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys < 0 || sys >= n_stp)
        return;

    // Load precomputed inertia constants from cached constant memory
    const sunrealtype Ix = d_inertia_constants[0];     // Moment of inertia about x-axis
    const sunrealtype Iy = d_inertia_constants[1];     // Moment of inertia about y-axis
    const sunrealtype Iz = d_inertia_constants[2];     // Moment of inertia about z-axis
    const sunrealtype half = d_inertia_constants[3];   // Constant 0.5 for quaternion kinematics
    const sunrealtype Ix_inv = d_inertia_constants[4]; // 1/Ix for efficient division
    const sunrealtype Iy_inv = d_inertia_constants[5]; // 1/Iy for efficient division
    const sunrealtype Iz_inv = d_inertia_constants[6]; // 1/Iz for efficient division

    // Calculate base index for this system's state variables
    int base_idx = sys * n_states;
    if (base_idx + n_states > n_total)
        return;

    // Extract individual state components for readability
    sunrealtype q0 = y[base_idx + 0], q1 = y[base_idx + 1]; // Quaternion components
    sunrealtype q2 = y[base_idx + 2], q3 = y[base_idx + 3];
    sunrealtype wx = y[base_idx + 4], wy = y[base_idx + 5], wz = y[base_idx + 6]; // Angular velocities

    // Load control parameters for this system
    sunrealtype tau_x = torque_params[sys * n_controls + 0]; // Torque about x-axis
    sunrealtype tau_y = torque_params[sys * n_controls + 1]; // Torque about y-axis
    sunrealtype tau_z = torque_params[sys * n_controls + 2]; // Torque about z-axis
    sunrealtype dt = dt_params[sys];                         // Time step for this system

    // Quaternion kinematic equations: q̇ = 0.5 * Ω(ω) * q
    // where Ω(ω) is the skew-symmetric matrix of angular velocity
    ydot[base_idx + 0] = half * (-q1 * wx - q2 * wy - q3 * wz) * dt; // q0_dot
    ydot[base_idx + 1] = half * (q0 * wx - q3 * wy + q2 * wz) * dt;  // q1_dot
    ydot[base_idx + 2] = half * (q3 * wx + q0 * wy - q1 * wz) * dt;  // q2_dot
    ydot[base_idx + 3] = half * (-q2 * wx + q1 * wy + q0 * wz) * dt; // q3_dot

    // Euler's equations: İω̇ = τ - ω × (İω)
    // Pre-compute angular momentum components for efficiency
    sunrealtype Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;

    // Compute angular acceleration including gyroscopic terms
    ydot[base_idx + 4] = Ix_inv * (tau_x - (wy * Iw_z - wz * Iw_y)) * dt; // wx_dot
    ydot[base_idx + 5] = Iy_inv * (tau_y - (wz * Iw_x - wx * Iw_z)) * dt; // wy_dot
    ydot[base_idx + 6] = Iz_inv * (tau_z - (wx * Iw_y - wy * Iw_x)) * dt; // wz_dot
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
__global__ void sparseJacobian(int n_blocks, sunrealtype *block_data,
                               sunrealtype *y, sunrealtype *dt_params)
{
    // One thread block per spacecraft system
    int block_id = blockIdx.x;
    if (block_id >= n_blocks)
        return;

    // Calculate pointer to this block's Jacobian data
    sunrealtype *block_jac = block_data + block_id * nnz;
    int base_state_idx = block_id * n_states;

    // Load current state variables for Jacobian evaluation
    sunrealtype q0 = y[base_state_idx + 0], q1 = y[base_state_idx + 1];
    sunrealtype q2 = y[base_state_idx + 2], q3 = y[base_state_idx + 3];
    sunrealtype wx = y[base_state_idx + 4], wy = y[base_state_idx + 5], wz = y[base_state_idx + 6];

    // Load time step for this system
    sunrealtype dt = dt_params[block_id];

    // Load constants from constant memory
    const sunrealtype half = d_inertia_constants[3];        // 0.5
    const sunrealtype Ix_inv = d_inertia_constants[4];      // 1/Ix
    const sunrealtype Iy_inv = d_inertia_constants[5];      // 1/Iy
    const sunrealtype Iz_inv = d_inertia_constants[6];      // 1/Iz
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7]; // Iz - Iy
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8]; // Ix - Iz
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9]; // Iy - Ix
    const sunrealtype minus_half = d_inertia_constants[10]; // -0.5
    const sunrealtype zero = d_inertia_constants[11];       // 0.0

    // Fill Jacobian entries in CSR order (row-by-row, within each row by column)
    int idx = 0;

    // Quaternion kinematic Jacobian: ∂q̇/∂[q,ω]
    // Each quaternion derivative couples to all state variables

    // Row 0: ∂q0_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = zero;                 // ∂q0_dot/∂q0 = 0
    block_jac[idx++] = minus_half * wx * dt; // ∂q0_dot/∂q1 = -0.5*wx*dt
    block_jac[idx++] = minus_half * wy * dt; // ∂q0_dot/∂q2 = -0.5*wy*dt
    block_jac[idx++] = minus_half * wz * dt; // ∂q0_dot/∂q3 = -0.5*wz*dt
    block_jac[idx++] = minus_half * q1 * dt; // ∂q0_dot/∂wx = -0.5*q1*dt
    block_jac[idx++] = minus_half * q2 * dt; // ∂q0_dot/∂wy = -0.5*q2*dt
    block_jac[idx++] = minus_half * q3 * dt; // ∂q0_dot/∂wz = -0.5*q3*dt

    // Row 1: ∂q1_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wx * dt;       // ∂q1_dot/∂q0 = 0.5*wx*dt
    block_jac[idx++] = zero;                 // ∂q1_dot/∂q1 = 0
    block_jac[idx++] = half * wz * dt;       // ∂q1_dot/∂q2 = 0.5*wz*dt
    block_jac[idx++] = minus_half * wy * dt; // ∂q1_dot/∂q3 = -0.5*wy*dt
    block_jac[idx++] = half * q0 * dt;       // ∂q1_dot/∂wx = 0.5*q0*dt
    block_jac[idx++] = minus_half * q3 * dt; // ∂q1_dot/∂wy = -0.5*q3*dt
    block_jac[idx++] = half * q2 * dt;       // ∂q1_dot/∂wz = 0.5*q2*dt

    // Row 2: ∂q2_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wy * dt;       // ∂q2_dot/∂q0 = 0.5*wy*dt
    block_jac[idx++] = minus_half * wz * dt; // ∂q2_dot/∂q1 = -0.5*wz*dt
    block_jac[idx++] = zero;                 // ∂q2_dot/∂q2 = 0
    block_jac[idx++] = half * wx * dt;       // ∂q2_dot/∂q3 = 0.5*wx*dt
    block_jac[idx++] = half * q3 * dt;       // ∂q2_dot/∂wx = 0.5*q3*dt
    block_jac[idx++] = half * q0 * dt;       // ∂q2_dot/∂wy = 0.5*q0*dt
    block_jac[idx++] = minus_half * q1 * dt; // ∂q2_dot/∂wz = -0.5*q1*dt

    // Row 3: ∂q3_dot/∂[q0,q1,q2,q3,wx,wy,wz]
    block_jac[idx++] = half * wz * dt;       // ∂q3_dot/∂q0 = 0.5*wz*dt
    block_jac[idx++] = half * wy * dt;       // ∂q3_dot/∂q1 = 0.5*wy*dt
    block_jac[idx++] = minus_half * wx * dt; // ∂q3_dot/∂q2 = -0.5*wx*dt
    block_jac[idx++] = zero;                 // ∂q3_dot/∂q3 = 0
    block_jac[idx++] = minus_half * q2 * dt; // ∂q3_dot/∂wx = -0.5*q2*dt
    block_jac[idx++] = half * q1 * dt;       // ∂q3_dot/∂wy = 0.5*q1*dt
    block_jac[idx++] = half * q0 * dt;       // ∂q3_dot/∂wz = 0.5*q0*dt

    // Angular velocity dynamics Jacobian: ∂ω̇/∂ω (gyroscopic coupling only)
    // Quaternions don't directly affect angular acceleration, so ∂ω̇/∂q = 0

    // Row 4: ∂wx_dot/∂[wx,wy,wz] (only non-zero angular velocity derivatives)
    block_jac[idx++] = zero;                            // ∂wx_dot/∂wx = 0
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy)*wz * dt; // ∂wx_dot/∂wy (gyroscopic)
    block_jac[idx++] = -Ix_inv * (Iz_minus_Iy)*wy * dt; // ∂wx_dot/∂wz (gyroscopic)

    // Row 5: ∂wy_dot/∂[wx,wy,wz]
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz)*wz * dt; // ∂wy_dot/∂wx (gyroscopic)
    block_jac[idx++] = zero;                            // ∂wy_dot/∂wy = 0
    block_jac[idx++] = -Iy_inv * (Ix_minus_Iz)*wx * dt; // ∂wy_dot/∂wz (gyroscopic)

    // Row 6: ∂wz_dot/∂[wx,wy,wz]
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix)*wy * dt; // ∂wz_dot/∂wx (gyroscopic)
    block_jac[idx++] = -Iz_inv * (Iy_minus_Ix)*wx * dt; // ∂wz_dot/∂wy (gyroscopic)
    block_jac[idx++] = zero;                            // ∂wz_dot/∂wz = 0
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
 * @param torque_params Torque parameters for each system (device memory)
 * @param dt_params Time step parameters for each system (device memory)
 */
__global__ void sensitivityRHS(int Ns, sunrealtype *y, sunrealtype **yS_data_array,
                               sunrealtype **ySdot_data_array, sunrealtype *torque_params,
                               sunrealtype *dt_params)
{
    int sys = blockIdx.x;         // One thread block per spacecraft system
    int param_type = threadIdx.x; // Parameter type within this system

    if (sys >= n_stp)
        return;

    // Shared memory for efficient broadcasting of system state to all threads
    __shared__ sunrealtype shared_state[n_states];
    __shared__ sunrealtype shared_dt;
    __shared__ sunrealtype shared_torques[n_controls];

    // Load system state into shared memory using first n_states threads
    if (param_type < n_states)
    {
        shared_state[param_type] = y[sys * n_states + param_type];
    }
    else if (param_type < n_states + n_controls)
    {
        int torque_idx = param_type - n_states;
        shared_torques[torque_idx] = torque_params[sys * n_controls + torque_idx];
    }
    else if (param_type == n_states + n_controls)
    {
        shared_dt = dt_params[sys];
    }
    else
        return; // Invalid parameter type

    __syncthreads(); // Ensure all threads see the loaded states

    // Decode global parameter index based on parameter type and system
    int global_param_idx;
    if (param_type < n_states)
    {
        // Initial condition parameter: ∂y/∂y₀
        global_param_idx = sys * n_states + param_type;
    }
    else if (param_type < n_states + n_controls)
    {
        // Control parameter: ∂y/∂u
        global_param_idx = n_stp * n_states + sys * n_controls + (param_type - n_states);
    }
    else if (param_type == n_states + n_controls)
    {
        // Time step parameter: ∂y/∂dt
        global_param_idx = n_stp * (n_states + n_controls) + sys;
    }
    else
    {
        return; // Invalid parameter index
    }

    if (global_param_idx >= Ns)
        return;

    // Access sensitivity vectors for this parameter
    sunrealtype *yS_data = yS_data_array[global_param_idx];
    sunrealtype *ySdot_data = ySdot_data_array[global_param_idx];

    // Load current sensitivity state for this system
    sunrealtype s_state[n_states];
#pragma unroll
    for (int i = 0; i < n_states; i++)
    {
        s_state[i] = yS_data[sys * n_states + i];
    }

    // Extract system state components from shared memory
    sunrealtype q0 = shared_state[0], q1 = shared_state[1];
    sunrealtype q2 = shared_state[2], q3 = shared_state[3];
    sunrealtype wx = shared_state[4], wy = shared_state[5], wz = shared_state[6];

    // Load constants for Jacobian computation
    const sunrealtype half = d_inertia_constants[3];        // 0.5
    const sunrealtype Ix_inv = d_inertia_constants[4];      // 1/Ix
    const sunrealtype Iy_inv = d_inertia_constants[5];      // 1/Iy
    const sunrealtype Iz_inv = d_inertia_constants[6];      // 1/Iz
    const sunrealtype Iz_minus_Iy = d_inertia_constants[7]; // Iz - Iy
    const sunrealtype Ix_minus_Iz = d_inertia_constants[8]; // Ix - Iz
    const sunrealtype Iy_minus_Ix = d_inertia_constants[9]; // Iy - Ix

    // For sensitivity analysis, we need: ds/dp = ∂f/∂y * s + ∂f/∂p, where f(y,u,dt) = dt * g(y,u)
    // For initial conditions: ∂f/∂y₀ = 0, so ds/dp = dt * ∂g/∂y * s
    // For controls: ∂f/∂u = dt * ∂g/∂u, so ds/dp = dt * (∂g/∂y * s + ∂g/∂u)
    // For dt parameter: ∂f/∂dt = g(y,u), so ds/dp = dt * ∂g/∂y * s + g(y,u)

    // Compute Jacobian-vector product: ∂g/∂y * s
    sunrealtype Js[n_states];

    // Quaternion sensitivity derivatives (couple to all variables)
    Js[0] = half * (-wx * s_state[1] - wy * s_state[2] - wz * s_state[3] - q1 * s_state[4] - q2 * s_state[5] - q3 * s_state[6]);
    Js[1] = half * (wx * s_state[0] + wz * s_state[2] - wy * s_state[3] + q0 * s_state[4] - q3 * s_state[5] + q2 * s_state[6]);
    Js[2] = half * (wy * s_state[0] - wz * s_state[1] + wx * s_state[3] + q3 * s_state[4] + q0 * s_state[5] - q1 * s_state[6]);
    Js[3] = half * (wz * s_state[0] + wy * s_state[1] - wx * s_state[2] - q2 * s_state[4] + q1 * s_state[5] + q0 * s_state[6]);

    // Angular velocity sensitivity derivatives (gyroscopic coupling only)
    Js[4] = -Ix_inv * Iz_minus_Iy * wz * s_state[5] - Ix_inv * Iz_minus_Iy * wy * s_state[6];
    Js[5] = -Iy_inv * Ix_minus_Iz * wz * s_state[4] - Iy_inv * Ix_minus_Iz * wx * s_state[6];
    Js[6] = -Iz_inv * Iy_minus_Ix * wy * s_state[4] - Iz_inv * Iy_minus_Ix * wx * s_state[5];

    // Add direct parameter dependencies: ∂g/∂u terms
    if (param_type >= n_states && param_type < n_states + n_controls)
    {
        // Control parameter affects corresponding angular acceleration directly
        int torque_idx = param_type - n_states;
        if (torque_idx == 0)
            Js[4] += Ix_inv; // ∂wx_dot/∂tau_x = 1/Ix
        else if (torque_idx == 1)
            Js[5] += Iy_inv; // ∂wy_dot/∂tau_y = 1/Iy
        else if (torque_idx == 2)
            Js[6] += Iz_inv; // ∂wz_dot/∂tau_z = 1/Iz
    }
    else if (param_type == n_states + n_controls)
    {
        // Time step parameter: ∂f/∂dt = g(y,u) (the original dynamics)
        // Get torques from shared memory
        sunrealtype tau_x = shared_torques[0];
        sunrealtype tau_y = shared_torques[1];
        sunrealtype tau_z = shared_torques[2];

        // Quaternion dt sensitivities (divided by dt to cancel out the multiplication at the end)
        Js[0] += half * (-wx * q1 - wy * q2 - wz * q3) / shared_dt;
        Js[1] += half * (wx * q0 + wz * q2 - wy * q3) / shared_dt;
        Js[2] += half * (wy * q0 - wz * q1 + wx * q3) / shared_dt;
        Js[3] += half * (wz * q0 + wy * q1 - wx * q2) / shared_dt;

        // Angular velocity dt sensitivities (divided by dt to cancel out the multiplication at the end)
        const sunrealtype Ix = d_inertia_constants[0];
        const sunrealtype Iy = d_inertia_constants[1];
        const sunrealtype Iz = d_inertia_constants[2];

        sunrealtype Iw_x = Ix * wx;
        sunrealtype Iw_y = Iy * wy;
        sunrealtype Iw_z = Iz * wz;

        Js[4] += Ix_inv * (tau_x - (wy * Iw_z - wz * Iw_y)) / shared_dt;
        Js[5] += Iy_inv * (tau_y - (wz * Iw_x - wx * Iw_z)) / shared_dt;
        Js[6] += Iz_inv * (tau_z - (wx * Iw_y - wy * Iw_x)) / shared_dt;
    }

// Store sensitivity derivatives for this system (scaled by time step)
#pragma unroll
    for (int i = 0; i < n_states; i++)
    {
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
DynamicsIntegrator::DynamicsIntegrator(bool enable_sensitivity) : setup_time(0), solve_time(0),
                                                                  d_yS_ptrs(nullptr), d_ySdot_ptrs(nullptr), yS(nullptr), Ns(0),
                                                                  sensitivity_enabled(enable_sensitivity), sens_was_setup(false),
                                                                  y0_row_idx(nullptr), y0_nnz(0), u_row_idx(nullptr), u_nnz(0),
                                                                  dt_row_idx(nullptr), dt_nnz(0), sparsity_computed(false)
{

    // Create CUDA events for precise timing of setup operations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    try
    {
        //======================================================================
        // Initialize CUDA constant memory with spacecraft parameters
        //======================================================================

        sunrealtype h_constants[12] = {
            i_x, i_y, i_z, 0.5,                    // [0-3]: Principal moments and 0.5
            1.0 / i_x, 1.0 / i_y, 1.0 / i_z,       // [4-6]: Inverse inertias for efficiency
            (i_z - i_y), (i_x - i_z), (i_y - i_x), // [7-9]: Inertia differences (Euler terms)
            -0.5, 0.0                              // [10-11]: Additional constants
        };

        cudaError_t cuda_err = cudaMemcpyToSymbol(d_inertia_constants, h_constants,
                                                  12 * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error copying spacecraft parameters to constant memory: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Allocate device memory for control parameters and time steps
        //======================================================================

        cuda_err = cudaMalloc(&d_torque_params_ptr,
                              n_controls_total * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error allocating device memory for torque parameters: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMalloc(&d_dt_params_ptr, n_stp * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error allocating device memory for time step parameters: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Initialize SUNDIALS context and library handles
        //======================================================================

        if (SUNContext_Create(NULL, &sunctx) != 0)
        {
            throw runtime_error("Error creating SUNDIALS execution context");
        }

        // Create cuSPARSE handle for sparse matrix operations
        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS)
        {
            throw runtime_error("Error creating cuSPARSE handle: " +
                                to_string(cusparse_status));
        }

        // Create cuSOLVER handle for linear system solving
        cusolverStatus_t cusolver_status = cusolverSpCreate(&cusolver_handle);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
        {
            throw runtime_error("Error creating cuSOLVER handle: " +
                                to_string(cusolver_status));
        }

        //======================================================================
        // Create SUNDIALS vectors and allocate pinned host memory
        //======================================================================

        y = N_VNew_Cuda(n_states_total, sunctx);
        if (!y)
        {
            throw runtime_error("Error creating CUDA state vector");
        }

        // Allocate pinned host memory for fast GPU transfers
        cuda_err = cudaMallocHost(&h_y_pinned, n_states_total * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error allocating pinned memory for states: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMallocHost(&h_tau_pinned, n_controls_total * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error allocating pinned memory for controls: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        cuda_err = cudaMallocHost(&h_dt_pinned, n_stp * sizeof(sunrealtype));
        if (cuda_err != cudaSuccess)
        {
            throw runtime_error("Error allocating pinned memory for time steps: " +
                                string(cudaGetErrorString(cuda_err)));
        }

        //======================================================================
        // Initialize CVODES integrator
        //======================================================================

        cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
        if (!cvode_mem)
        {
            throw runtime_error("Error creating CVODES integrator instance");
        }

        int retval = CVodeInit(cvode_mem, rhsFunction, 0.0, y);
        if (retval != 0)
        {
            throw runtime_error("Error initializing CVODES: " + to_string(retval));
        }

        //======================================================================
        // Setup sparse Jacobian matrix and linear solver
        //======================================================================

        Jac = SUNMatrix_cuSparse_NewBlockCSR(n_stp, n_states, n_states, nnz,
                                             cusparse_handle, sunctx);
        if (!Jac)
        {
            throw runtime_error("Error creating cuSPARSE block-diagonal matrix");
        }

        setupJacobianStructure();

        // Create batch QR linear solver optimized for block-diagonal systems
        LS = SUNLinSol_cuSolverSp_batchQR(y, Jac, cusolver_handle, sunctx);
        if (!LS)
        {
            throw runtime_error("Error creating cuSOLVER batch QR linear solver");
        }

        //======================================================================
        // Configure CVODES with tolerances and solver
        //======================================================================

        retval = CVodeSetUserData(cvode_mem, this);
        if (retval != 0)
        {
            throw runtime_error("Error setting CVODES user data: " + to_string(retval));
        }

        retval = CVodeSStolerances(cvode_mem, DEFAULT_RTOL, DEFAULT_ATOL);
        if (retval != 0)
        {
            throw runtime_error("Error setting CVODES tolerances: " + to_string(retval));
        }

        retval = CVodeSetLinearSolver(cvode_mem, LS, Jac);
        if (retval != 0)
        {
            throw runtime_error("Error attaching linear solver to CVODES: " + to_string(retval));
        }

        retval = CVodeSetJacFn(cvode_mem, jacobianFunction);
        if (retval != 0)
        {
            throw runtime_error("Error setting Jacobian function: " + to_string(retval));
        }

        retval = CVodeSetMaxNumSteps(cvode_mem, MAX_CVODE_STEPS);
        if (retval != 0)
        {
            throw runtime_error("Error setting maximum integration steps: " + to_string(retval));
        }

        //======================================================================
        // Initialize sensitivity analysis if requested
        //======================================================================

        if (enable_sensitivity)
        {
            retval = setupSensitivityAnalysis();
            if (retval != 0)
            {
                throw runtime_error("Error setting up sensitivity analysis: " + to_string(retval));
            }
        }
    }
    catch (const runtime_error &e)
    {
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
 *
 * Ensures proper cleanup of all memory allocations and library handles
 * to prevent resource leaks. Does not free caller-allocated arrays.
 */
DynamicsIntegrator::~DynamicsIntegrator()
{
    cleanup();
}

/**
 * @brief Comprehensive cleanup of all allocated resources
 *
 * Safely releases all GPU memory, SUNDIALS contexts, and library handles.
 * Called by destructor and in error conditions.
 */
void DynamicsIntegrator::cleanup()
{

    // SUNDIALS cleanup
    if (cvode_mem)
        CVodeFree(&cvode_mem);
    if (Jac)
        SUNMatDestroy(Jac);
    if (LS)
        SUNLinSolFree(LS);
    if (y)
        N_VDestroy(y);
    if (sunctx)
        SUNContext_Free(&sunctx);

    // CUDA library handles
    if (cusparse_handle)
        cusparseDestroy(cusparse_handle);
    if (cusolver_handle)
        cusolverSpDestroy(cusolver_handle);

    // Device memory
    if (d_torque_params_ptr)
    {
        cudaFree(d_torque_params_ptr);
        d_torque_params_ptr = nullptr;
    }
    if (d_dt_params_ptr)
    {
        cudaFree(d_dt_params_ptr);
        d_dt_params_ptr = nullptr;
    }

    // Pinned host memory
    if (h_y_pinned)
    {
        cudaFreeHost(h_y_pinned);
        h_y_pinned = nullptr;
    }
    if (h_tau_pinned)
    {
        cudaFreeHost(h_tau_pinned);
        h_tau_pinned = nullptr;
    }
    if (h_dt_pinned)
    {
        cudaFreeHost(h_dt_pinned);
        h_dt_pinned = nullptr;
    }

    // Sensitivity analysis resources
    if (yS)
        N_VDestroyVectorArray(yS, Ns);
    if (d_yS_ptrs)
    {
        cudaFree(d_yS_ptrs);
        d_yS_ptrs = nullptr;
    }
    if (d_ySdot_ptrs)
    {
        cudaFree(d_ySdot_ptrs);
        d_ySdot_ptrs = nullptr;
    }
    if (compute_stream)
    {
        cudaStreamDestroy(compute_stream);
        compute_stream = nullptr;
    }

    // Internal sparsity storage
    if (y0_row_idx)
    {
        free(y0_row_idx);
        y0_row_idx = nullptr;
    }
    if (u_row_idx)
    {
        free(u_row_idx);
        u_row_idx = nullptr;
    }
    if (dt_row_idx)
    {
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
void DynamicsIntegrator::setupJacobianStructure()
{
    sunindextype h_rowptrs[n_states + 1];
    sunindextype h_colvals[nnz];

    int nnz_count = 0;

    // Quaternion rows: each couples to all 7 state variables
    for (int row = 0; row < n_quat; row++)
    {
        h_rowptrs[row] = nnz_count;
        for (int j = 0; j < n_states; j++)
        {
            h_colvals[nnz_count++] = j; // Columns 0,1,2,3,4,5,6
        }
    }

    // Angular velocity rows: each couples only to angular velocities
    for (int row = n_quat; row < n_states; row++)
    {
        h_rowptrs[row] = nnz_count;
        for (int j = n_quat; j < n_states; j++)
        {
            h_colvals[nnz_count++] = j; // Columns 4,5,6 only
        }
    }

    h_rowptrs[n_states] = nnz_count;

    // Transfer sparsity pattern to device memory
    sunindextype *d_rowptrs = SUNMatrix_cuSparse_IndexPointers(Jac);
    sunindextype *d_colvals = SUNMatrix_cuSparse_IndexValues(Jac);

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
int DynamicsIntegrator::setupSensitivityAnalysis()
{
    // Calculate total number of sensitivity parameters
    Ns = n_stp * (n_states + n_controls + 1);

    // Create array of sensitivity vectors (one per parameter)
    yS = N_VCloneVectorArray(Ns, y);
    if (!yS)
    {
        cerr << "Error creating sensitivity vector array" << endl;
        return -1;
    }

    // Initialize sensitivity vectors with appropriate initial conditions
    initializeSensitivityVectors();

    // Initialize CVODES sensitivity module
    SUNDIALS_CHECK(
        CVodeSensInit(cvode_mem, Ns, CV_STAGGERED, sensitivityRHSFunction, yS),
        "Error initializing CVODES sensitivity analysis");

    // Set sensitivity-specific tolerances (typically more relaxed)
    sunrealtype *abstol_S = (sunrealtype *)malloc(Ns * sizeof(sunrealtype));
    for (int i = 0; i < Ns; i++)
    {
        abstol_S[i] = SENSITIVITY_ATOL;
    }

    SUNDIALS_CHECK(
        CVodeSensSStolerances(cvode_mem, SENSITIVITY_RTOL, abstol_S),
        "Error setting sensitivity tolerances");

    free(abstol_S);

    // Enable sensitivity error control for robust integration
    SUNDIALS_CHECK(
        CVodeSetSensErrCon(cvode_mem, SUNTRUE),
        "Error enabling sensitivity error control");

    // Allocate device memory for sensitivity vector pointer arrays
    CUDA_CHECK(cudaMalloc(&d_yS_ptrs, Ns * sizeof(sunrealtype *)));
    CUDA_CHECK(cudaMalloc(&d_ySdot_ptrs, Ns * sizeof(sunrealtype *)));

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
void DynamicsIntegrator::initializeSensitivityVectors()
{
    sunrealtype one = 1.0;

    // Zero all sensitivity vectors
    for (int is = 0; is < Ns; is++)
    {
        sunrealtype *yS_data = N_VGetDeviceArrayPointer_Cuda(yS[is]);
        CUDA_CHECK(cudaMemset(yS_data, 0, n_states_total * sizeof(sunrealtype)));

        if (is < n_stp * n_states)
        {
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
void DynamicsIntegrator::computeSparsities()
{
    if (sparsity_computed)
        return; // Already computed

    //==========================================================================
    // ∂y/∂y₀ sparsity pattern (state-to-state sensitivities)
    //==========================================================================

    // Count non-zeros first
    y0_nnz = 0;
    for (int sys = 0; sys < n_stp; sys++)
    {
        for (int state = 0; state < n_states; state++)
        {
            if (state < n_quat)
            {
                // Quaternion column: affects only quaternions of same system
                y0_nnz += n_quat;
            }
            else
            {
                // Angular velocity column: affects all states of same system
                y0_nnz += n_states;
            }
        }
    }

    // Allocate internal storage
    y0_row_idx = (long long int *)malloc(y0_nnz * sizeof(long long int));
    if (!y0_row_idx)
    {
        cerr << "Error allocating memory for Y0 sparsity" << endl;
        return;
    }

    // Fill sparsity pattern
    int idx = 0;
    for (int sys = 0; sys < n_stp; sys++)
    {
        for (int state = 0; state < n_states; state++)
        {
            if (state < n_quat)
            {
                // Quaternion column: affects only quaternions of same system
                for (int row = 0; row < n_quat; row++)
                {
                    y0_row_idx[idx++] = sys * n_states + row;
                }
            }
            else
            {
                // Angular velocity column: affects all states of same system
                for (int row = 0; row < n_states; row++)
                {
                    y0_row_idx[idx++] = sys * n_states + row;
                }
            }
        }
    }

    //==========================================================================
    // ∂y/∂u sparsity pattern (control-to-state sensitivities)
    //==========================================================================

    u_nnz = n_controls_total * n_states; // Dense within each system
    u_row_idx = (long long int *)malloc(u_nnz * sizeof(long long int));
    if (!u_row_idx)
    {
        cerr << "Error allocating memory for U sparsity" << endl;
        return;
    }

    idx = 0;
    for (int sys = 0; sys < n_stp; sys++)
    {
        for (int ctrl = 0; ctrl < n_controls; ctrl++)
        {
            // Each control affects all states of the same system
            for (int row = 0; row < n_states; row++)
            {
                u_row_idx[idx++] = sys * n_states + row;
            }
        }
    }

    //==========================================================================
    // ∂y/∂dt sparsity pattern (time-to-state sensitivities)
    //==========================================================================

    dt_nnz = n_stp * n_states; // Dense within each system
    dt_row_idx = (long long int *)malloc(dt_nnz * sizeof(long long int));
    if (!dt_row_idx)
    {
        cerr << "Error allocating memory for dt sparsity" << endl;
        return;
    }

    idx = 0;
    for (int sys = 0; sys < n_stp; sys++)
    {
        // Each time step affects all states of the same system
        for (int row = 0; row < n_states; row++)
        {
            dt_row_idx[idx++] = sys * n_states + row;
        }
    }
    sparsity_computed = true; // Mark sparsity as computed
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
int DynamicsIntegrator::rhsFunction(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    DynamicsIntegrator *integrator = static_cast<DynamicsIntegrator *>(user_data);
    sunrealtype *y_data = N_VGetDeviceArrayPointer_Cuda(y);
    sunrealtype *ydot_data = N_VGetDeviceArrayPointer_Cuda(ydot);

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
                                         SUNMatrix Jac, void *user_data,
                                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    DynamicsIntegrator *integrator = static_cast<DynamicsIntegrator *>(user_data);
    int n_blocks = SUNMatrix_cuSparse_NumBlocks(Jac);
    sunrealtype *data = SUNMatrix_cuSparse_Data(Jac);
    sunrealtype *y_data = N_VGetDeviceArrayPointer_Cuda(y);

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
                                               N_Vector *yS, N_Vector *ySdot, void *user_data,
                                               N_Vector tmp1, N_Vector tmp2)
{

    DynamicsIntegrator *integrator = static_cast<DynamicsIntegrator *>(user_data);
    sunrealtype *y_data = N_VGetDeviceArrayPointer_Cuda(y);

    // Prepare array of device pointers for kernel access
    sunrealtype **h_yS_ptrs = (sunrealtype **)malloc(Ns * sizeof(sunrealtype *));
    sunrealtype **h_ySdot_ptrs = (sunrealtype **)malloc(Ns * sizeof(sunrealtype *));

    for (int i = 0; i < Ns; i++)
    {
        h_yS_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(yS[i]);
        h_ySdot_ptrs[i] = N_VGetDeviceArrayPointer_Cuda(ySdot[i]);
    }

    // Transfer pointer arrays to device memory
    CUDA_CHECK(cudaMemcpyAsync(integrator->d_yS_ptrs, h_yS_ptrs,
                               Ns * sizeof(sunrealtype *), cudaMemcpyHostToDevice,
                               integrator->compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(integrator->d_ySdot_ptrs, h_ySdot_ptrs,
                               Ns * sizeof(sunrealtype *), cudaMemcpyHostToDevice,
                               integrator->compute_stream));

    // Launch sensitivity kernel with one block per system
    dim3 sensBlockSize(n_states + n_controls + 1); // One thread per parameter type
    dim3 sensGridSize(n_stp);                      // One block per system
    sensitivityRHS<<<sensGridSize, sensBlockSize, 0, integrator->compute_stream>>>(
        Ns, y_data, integrator->d_yS_ptrs, integrator->d_ySdot_ptrs,
        integrator->d_torque_params_ptr, integrator->d_dt_params_ptr);

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
int DynamicsIntegrator::solve(const sunrealtype *initial_states,
                              const sunrealtype *torque_params,
                              const sunrealtype *delta_t,
                              bool enable_sensitivity)
{

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

    if (enable_sensitivity)
    {
        if (!sens_was_setup)
        {
            // First time setup - expensive operation
            SUNDIALS_CHECK(setupSensitivityAnalysis(),
                           "Error setting up sensitivity analysis");
        }
        else
        {
            // Subsequent times: just reinitialize vectors
            initializeSensitivityVectors();
            SUNDIALS_CHECK(CVodeSensReInit(cvode_mem, CV_STAGGERED, yS),
                           "Error reinitializing sensitivity analysis");
        }
        sensitivity_enabled = true;
    }
    else if (sensitivity_enabled)
    {
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
    if (sensitivity_enabled)
    {
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
void DynamicsIntegrator::setInitialConditions(const sunrealtype *initial_states,
                                              const sunrealtype *torque_params,
                                              const sunrealtype *delta_t)
{

    // Copy to pinned memory for fast GPU transfer
    memcpy(h_y_pinned, initial_states, n_states_total * sizeof(sunrealtype));
    memcpy(h_tau_pinned, torque_params, n_controls_total * sizeof(sunrealtype));
    memcpy(h_dt_pinned, delta_t, n_stp * sizeof(sunrealtype));

    // Transfer to GPU memory
    sunrealtype *d_y = N_VGetDeviceArrayPointer_Cuda(y);
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_states_total * sizeof(sunrealtype),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, h_tau_pinned,
                          n_controls_total * sizeof(sunrealtype),
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
int DynamicsIntegrator::getSolution(sunrealtype *next_state) const
{

    if (!next_state)
    {
        return -1;
    }

    sunrealtype *d_y = N_VGetDeviceArrayPointer_Cuda(y);

    CUDA_CHECK(cudaMemcpy(next_state, d_y, n_states_total * sizeof(sunrealtype),
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
int DynamicsIntegrator::getSensitivitiesY0(sunrealtype *values) const
{
    if (!sensitivity_enabled)
    {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!values)
    {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }

    sunrealtype *sens_data = (sunrealtype *)malloc(n_states_total * sizeof(sunrealtype));
    if (!sens_data)
    {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }

    int values_idx = 0;
    int sparsity_idx = 0;

    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_stp * n_states; col++)
    {
        int param_idx = col; // Initial condition parameters come first

        // Copy sensitivity data from GPU for this column
        sunrealtype *d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_states_total * sizeof(sunrealtype),
                              cudaMemcpyDeviceToHost));

        // Determine number of entries for this column based on state type
        int entries_this_col;
        int state_type = col % n_states;
        if (state_type < n_quat)
        {
            entries_this_col = n_quat; // Quaternion affects 4 quaternions
        }
        else
        {
            entries_this_col = n_states; // Angular velocity affects all states
        }

        // Extract values for non-zero entries in this column
        for (int entry = 0; entry < entries_this_col; entry++)
        {
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
int DynamicsIntegrator::getSensitivitiesU(sunrealtype *values) const
{
    if (!sensitivity_enabled)
    {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!sens_was_setup || !values)
    {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }

    sunrealtype *sens_data = (sunrealtype *)malloc(n_states_total * sizeof(sunrealtype));
    if (!sens_data)
    {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }

    int values_idx = 0;
    int sparsity_idx = 0;

    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_controls_total; col++)
    {
        int param_idx = n_stp * n_states + col; // Control parameters start after initial conditions

        sunrealtype *d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_states_total * sizeof(sunrealtype),
                              cudaMemcpyDeviceToHost));

        // Each control affects all states of its system
        for (int entry = 0; entry < n_states; entry++)
        {
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
int DynamicsIntegrator::getSensitivitiesDt(sunrealtype *values) const
{
    if (!sensitivity_enabled)
    {
        cerr << "Sensitivity analysis not enabled" << endl;
        return -1;
    }
    if (!sens_was_setup || !values)
    {
        cerr << "Sparsity not computed or invalid values array" << endl;
        return -1;
    }

    sunrealtype *sens_data = (sunrealtype *)malloc(n_states_total * sizeof(sunrealtype));
    if (!sens_data)
    {
        cerr << "Error allocating temporary memory" << endl;
        return -1;
    }

    int values_idx = 0;
    int sparsity_idx = 0;

    // Extract values column by column using pre-computed sparsity
    for (int col = 0; col < n_stp; col++)
    {
        int param_idx = n_stp * (n_states + n_controls) + col; // dt parameters come last

        sunrealtype *d_yS = N_VGetDeviceArrayPointer_Cuda(yS[param_idx]);
        CUDA_CHECK(cudaMemcpy(sens_data, d_yS, n_states_total * sizeof(sunrealtype),
                              cudaMemcpyDeviceToHost));

        // Each dt affects all states of its system
        for (int entry = 0; entry < n_states; entry++)
        {
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
int DynamicsIntegrator::getSparsityY0(long long int *row_indices, long long int *col_pointers)
{

    computeSparsities(); // Ensure sparsity is computed

    if (!row_indices || !col_pointers)
    {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }

    // Copy row indices
    memcpy(row_indices, y0_row_idx, y0_nnz * sizeof(long long int));

    // Build column pointers
    int col_ptr_idx = 0;
    int sparsity_idx = 0;
    col_pointers[col_ptr_idx++] = 0;

    for (int col = 0; col < n_stp * n_states; col++)
    {
        int state_type = col % n_states;
        int entries_this_col = (state_type < n_quat) ? n_quat : n_states;
        sparsity_idx += entries_this_col;
        col_pointers[col_ptr_idx++] = sparsity_idx;
    }

    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂u Jacobian
 */
int DynamicsIntegrator::getSparsityU(long long int *row_indices, long long int *col_pointers) 
{
    computeSparsities(); // Ensure sparsity is computed
    if (!row_indices || !col_pointers)
    {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }

    // Copy row indices
    memcpy(row_indices, u_row_idx, u_nnz * sizeof(long long int));

    // Build column pointers
    col_pointers[0] = 0;
    for (int col = 0; col < n_controls_total; col++)
    {
        col_pointers[col + 1] = col_pointers[col] + n_states;
    }

    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂dt Jacobian
 */
int DynamicsIntegrator::getSparsityDt(long long int *row_indices, long long int *col_pointers) 
{
    computeSparsities(); // Ensure sparsity is computed

    if (!row_indices || !col_pointers)
    {
        cerr << "Invalid input arrays" << endl;
        return -1;
    }

    // Copy row indices
    memcpy(row_indices, dt_row_idx, dt_nnz * sizeof(long long int));

    // Build column pointers
    col_pointers[0] = 0;
    for (int col = 0; col < n_stp; col++)
    {
        col_pointers[col + 1] = col_pointers[col] + n_states;
    }

    return 0;
}

//==============================================================================
// SPARSITY SIZE QUERY METHODS
//==============================================================================

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂y_initial Jacobian
 */
int DynamicsIntegrator::getSparsitySizeY0() 
{
    computeSparsities(); // Ensure sparsity is computed
    return y0_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂u Jacobian
 */
int DynamicsIntegrator::getSparsitySizeU() 
{
    computeSparsities(); // Ensure sparsity is computed
    return u_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂dt Jacobian
 */
int DynamicsIntegrator::getSparsitySizeDt() 
{
    computeSparsities(); // Ensure sparsity is computed
    return dt_nnz;
}