#include <toac/cuda_rk4.h>

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
__device__ __constant__ double d_inertia_constants[12];

//==============================================================================
// RK4 INTEGRATION KERNEL
//==============================================================================

/**
 * @brief GPU kernel for 4th-order Runge-Kutta integration step
 *
 * Performs one RK4 step for all spacecraft systems in parallel.
 * Each thread processes one spacecraft system through all 4 RK stages.
 */
__global__ void rk4Step(double *y_in, double *y_out,
                        double *k1, double *k2, double *k3, double *k4,
                        double *torque_params, double *dt_params,
                        int n_states_total)
{

    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys >= n_stp)
        return;

    int base_idx = sys * n_states;
    double dt = dt_params[sys];
    double dt_half = dt * 0.5;
    double dt_sixth = dt / 6.0;

    // Load initial state
    double y0[n_states];
    for (int i = 0; i < n_states; i++)
    {
        y0[i] = y_in[base_idx + i];
    }

    // Stage 1: k1 = f(y0)
    computeDerivatives(sys, y0, k1 + base_idx, torque_params, dt);

    // Stage 2: k2 = f(y0 + dt/2 * k1)
    double y_temp[n_states];
    for (int i = 0; i < n_states; i++)
    {
        y_temp[i] = y0[i] + dt_half * k1[base_idx + i];
    }
    computeDerivatives(sys, y_temp, k2 + base_idx, torque_params, dt);

    // Stage 3: k3 = f(y0 + dt/2 * k2)
    for (int i = 0; i < n_states; i++)
    {
        y_temp[i] = y0[i] + dt_half * k2[base_idx + i];
    }
    computeDerivatives(sys, y_temp, k3 + base_idx, torque_params, dt);

    // Stage 4: k4 = f(y0 + dt * k3)
    for (int i = 0; i < n_states; i++)
    {
        y_temp[i] = y0[i] + dt * k3[base_idx + i];
    }
    computeDerivatives(sys, y_temp, k4 + base_idx, torque_params, dt);

    // Final combination: y_new = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (int i = 0; i < n_states; i++)
    {
        y_out[base_idx + i] = y0[i] + dt_sixth * (k1[base_idx + i] +
                                                  2.0 * k2[base_idx + i] +
                                                  2.0 * k3[base_idx + i] +
                                                  k4[base_idx + i]);
    }
}

/**
 * @brief Device function to compute derivatives for RK4 stages
 *
 * Extracted from original dynamicsRHS kernel for reuse in RK4 stages.
 * Note: dt scaling removed since RK4 handles time step multiplication.
 */
__device__ void computeDerivatives(int sys, double *y_local, double *ydot_local,
                                   double *torque_params, double dt)
{

    // Load constants from constant memory
    const double Ix = d_inertia_constants[0];
    const double Iy = d_inertia_constants[1];
    const double Iz = d_inertia_constants[2];
    const double half = d_inertia_constants[3];
    const double Ix_inv = d_inertia_constants[4];
    const double Iy_inv = d_inertia_constants[5];
    const double Iz_inv = d_inertia_constants[6];

    // Extract state variables
    double q0 = y_local[0], q1 = y_local[1];
    double q2 = y_local[2], q3 = y_local[3];
    double wx = y_local[4], wy = y_local[5], wz = y_local[6];

    // Load control parameters
    double tau_x = torque_params[sys * n_controls + 0];
    double tau_y = torque_params[sys * n_controls + 1];
    double tau_z = torque_params[sys * n_controls + 2];

    // Quaternion kinematics (NO dt scaling - RK4 handles it)
    ydot_local[0] = half * (-q1 * wx - q2 * wy - q3 * wz);
    ydot_local[1] = half * (q0 * wx - q3 * wy + q2 * wz);
    ydot_local[2] = half * (q3 * wx + q0 * wy - q1 * wz);
    ydot_local[3] = half * (-q2 * wx + q1 * wy + q0 * wz);

    // Euler's equations (NO dt scaling - RK4 handles it)
    double Iw_x = Ix * wx, Iw_y = Iy * wy, Iw_z = Iz * wz;
    ydot_local[4] = Ix_inv * (tau_x - (wy * Iw_z - wz * Iw_y));
    ydot_local[5] = Iy_inv * (tau_y - (wz * Iw_x - wx * Iw_z));
    ydot_local[6] = Iz_inv * (tau_z - (wx * Iw_y - wy * Iw_x));
}

//==============================================================================
// CONSTRUCTOR
//==============================================================================

CudaRk4::CudaRk4() : setup_time(0), solve_time(0),
                                           d_y(nullptr), d_k1(nullptr), d_k2(nullptr), d_k3(nullptr), d_k4(nullptr),
                                           h_y_temp(nullptr), h_tau_temp(nullptr), h_dt_temp(nullptr), compute_stream(nullptr),
                                           y0_sens_values(nullptr), u_sens_values(nullptr), dt_sens_values(nullptr),
                                           d_torque_params_ptr(nullptr), d_dt_params_ptr(nullptr),
                                           h_y_pinned(nullptr), h_tau_pinned(nullptr), h_dt_pinned(nullptr),
                                           fd_sens_was_setup(false), y0_col_ptr(nullptr), u_col_ptr(nullptr), dt_col_ptr(nullptr),
                                           y0_row_idx(nullptr), u_row_idx(nullptr), dt_row_idx(nullptr), spar_computed(false)
{

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    try
    {
        //======================================================================
        // Initialize CUDA constant memory
        //======================================================================

        double h_constants[12] = {
            i_x, i_y, i_z, 0.5,
            1.0 / i_x, 1.0 / i_y, 1.0 / i_z,
            (i_z - i_y), (i_x - i_z), (i_y - i_x),
            -0.5, 0.0};

        CUDA_CHECK(cudaMemcpyToSymbol(d_inertia_constants, h_constants,
                                      12 * sizeof(double)));

        //======================================================================
        // Allocate device memory for RK4 integration
        //======================================================================

        // State vectors
        CUDA_CHECK(cudaMalloc(&d_y, n_states_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1, n_states_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2, n_states_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3, n_states_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4, n_states_total * sizeof(double)));

        // Parameter arrays (UNCHANGED)
        CUDA_CHECK(cudaMalloc(&d_torque_params_ptr,
                              n_controls_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dt_params_ptr, n_stp * sizeof(double)));

        //======================================================================
        // Allocate pinned host memory
        //======================================================================

        CUDA_CHECK(cudaMallocHost(&h_y_pinned, n_states_total * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_tau_pinned, n_controls_total * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_dt_pinned, n_stp * sizeof(double)));

        //======================================================================
        // Create CUDA stream for asynchronous operations
        //======================================================================

        CUDA_CHECK(cudaStreamCreate(&compute_stream));
    }
    catch (const runtime_error &e)
    {
        cleanup();
        throw;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setup_time, start, stop);
}

void CudaRk4::setupFDSensitivities()
{
    if (fd_sens_was_setup)
        return;

    computeSparsityPatterns();

    h_y_temp = (double *)malloc(n_states_total * sizeof(double));
    h_tau_temp = (double *)malloc(n_controls_total * sizeof(double));
    h_dt_temp = (double *)malloc(n_stp * sizeof(double));

    // Allocate sensitivity value arrays
    y0_sens_values = (double *)malloc(y0_nnz * sizeof(double));
    u_sens_values = (double *)malloc(u_nnz * sizeof(double));
    dt_sens_values = (double *)malloc(dt_nnz * sizeof(double));

    fd_sens_was_setup = true;
}

/**
 * @brief Destructor: Clean up all allocated GPU and SUNDIALS resources
 */
CudaRk4::~CudaRk4()
{
    cleanup();
}

//==============================================================================
// MAIN SOLVE METHOD - RK4 INTEGRATION
//==============================================================================

int CudaRk4::solve(const double *initial_states,
                              const double *torque_params,
                              const double *delta_t)
{

    cudaEventRecord(start);

    // Transfer data to GPU
    setInitialConditions(initial_states, torque_params, delta_t);

    // Configure kernel launch parameters
    int blockSize = n_states;
    int gridSize = n_stp;

    // Single RK4 step (unit time with dt scaling in kernel)
    rk4Step<<<gridSize, blockSize, 0, compute_stream>>>(
        d_y, d_y, // in-place update
        d_k1, d_k2, d_k3, d_k4,
        d_torque_params_ptr, d_dt_params_ptr,
        n_states_total);

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&solve_time, start, stop);

    return 0;
}

//==============================================================================
// FINITE DIFFERENCE SENSITIVITY COMPUTATION
//==============================================================================

int CudaRk4::computeFiniteDifferenceSensitivities(
    const double *initial_states,
    const double *torque_params,
    const double *delta_t,
    const double *baseline_solution,
    double epsilon)
{

    setupFDSensitivities();

    //==========================================================================
    // Step 1: Initial state sensitivities ∂y/∂y₀
    //==========================================================================
    for (int state_idx = 0; state_idx < n_states; state_idx++)
    {

        // Copy baseline and perturb this state for all systems
        memcpy(h_y_temp, initial_states, n_states_total * sizeof(double));
        for (int sys = 0; sys < n_stp; sys++)
        {
            h_y_temp[sys * n_states + state_idx] += epsilon;
        }

        // Solve with perturbed initial conditions
        int result = solve(h_y_temp, torque_params, delta_t);
        if (result != 0)
            return result;

        // Compute finite differences and store in CCS format
        extractY0Sensitivities(state_idx, epsilon, baseline_solution);
    }

    //==========================================================================
    // Step 2: Control parameter sensitivities ∂y/∂u
    //==========================================================================
    for (int torque_idx = 0; torque_idx < n_controls; torque_idx++)
    {

        // Copy baseline and perturb this torque for all systems
        memcpy(h_tau_temp, torque_params, n_controls_total * sizeof(double));
        for (int sys = 0; sys < n_stp; sys++)
        {
            h_tau_temp[sys * n_controls + torque_idx] += epsilon;
        }

        // Solve with perturbed controls
        int result = solve(initial_states, h_tau_temp, delta_t);
        if (result != 0)
            return result;

        // Extract sensitivities
        extractUSensitivities(torque_idx, epsilon, baseline_solution);
    }

    //==========================================================================
    // Step 3: Time step sensitivities ∂y/∂dt
    //==========================================================================

    // Perturb all time steps simultaneously
    for (int sys = 0; sys < n_stp; sys++)
    {
        h_dt_temp[sys] = delta_t[sys] + epsilon;
    }

    int result = solve(initial_states, torque_params, h_dt_temp);
    if (result == 0)
    {
        extractDtSensitivities(epsilon, baseline_solution);
    }

    return result;
}

//==============================================================================
// FINITE DIFFERENCE SENSITIVITY EXTRACTION METHODS
//==============================================================================

void CudaRk4::extractY0Sensitivities(int state_idx, double epsilon,
                                      const double *baseline_solution)
{
    getSolution(h_y_temp);

    // Extract gradients for all systems' corresponding state columns
    for (int sys = 0; sys < n_stp; sys++)
    {
        int col = sys * n_states + state_idx;  // Full system column index
        int values_start = y0_col_ptr[col];
        int values_end = y0_col_ptr[col + 1];

        for (int idx = values_start; idx < values_end; idx++)
        {
            int row = y0_row_idx[idx];
            double fd_grad = (h_y_temp[row] - baseline_solution[row]) / epsilon;
            y0_sens_values[idx] = fd_grad;
        }
    }
}

void CudaRk4::extractUSensitivities(int torque_idx, double epsilon,
                                     const double *baseline_solution)
{
    getSolution(h_y_temp);

    // Extract gradients for all systems' corresponding control columns  
    for (int sys = 0; sys < n_stp; sys++)
    {
        int col = sys * n_controls + torque_idx;  // Full system column index
        int values_start = u_col_ptr[col];
        int values_end = u_col_ptr[col + 1];

        for (int idx = values_start; idx < values_end; idx++)
        {
            int row = u_row_idx[idx];
            double fd_grad = (h_y_temp[row] - baseline_solution[row]) / epsilon;
            u_sens_values[idx] = fd_grad;
        }
    }
}

void CudaRk4::extractDtSensitivities(double epsilon,
                                                const double *baseline_solution)
{

    getSolution(h_y_temp);

    // All systems perturbed simultaneously, extract all sensitivities
    for (int sys = 0; sys < n_stp; sys++)
    {
        int col = sys;
        int values_start = dt_col_ptr[col];
        int values_end = dt_col_ptr[col + 1];

        for (int idx = values_start; idx < values_end; idx++)
        {
            int row = dt_row_idx[idx];
            double fd_grad = (h_y_temp[row] - baseline_solution[row]) / epsilon;
            dt_sens_values[idx] = fd_grad;
        }
    }
}

//==============================================================================
// CCS SPARSITY PATTERN COMPUTATION
//==============================================================================

void CudaRk4::computeSparsityPatterns()
{
    if (spar_computed)
        return; // Already computed

    //==========================================================================
    // ∂y/∂y₀ sparsity pattern (state-to-state sensitivities)
    //==========================================================================

    // Count non-zeros first
    y0_nnz = nnz * n_stp;



    // Allocate internal storage
    y0_col_ptr = (long long int *)malloc((n_states_total + 1) * sizeof(long long int));
    y0_row_idx = (long long int *)malloc(y0_nnz * sizeof(long long int));
    if (!y0_row_idx || !y0_col_ptr)
    {
        cerr << "Error allocating memory for Y0 sparsity" << endl;
        return;
    }

    // Build column pointers
    int col_ptr_idx = 0;
    int sparsity_idx = 0;
    y0_col_ptr[col_ptr_idx++] = 0;

    for (int col = 0; col < n_states_total; col++)
    {
        int state_type = col % n_states;
        int entries_this_col = (state_type < n_quat) ? n_quat : n_states;
        sparsity_idx += entries_this_col;
        y0_col_ptr[col_ptr_idx++] = sparsity_idx;
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
    u_col_ptr = (long long int *)malloc((n_controls_total + 1) * sizeof(long long int));
    u_row_idx = (long long int *)malloc(u_nnz * sizeof(long long int));
    if (!u_row_idx || !u_col_ptr)
    {
        cerr << "Error allocating memory for U sparsity" << endl;
        return;
    }

    // Build column pointers
    col_ptr_idx = 0;
    sparsity_idx = 0;
    u_col_ptr[col_ptr_idx++] = 0;

    for (int col = 0; col < n_controls_total; col++)
    {
        sparsity_idx += n_states;
        u_col_ptr[col_ptr_idx++] = sparsity_idx;
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
    dt_col_ptr = (long long int *)malloc((n_stp + 1) * sizeof(long long int));
    dt_row_idx = (long long int *)malloc(dt_nnz * sizeof(long long int));
    if (!dt_row_idx || !dt_col_ptr)
    {
        cerr << "Error allocating memory for dt sparsity" << endl;
        return;
    }

    // Build column pointers
    col_ptr_idx = 0;
    sparsity_idx = 0;
    dt_col_ptr[col_ptr_idx++] = 0;

    for (int col = 0; col < n_stp; col++)
    {
        sparsity_idx += n_states;
        dt_col_ptr[col_ptr_idx++] = sparsity_idx;
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
    spar_computed = true; // Mark sparsity as computed
}

//==============================================================================
// SPARSITY SIZE QUERY METHODS
//==============================================================================

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂y_initial Jacobian
 */
int CudaRk4::getSparsitySizeY0()
{
    computeSparsityPatterns(); // Ensure sparsity is computed
    return y0_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂u Jacobian
 */
int CudaRk4::getSparsitySizeU()
{
    computeSparsityPatterns(); // Ensure sparsity is computed
    return u_nnz;
}

/**
 * @brief Get the number of non-zero entries in ∂y_final/∂dt Jacobian
 */
int CudaRk4::getSparsitySizeDt()
{
    computeSparsityPatterns(); // Ensure sparsity is computed
    return dt_nnz;
}

//==============================================================================
// SENSITIVITY ACCESS METHODS
//==============================================================================

int CudaRk4::getSensitivitiesY0(double *values)
{
    if (!values) return -1;
    if (!spar_computed) computeSparsityPatterns();

    memcpy(values, y0_sens_values, y0_nnz * sizeof(double));
    return 0;
}

int CudaRk4::getSensitivitiesU(double *values)
{
    if (!values) return -1;
    if (!spar_computed) computeSparsityPatterns();

    memcpy(values, u_sens_values, u_nnz * sizeof(double));
    return 0;
}

int CudaRk4::getSensitivitiesDt(double *values)
{
    if (!values) return -1;
    if (!spar_computed) computeSparsityPatterns();

    memcpy(values, dt_sens_values, dt_nnz * sizeof(double));
    return 0;
}

//==============================================================================
// CLEANUP METHOD
//==============================================================================

void CudaRk4::cleanup()
{

    // Device memory
    if (d_y)
    {
        cudaFree(d_y);
        d_y = nullptr;
    }
    if (d_k1)
    {
        cudaFree(d_k1);
        d_k1 = nullptr;
    }
    if (d_k2)
    {
        cudaFree(d_k2);
        d_k2 = nullptr;
    }
    if (d_k3)
    {
        cudaFree(d_k3);
        d_k3 = nullptr;
    }
    if (d_k4)
    {
        cudaFree(d_k4);
        d_k4 = nullptr;
    }
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

    // Host memory
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
    if (h_y_temp)
    {
        cudaFreeHost(h_y_temp);
        h_y_temp = nullptr;
    }
    if (h_tau_temp)
    {
        free(h_tau_temp);
        h_tau_temp = nullptr;
    }
    if (h_dt_temp)
    {
        free(h_dt_temp);
        h_dt_temp = nullptr;
    }

    // Sparsity storage
    if (y0_row_idx)
    {
        free(y0_row_idx);
        y0_row_idx = nullptr;
    }
    if (y0_col_ptr)
    {
        free(y0_col_ptr);
        y0_col_ptr = nullptr;
    }
    if (u_row_idx)
    {
        free(u_row_idx);
        u_row_idx = nullptr;
    }
    if (u_col_ptr)
    {
        free(u_col_ptr);
        u_col_ptr = nullptr;
    }
    if (dt_row_idx)
    {
        free(dt_row_idx);
        dt_row_idx = nullptr;
    }
    if (dt_col_ptr)
    {
        free(dt_col_ptr);
        dt_col_ptr = nullptr;
    }
    if (y0_sens_values)
    {
        free(y0_sens_values);
        y0_sens_values = nullptr;
    }
    if (u_sens_values)
    {
        free(u_sens_values);
        u_sens_values = nullptr;
    }
    if (dt_sens_values)
    {
        free(dt_sens_values);
        dt_sens_values = nullptr;
    }

    if (compute_stream)
    {
        cudaStreamDestroy(compute_stream);
        compute_stream = nullptr;
    }
    if (start)
    {
        cudaEventDestroy(start);
        start = nullptr;
    }
    if (stop)
    {
        cudaEventDestroy(stop);
        stop = nullptr;
    }
}

/**
 * @brief Transfer initial conditions and parameters to GPU memory
 */
void CudaRk4::setInitialConditions(const double *initial_states,
                                              const double *torque_params,
                                              const double *delta_t)
{

    // Copy to pinned memory for fast GPU transfer
    memcpy(h_y_pinned, initial_states, n_states_total * sizeof(double));
    memcpy(h_tau_pinned, torque_params, n_controls_total * sizeof(double));
    memcpy(h_dt_pinned, delta_t, n_stp * sizeof(double));

    // Transfer to GPU memory
    CUDA_CHECK(cudaMemcpy(d_y, h_y_pinned, n_states_total * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_torque_params_ptr, h_tau_pinned,
                          n_controls_total * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_dt_params_ptr, h_dt_pinned,
                          n_stp * sizeof(double),
                          cudaMemcpyHostToDevice));
}

/**
 * @brief Copy final integrated states from GPU to caller-allocated buffer
 */
int CudaRk4::getSolution(double *next_state) const
{
    if (!next_state) return -1;

    CUDA_CHECK(cudaMemcpy(next_state, d_y, n_states_total * sizeof(double),
                          cudaMemcpyDeviceToHost));
    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂y_initial Jacobian
 */
int CudaRk4::getSparsityY0(long long int *row_indices,
                                      long long int *col_pointers)
{
    if (!row_indices || !col_pointers)
        return -1;

    if (!spar_computed)
        computeSparsityPatterns();

    // Copy row indices
    memcpy(row_indices, y0_row_idx,
           y0_nnz * sizeof(long long int));

    // Copy column pointers
    memcpy(col_pointers, y0_col_ptr,
           (n_states_total + 1) * sizeof(long long int));

    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂u Jacobian
 */
int CudaRk4::getSparsityU(long long int *row_indices,
                                     long long int *col_pointers)
{
    if (!row_indices || !col_pointers)
        return -1;

    if (!spar_computed)
        computeSparsityPatterns();

    memcpy(row_indices, u_row_idx,
           u_nnz * sizeof(long long int));

    memcpy(col_pointers, u_col_ptr,
           (n_controls_total + 1) * sizeof(long long int));

    return 0;
}

/**
 * @brief Get sparsity pattern for ∂y_final/∂dt Jacobian
 */
int CudaRk4::getSparsityDt(long long int *row_indices,
                                      long long int *col_pointers)
{
    if (!row_indices || !col_pointers)
        return -1;

    if (!spar_computed)
        computeSparsityPatterns();

    memcpy(row_indices, dt_row_idx,
           dt_nnz * sizeof(long long int));

    memcpy(col_pointers, dt_col_ptr,
           (n_stp + 1) * sizeof(long long int));

    return 0;
}