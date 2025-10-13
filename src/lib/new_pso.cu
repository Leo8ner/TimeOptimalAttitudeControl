/**
 * @file pso_optimizer.cu
 * @brief C++ Class Implementation for CUDA-accelerated PSO Spacecraft Attitude Control
 *
 * This implementation provides the complete class functionality for PSO optimization
 * of spacecraft attitude maneuvers with clean input/output interfaces.
 *
 * @author Leonardo Eitner
 * @date 11/09/2025
 * @version 2.0
 */

/*==============================================================================
 * INCLUDES
 *============================================================================*/
#include <toac/pso.h>

/*==============================================================================
 * CUDA CONSTANT MEMORY DECLARATIONS
 *============================================================================*/

/** @brief PSO algorithm parameters in device constant memory */
__constant__ real w_d, c1_d, c2_d;

/** @brief Physical constraint bounds in device constant memory */
__constant__ real max_torque_d, min_torque_d;
__constant__ real max_dt_d, min_dt_d;

/** @brief PSO velocity limits in device constant memory */
__constant__ real max_v_torque_d, max_v_dt_d;

/** @brief Problem dimensions in device constant memory */
__constant__ int particle_cnt_d, dimensions_d;

/** @brief Complete attitude parameters structure in device constant memory */
__constant__ attitude_params att_params_d;

/*==============================================================================
 * MATHEMATICAL UTILITY FUNCTIONS (CUDA KERNELS)
 *============================================================================*/

__host__ __device__ void skew_matrix_4(real *w, real *S)
{
    S[0] = REAL(0.0);
    S[1] = -w[0];
    S[2] = -w[1];
    S[3] = -w[2];
    S[4] = w[0];
    S[5] = REAL(0.0);
    S[6] = w[2];
    S[7] = -w[1];
    S[8] = w[1];
    S[9] = -w[2];
    S[10] = REAL(0.0);
    S[11] = w[0];
    S[12] = w[2];
    S[13] = w[1];
    S[14] = -w[0];
    S[15] = REAL(0.0);
}

__host__ __device__ void cross_product(real *a, real *b, real *result)
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

__host__ __device__ real quaternion_norm(real *q)
{
    return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

__host__ __device__ void attitude_dynamics(real *X, real *U, real *X_dot, attitude_params *params)
{
    real *q = X;
    real *w = &X[n_quat];

    // Quaternion kinematics: q̇ = 0.5 * S(ω) * q
    real S[16];
    skew_matrix_4(w, S);
    for (int i = 0; i < n_quat; i++)
    {
        X_dot[i] = REAL(0.5) * (S[i * 4] * q[0] + S[i * 4 + 1] * q[1] + S[i * 4 + 2] * q[2] + S[i * 4 + 3] * q[3]);
    }

    // Angular dynamics: ω̇ = I⁻¹ * (τ - ω × (I*ω))
    real Iw[n_vel] = {
        params->inertia[0] * w[0],
        params->inertia[1] * w[1],
        params->inertia[2] * w[2]};

    real w_cross_Iw[n_vel];
    cross_product(w, Iw, w_cross_Iw);

    for (int i = 0; i < n_vel; i++)
    {
        X_dot[n_quat + i] = (U[i] - w_cross_Iw[i]) / params->inertia[i];
    }
}

/*==============================================================================
 * NUMERICAL INTEGRATION SCHEMES
 *============================================================================*/

/**
 * @brief Fourth-order Runge-Kutta integration for attitude dynamics
 *
 * Provides high-accuracy numerical integration using the classical RK4 method:
 * k₁ = f(t, y)
 * k₂ = f(t + h/2, y + h*k₁/2)
 * k₃ = f(t + h/2, y + h*k₂/2)
 * k₄ = f(t + h, y + h*k₃)
 * y_{n+1} = y_n + h/6 * (k₁ + 2*k₂ + 2*k₃ + k₄)
 *
 * Used for final trajectory generation where accuracy is critical.
 * Automatically normalizes quaternion to maintain unit constraint.
 *
 * @param X Current state vector [q,ω] (7 elements)
 * @param U Control input vector [τx,τy,τz] (3 elements)
 * @param dt Integration time step (seconds)
 * @param X_next Output next state vector (7 elements)
 * @param params Spacecraft parameters
 */
__host__ __device__ void rk4(real *X, real *U, real dt, real *X_next, attitude_params *params)
{
    real k1[n_states], k2[n_states], k3[n_states], k4[n_states];
    real X_temp[n_states];

    // k1 = f(X, U)
    attitude_dynamics(X, U, k1, params);

    // k2 = f(X + dt/2*k1, U)
    for (int i = 0; i < n_states; i++)
        X_temp[i] = X[i] + dt / REAL(2.0) * k1[i];
    attitude_dynamics(X_temp, U, k2, params);

    // k3 = f(X + dt/2*k2, U)
    for (int i = 0; i < n_states; i++)
        X_temp[i] = X[i] + dt / REAL(2.0) * k2[i];
    attitude_dynamics(X_temp, U, k3, params);

    // k4 = f(X + dt*k3, U)
    for (int i = 0; i < n_states; i++)
        X_temp[i] = X[i] + dt * k3[i];
    attitude_dynamics(X_temp, U, k4, params);

    // Final integration step
    for (int i = 0; i < n_states; i++)
    {
        X_next[i] = X[i] + dt / REAL(6.0) * (k1[i] + REAL(2.0) * k2[i] + REAL(2.0) * k3[i] + k4[i]);
    }

    // Maintain quaternion unit constraint
    real q_norm = quaternion_norm(X_next);
    if (q_norm > REAL(1e-6))
    {
        for (int i = 0; i < n_quat; i++)
            X_next[i] /= q_norm;
    }
}

__host__ __device__ void euler(real *X, real *U, real dt, real *X_next, attitude_params *params)
{
    real X_dot[n_states];

    attitude_dynamics(X, U, X_dot, params);

    for (int i = 0; i < n_states; i++)
    {
        X_next[i] = X[i] + dt * X_dot[i];
    }

    real q_norm = quaternion_norm(X_next);
    if (q_norm > REAL(1e-6))
    {
        for (int i = 0; i < n_quat; i++)
            X_next[i] /= q_norm;
    }
}

__host__ __device__ real fit(real *solution_vector, int particle_id, attitude_params *params)
{
    real dt = solution_vector[PARTICLE_POS_IDX(particle_id, DT_IDX)];
    real total_time = dt * N_STEPS;

    // Decode initial control signs (map [-1,1] to {-1,+1})
    real initial_signs[n_controls];
    for (int axis = 0; axis < n_controls; axis++)
    {
        initial_signs[axis] = solution_vector[PARTICLE_POS_IDX(particle_id, SIGN_IDX(axis))];
        // real sign_val = solution_vector[PARTICLE_POS_IDX(particle_id, SIGN_IDX(axis))];
        // initial_signs[axis] = (sign_val >= REAL(0.0)) ? REAL(1.0) : REAL(-1.0);
    }

    // Decode and sort switch times for each axis (normalized to [0,1])
    real switch_times[n_controls][MAX_SWITCHES_PER_AXIS];
    int num_switches[n_controls];

    for (int axis = 0; axis < n_controls; axis++)
    {
        // Extract switch times for this axis
        real times[MAX_SWITCHES_PER_AXIS];
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++)
        {
            times[s] = solution_vector[PARTICLE_POS_IDX(particle_id, SWITCH_TIME_IDX(axis, s))];
            // Clamp to [0,1]
            times[s] = fmax(REAL(0.0), fmin(REAL(1.0), times[s]));
        }

        // Bubble sort (simple, efficient for small arrays)
        for (int i = 0; i < MAX_SWITCHES_PER_AXIS - 1; i++)
        {
            for (int j = 0; j < MAX_SWITCHES_PER_AXIS - i - 1; j++)
            {
                if (times[j] > times[j + 1])
                {
                    real temp = times[j];
                    times[j] = times[j + 1];
                    times[j + 1] = temp;
                }
            }
        }

        // Remove duplicates and count valid switches
        num_switches[axis] = 0;
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++)
        {
            // Only count if different from previous and not at boundaries
            if (times[s] > REAL(0.01) && times[s] < REAL(0.99))
            {
                if (num_switches[axis] == 0 ||
                    fabs(times[s] - switch_times[axis][num_switches[axis] - 1]) > REAL(0.01))
                {
                    switch_times[axis][num_switches[axis]] = times[s];
                    num_switches[axis]++;
                }
            }
        }
    }

    // Simulate trajectory with bang-bang control
    real constraints_violation = REAL(0.0);
    real X[n_states], X_next[n_states];

    for (int i = 0; i < n_quat; i++)
        X[i] = params->initial_quat[i];
    for (int i = 0; i < n_vel; i++)
        X[n_quat + i] = params->initial_omega[i];

    real current_time = REAL(0.0);

    for (int step = 0; step < N_STEPS; step++)
    {
        real step_start_time = current_time / total_time; // Normalized time [0,1]
        real U[n_controls];

        // Determine control for each axis based on switch times
        for (int axis = 0; axis < n_controls; axis++)
        {
            real control_sign = initial_signs[axis];

            // Count how many switches have occurred by this time
            for (int s = 0; s < num_switches[axis]; s++)
            {
                if (step_start_time >= switch_times[axis][s])
                {
                    control_sign *= REAL(-1.0); // Flip sign at each switch
                }
            }

            // Apply bang-bang control at max torque
            U[axis] = control_sign * params->max_torque;
        }

        rk4(X, U, dt, X_next, params);

        real q_norm = quaternion_norm(X_next);
        constraints_violation -= QUAT_NORM_PENALTY * fabs(q_norm - REAL(1.0));

        for (int i = 0; i < n_states; i++)
            X[i] = X_next[i];
        current_time += dt;
    }

    // Final state error
    real final_error = REAL(0.0);
    real diff;
    for (int i = 0; i < n_quat; i++)
    {
        diff = X[i] - params->target_quat[i];
        final_error += diff * diff;
    }
    for (int i = 0; i < n_vel; i++)
    {
        diff = X[n_quat + i] - params->target_omega[i];
        final_error += diff * diff;
    }
    final_error = sqrt(final_error);

    constraints_violation -= FINAL_STATE_PENALTY * final_error;
    constraints_violation -= DT_PENALTY * total_time;

    return constraints_violation;
}

/*==============================================================================
 * CUDA KERNEL IMPLEMENTATIONS
 *============================================================================*/

__global__ void move(real *position_d, real *velocity_d, real *fitness_d,
                     real *pbest_pos_d, real *pbest_fit_d,
                     particle_gbest *gbest_d, real *aux, real *aux_pos)
{

    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    extern __shared__ real sharedMemory[];
    real *privateBestQueue = (real *)sharedMemory;
    int *privateBestParticleQueue = (int *)&sharedMemory[blockDim.x];
    __shared__ unsigned int queue_num;

    if (particle_id >= particle_cnt_d)
        return;

    if (tidx == 0)
        queue_num = 0;
    __syncthreads();

    curandState state1, state2;
    curand_init((unsigned long long)clock() + particle_id * 2, 0, 0, &state1);
    curand_init((unsigned long long)clock() + particle_id * 2 + 1, 0, 0, &state2);

    real w = w_d, c1 = c1_d, c2 = c2_d;
    if (DEC_INERTIA)
    {
        w = w_d - (w_d - MIN_W) * (real)particle_id / N_PARTICLES;
    }

    if (DEC_C1)
    {
        c1 = c1_d - (c1_d - MIN_C1) * (real)particle_id / N_PARTICLES;
    }

    if (DEC_C2)
    {
        c2 = c2_d - (c2_d - MIN_C2) * (real)particle_id / N_PARTICLES;
    }

    for (int dim = 0; dim < dimensions_d; dim++)
    {
        int pos_idx = PARTICLE_POS_IDX(particle_id, dim);

        real pos = position_d[pos_idx];
        real vel = velocity_d[pos_idx];
        real pbest_pos = pbest_pos_d[pos_idx];
        real gbest_pos = gbest_d->position[dim];

        vel = w * vel +
              c1 * CURAND(state1) * (pbest_pos - pos) +
              c2 * CURAND(state2) * (gbest_pos - pos);

        // Apply bounds based on dimension type
        if (dim < N_SIGNS)
        {   
            // Clamp velocity for binary dimensions
            vel = fmax(REAL(-6.0), fmin(REAL(6.0), vel));
            
            // Sigmoid transfer function: S(v) = 1/(1 + exp(-alpha*v))
            real sigmoid = REAL(1.0) / (REAL(1.0) + exp(-SIGMOID_ALPHA * vel));
            
            // Probabilistic position update
            if (CURAND(state1) < sigmoid) {
                pos = REAL(1.0);   // Positive control
            } else {
                pos = REAL(-1.0);  // Negative control
            }
            // Sign dimensions: keep in [-1, 1]
            // vel = fmax(REAL(-2.0), fmin(REAL(2.0), vel));
            // pos = pos + vel;
            // pos = fmax(REAL(-1.0), fmin(REAL(1.0), pos));
        }
        else if (dim < N_SIGNS + N_SWITCH_TIMES)
        {
            // Switch time dimensions: keep in [0, 1]
            vel = fmax(REAL(-1.0), fmin(REAL(1.0), vel));
            pos = pos + vel;
            pos = fmax(REAL(0.0), fmin(REAL(1.0), pos));
        }
        else
        {
            // dt dimension: use existing bounds
            vel = fmax(-max_v_dt_d, fmin(max_v_dt_d, vel));
            pos = pos + vel;
            pos = fmax(min_dt_d, fmin(max_dt_d, pos));
        }

        position_d[pos_idx] = pos;
        velocity_d[pos_idx] = vel;
    }

    real new_fitness = fit(position_d, particle_id, &att_params_d);
    fitness_d[particle_id] = new_fitness;

    if (new_fitness > pbest_fit_d[particle_id])
    {
        pbest_fit_d[particle_id] = new_fitness;
        for (int dim = 0; dim < dimensions_d; dim++)
        {
            pbest_pos_d[PARTICLE_POS_IDX(particle_id, dim)] =
                position_d[PARTICLE_POS_IDX(particle_id, dim)];
        }
    }

    __syncthreads();

    if (new_fitness > gbest_d->fitness)
    {
        unsigned int my_index = atomicAdd(&queue_num, 1);
        if (my_index < blockDim.x)
        {
            privateBestQueue[my_index] = new_fitness;
            privateBestParticleQueue[my_index] = particle_id;
        }
    }

    __syncthreads();

    if (tidx == 0)
    {
        aux[blockIdx.x] = -REAL_MAX;
        aux_pos[blockIdx.x] = -1;

        if (queue_num > 0)
        {
            real best_fitness = privateBestQueue[0];
            int best_idx = 0;

            for (unsigned int i = 1; i < queue_num && i < blockDim.x; i++)
            {
                if (privateBestQueue[i] > best_fitness)
                {
                    best_fitness = privateBestQueue[i];
                    best_idx = i;
                }
            }
            aux[blockIdx.x] = best_fitness;
            aux_pos[blockIdx.x] = privateBestParticleQueue[best_idx];
        }
    }
}

__global__ void findBest(particle_gbest *gbest, real *aux, real *aux_pos, real *position_d)
{
    int tid = threadIdx.x;

    real my_fitness = (tid < BlocksPerGrid) ? aux[tid] : -REAL_MAX;
    int my_particle = (tid < BlocksPerGrid) ? (int)aux_pos[tid] : -1;

    for (int offset = 16; offset > 0; offset /= 2)
    {
        real other_fitness = __shfl_down_sync(0xffffffff, my_fitness, offset);
        int other_particle = __shfl_down_sync(0xffffffff, my_particle, offset);
        if (other_fitness > my_fitness)
        {
            my_fitness = other_fitness;
            my_particle = other_particle;
        }
    }

    if (tid == 0 && my_fitness > gbest->fitness)
    {
        gbest->fitness = my_fitness;
        if (my_particle >= 0)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
            {
                gbest->position[dim] = position_d[PARTICLE_POS_IDX(my_particle, dim)];
            }
        }
        __threadfence();
    }
}

/*==============================================================================
 * PSO OPTIMIZER CLASS IMPLEMENTATION
 *============================================================================*/
PSOOptimizer::PSOOptimizer(casadi::DM &state_matrix, casadi::DM &input_matrix, casadi::DM &dt_matrix, bool verbose)
    : configured_(false), results_valid_(false), max_iterations_(ITERATIONS), num_particles_(N_PARTICLES), inertia_weight_(W), cognitive_weight_(C1), social_weight_(C2), particles_(nullptr), position_d_(nullptr), velocity_d_(nullptr), fitness_d_(nullptr), pbest_pos_d_(nullptr), pbest_fit_d_(nullptr), gbest_d_(nullptr), aux_(nullptr), aux_pos_(nullptr), lhs_samples_(nullptr), lhs_generated_(false), verbose_(verbose), X(state_matrix), U(input_matrix), dt(dt_matrix)
{
    // Initialize CUDA events
    if (!handleCudaError(cudaEventCreate(&start_event_), __FILE__, __LINE__) ||
        !handleCudaError(cudaEventCreate(&stop_event_), __FILE__, __LINE__))
    {
        std::cerr << "Failed to create CUDA events" << std::endl;
    }

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    if ((X.size1() != n_states || X.size2() != N_STEPS + 1) ||
        (U.size1() != n_controls || U.size2() != N_STEPS) ||
        (dt.size1() != N_STEPS || dt.size2() != 1))
    {
        std::cerr << "Error: Output matrices have incorrect dimensions." << std::endl;
        std::cerr << "Expected dimensions | Given dimensions: "
                  << "X: (" << n_states << ", " << N_STEPS + 1 << ") | (" << X.size1() << ", " << X.size2() << "), "
                  << "U: (" << n_controls << ", " << N_STEPS << ") | (" << U.size1() << ", " << U.size2() << "), "
                  << "dt: (" << N_STEPS << ", 1) | (" << dt.size1() << ", " << dt.size2() << ")" << std::endl;
        cleanup();
        return;
    }

    // Initialize attitude parameters with default spacecraft values
    memset(&att_params_, 0, sizeof(attitude_params));

    att_params_.inertia[0] = static_cast<real>(i_x);
    att_params_.inertia[1] = static_cast<real>(i_y);
    att_params_.inertia[2] = static_cast<real>(i_z);
    att_params_.max_torque = static_cast<real>(tau_max);
    att_params_.min_torque = -static_cast<real>(tau_max);
    att_params_.max_dt = static_cast<real>(dt_max);
    att_params_.min_dt = static_cast<real>(dt_min);

    // Initialize velocity limits
    max_v_torque_ = REAL(2.0) * att_params_.max_torque;
    max_v_dt_ = att_params_.max_dt - att_params_.min_dt;

    // Allocate LHS samples storage
    lhs_samples_ = new real *[num_particles_];
    for (int i = 0; i < num_particles_; i++)
    {
        lhs_samples_[i] = new real[DIMENSIONS];
    }

    // Allocate host particle structure (once)
    particles_ = (particle *)malloc(sizeof(particle));
    if (!particles_)
    {
        std::cerr << "Failed to allocate particle structure" << std::endl;
        cleanup();
        return;
    }

    particles_->position = (real *)malloc(sizeof(real) * num_particles_ * DIMENSIONS);
    particles_->velocity = (real *)malloc(sizeof(real) * num_particles_ * DIMENSIONS);
    particles_->pbest_pos = (real *)malloc(sizeof(real) * num_particles_ * DIMENSIONS);
    particles_->fitness = (real *)malloc(sizeof(real) * num_particles_);
    particles_->pbest_fit = (real *)malloc(sizeof(real) * num_particles_);

    if (!particles_->position || !particles_->velocity || !particles_->pbest_pos ||
        !particles_->fitness || !particles_->pbest_fit)
    {
        std::cerr << "Failed to allocate particle arrays" << std::endl;
        cleanup();
        return;
    }

    // One-time GPU initialization
    if (!allocateDeviceMemory() || !copyImmutableConstants())
    {
        std::cerr << "GPU initialization failed" << std::endl;
        cleanup();
        return;
    }

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&setup_time_, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ /= 1000.0f;
}

PSOOptimizer::~PSOOptimizer()
{
    cleanup();
}

bool PSOOptimizer::allocateDeviceMemory()
{
    size_t particle_data_size = sizeof(real) * num_particles_ * DIMENSIONS;

    if (!handleCudaError(cudaMalloc((void **)&position_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&velocity_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&pbest_pos_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&fitness_d_, sizeof(real) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&pbest_fit_d_, sizeof(real) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&gbest_d_, sizeof(particle_gbest)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&aux_, sizeof(real) * BlocksPerGrid), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void **)&aux_pos_, sizeof(real) * BlocksPerGrid), __FILE__, __LINE__))
    {
        return false;
    }

    return true;
}

void PSOOptimizer::setPSOParameters(int max_iterations,
                                    double inertia_weight, double cognitive_weight, double social_weight)
{

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    max_iterations_ = max_iterations;
    inertia_weight_ = static_cast<real>(inertia_weight);
    cognitive_weight_ = static_cast<real>(cognitive_weight);
    social_weight_ = static_cast<real>(social_weight);

    copyImmutableConstants();

    results_valid_ = false;
    // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);

    // Calculate execution time
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time; // Accumulate
}

bool PSOOptimizer::copyImmutableConstants()
{
    // Copy PSO parameters
    if (!handleCudaError(cudaMemcpyToSymbol(w_d, &inertia_weight_, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c1_d, &cognitive_weight_, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c2_d, &social_weight_, sizeof(real)), __FILE__, __LINE__))
    {
        return false;
    }

    // Copy physical constraints (immutable)
    if (!handleCudaError(cudaMemcpyToSymbol(max_torque_d, &att_params_.max_torque, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_torque_d, &att_params_.min_torque, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_dt_d, &att_params_.max_dt, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_dt_d, &att_params_.min_dt, sizeof(real)), __FILE__, __LINE__))
    {
        return false;
    }

    // Copy velocity limits
    if (!handleCudaError(cudaMemcpyToSymbol(max_v_torque_d, &max_v_torque_, sizeof(real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_v_dt_d, &max_v_dt_, sizeof(real)), __FILE__, __LINE__))
    {
        return false;
    }

    // Copy dimensions
    if (!handleCudaError(cudaMemcpyToSymbol(particle_cnt_d, &num_particles_, sizeof(int)), __FILE__, __LINE__))
    {
        return false;
    }

    int dimensions = DIMENSIONS;
    if (!handleCudaError(cudaMemcpyToSymbol(dimensions_d, &dimensions, sizeof(int)), __FILE__, __LINE__))
    {
        return false;
    }

    return true;
}

bool PSOOptimizer::copyMutableStateParameters()
{
    // Only copy the att_params structure (contains initial/target states)
    return handleCudaError(cudaMemcpyToSymbol(att_params_d, &att_params_, sizeof(attitude_params)), __FILE__, __LINE__);
}

void PSOOptimizer::setStates(const double *initial_state, const double *target_state)
{
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);
    // Update initial and target states
    for (int i = 0; i < n_quat; i++)
    {
        att_params_.initial_quat[i] = static_cast<real>(initial_state[i]);
        att_params_.target_quat[i] = static_cast<real>(target_state[i]);
    }
    for (int i = 0; i < n_vel; i++)
    {
        att_params_.initial_omega[i] = static_cast<real>(initial_state[i + n_quat]);
        att_params_.target_omega[i] = static_cast<real>(target_state[i + n_quat]);
    }

    // Copy updated parameters to device constant memory
    copyMutableStateParameters();

    results_valid_ = false;

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);

    // Calculate execution time
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time; // Accumulate
}

bool PSOOptimizer::initializeParticles(bool regenerate_lhs)
{
    srand((unsigned)time(NULL));

    // Generate or reuse LHS samples
    if (regenerate_lhs || !lhs_generated_)
    {
        generateLHSSamples(lhs_samples_);
        lhs_generated_ = true;
    }

    // Initialize global best
    gbest_.fitness = -REAL_MAX;

    // Initialize each particle using LHS samples
    int best_particle = 0;
    for (int i = 0; i < num_particles_; i++)
    {
        // Initialize sign variables [-1, 1]
        for (int dim = 0; dim < N_SIGNS; dim++)
        {
            int idx = PARTICLE_POS_IDX(i, dim);
            // particles_->position[idx] = lhs_samples_[i][dim] * REAL(2.0) - REAL(1.0); // Map [0,1] to [-1,1]
            particles_->position[idx] = (lhs_samples_[i][dim] > REAL(0.5)) ? REAL(1.0) : REAL(-1.0);
            particles_->velocity[idx] = (lhs_samples_[i][dim] - REAL(0.5)) * REAL(4.0);
            particles_->pbest_pos[idx] = particles_->position[idx];
        }

        // Initialize switch time variables [0, 1]
        for (int dim = N_SIGNS; dim < N_SIGNS + N_SWITCH_TIMES; dim++)
        {
            int idx = PARTICLE_POS_IDX(i, dim);
            particles_->position[idx] = lhs_samples_[i][dim]; // Already in [0,1]
            particles_->velocity[idx] = (lhs_samples_[i][dim] - REAL(0.5)) * REAL(2.0);
            particles_->pbest_pos[idx] = particles_->position[idx];
        }

        // Initialize dt (unchanged)
        int dt_idx = PARTICLE_POS_IDX(i, DT_IDX);
        real dt_range = att_params_.max_dt - att_params_.min_dt;
        particles_->position[dt_idx] = lhs_samples_[i][DT_IDX] * dt_range + att_params_.min_dt;
        particles_->velocity[dt_idx] = (lhs_samples_[i][DT_IDX] - REAL(0.5)) * REAL(2.0) * max_v_dt_;
        particles_->pbest_pos[dt_idx] = particles_->position[dt_idx];

        // Evaluate initial fitness
        particles_->fitness[i] = fit(particles_->position, i, &att_params_);
        particles_->pbest_fit[i] = particles_->fitness[i];

        // Track global best
        if (i == 0 || particles_->pbest_fit[i] > gbest_.fitness)
        {
            best_particle = i;
            gbest_.fitness = particles_->fitness[i];
        }
    }

    // Copy global best position
    for (int dim = 0; dim < DIMENSIONS; dim++)
    {
        int idx = PARTICLE_POS_IDX(best_particle, dim);
        gbest_.position[dim] = particles_->position[idx];
    }

    return true;
}

/**
 * Generate Latin Hypercube Samples for particle initialization
 * @param samples Output matrix [num_particles_][DIMENSIONS]
 */
void PSOOptimizer::generateLHSSamples(real **samples)
{
    // Generate LHS samples in [0,1] for each dimension
    for (int dim = 0; dim < DIMENSIONS; dim++)
    {
        std::vector<real> intervals(num_particles_);

        // Create stratified intervals
        for (int i = 0; i < num_particles_; i++)
        {
            real interval_start = static_cast<real>(i) / num_particles_;
            real interval_width = REAL(1.0) / num_particles_;
            intervals[i] = interval_start + RND() * interval_width;
        }

        // Shuffle to break correlations
        for (int i = num_particles_ - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);
            std::swap(intervals[i], intervals[j]);
        }

        // Assign to samples matrix
        for (int i = 0; i < num_particles_; i++)
        {
            samples[i][dim] = intervals[i];
        }
    }
}

bool PSOOptimizer::optimize(bool regenerate_lhs)
{

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    // Initialize particles (regenerate LHS based on parameter)
    if (!initializeParticles(regenerate_lhs))
    {
        std::cerr << "Particle initialization failed" << std::endl;
        return false;
    }

    // Copy initial data to device
    size_t particle_data_size = sizeof(real) * num_particles_ * DIMENSIONS;
    if (!handleCudaError(cudaMemcpy(position_d_, particles_->position, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(velocity_d_, particles_->velocity, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_pos_d_, particles_->pbest_pos, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(fitness_d_, particles_->fitness, sizeof(real) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_fit_d_, particles_->pbest_fit, sizeof(real) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(gbest_d_, &gbest_, sizeof(particle_gbest), cudaMemcpyHostToDevice), __FILE__, __LINE__))
    {
        return false;
    }

    if (verbose_)
    {
        std::cout << "Starting PSO optimization..." << std::endl;
        std::cout << "LHS: " << (regenerate_lhs ? "Regenerated" : "Reused") << std::endl;
        std::cout << "Initial best fitness: " << gbest_.fitness << std::endl;
    }

    // Main optimization loop
    int shared_mem_size = sizeof(real) * ThreadsPerBlock + sizeof(int) * ThreadsPerBlock;

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time / 1000.0f; // Convert ms to seconds
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    for (int iter = 0; iter < max_iterations_; iter++)
    {
        move<<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>(
            position_d_, velocity_d_, fitness_d_, pbest_pos_d_, pbest_fit_d_,
            gbest_d_, aux_, aux_pos_);

        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__))
        {
            return false;
        }

        findBest<<<1, 32>>>(gbest_d_, aux_, aux_pos_, position_d_);

        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__))
        {
            return false;
        }

        if (verbose_ && (iter % 100 == 0 || iter == max_iterations_ - 1))
        {
            particle_gbest current_best;
            if (handleCudaError(cudaMemcpy(&current_best, gbest_d_, sizeof(particle_gbest), cudaMemcpyDeviceToHost), __FILE__, __LINE__))
            {
                std::cout << "Iteration " << iter << ": Best fitness = " << current_best.fitness << std::endl;
            }
        }
    }

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&exec_time_, start_event_, stop_event_), __FILE__, __LINE__);

    // Copy final results
    if (!handleCudaError(cudaMemcpy(&gbest_, gbest_d_, sizeof(particle_gbest), cudaMemcpyDeviceToHost), __FILE__, __LINE__))
    {
        std::cerr << "Error copying final results" << std::endl;
        return false;
    }

    dt_opt_ = gbest_.position[DT_IDX];
    total_time_ = dt_opt_ * N_STEPS;
    final_fitness_ = gbest_.fitness;
    exec_time_ = exec_time_ / 1000.0f;

    if (!extractResults())
    {
        std::cerr << "Error extracting results or invalid results" << std::endl;
        return false;
    }

    results_valid_ = true;
    configured_ = true;

    if (verbose_)
    {
        printResults();
    }

    return true;
}

bool PSOOptimizer::extractResults()
{
    double dt_double = static_cast<double>(dt_opt_);

    // Decode initial signs
    real initial_signs[n_controls];
    for (int axis = 0; axis < n_controls; axis++)
    {
        real sign_val = gbest_.position[SIGN_IDX(axis)];
        initial_signs[axis] = (sign_val >= REAL(0.0)) ? REAL(1.0) : REAL(-1.0);
    }

    // Decode and sort switch times
    real switch_times[n_controls][MAX_SWITCHES_PER_AXIS];
    int num_switches[n_controls];

    for (int axis = 0; axis < n_controls; axis++)
    {
        real times[MAX_SWITCHES_PER_AXIS];
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++)
        {
            times[s] = gbest_.position[SWITCH_TIME_IDX(axis, s)];
            times[s] = std::max(REAL(0.0), std::min(REAL(1.0), times[s]));
        }

        std::sort(times, times + MAX_SWITCHES_PER_AXIS);

        num_switches[axis] = 0;
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++)
        {
            if (times[s] > REAL(0.01) && times[s] < REAL(0.99))
            {
                if (num_switches[axis] == 0 ||
                    std::abs(times[s] - switch_times[axis][num_switches[axis] - 1]) > REAL(0.01))
                {
                    switch_times[axis][num_switches[axis]] = times[s];
                    num_switches[axis]++;
                }
            }
        }
    }

    // Simulate trajectory
    real current_state[n_states], next_state[n_states];
    real total_time = dt_opt_ * N_STEPS;

    for (int i = 0; i < n_quat; i++)
    {
        current_state[i] = att_params_.initial_quat[i];
        X(i, 0) = static_cast<double>(current_state[i]);
    }
    for (int i = 0; i < n_vel; i++)
    {
        current_state[n_quat + i] = att_params_.initial_omega[i];
        X(n_quat + i, 0) = static_cast<double>(current_state[n_quat + i]);
    }

    real current_time = REAL(0.0);

    for (int step = 0; step < N_STEPS; step++)
    {
        real step_start_time = current_time / total_time;
        real controls[n_controls];

        for (int axis = 0; axis < n_controls; axis++)
        {
            real control_sign = initial_signs[axis];

            for (int s = 0; s < num_switches[axis]; s++)
            {
                if (step_start_time >= switch_times[axis][s])
                {
                    control_sign *= REAL(-1.0); // Flip sign at each switch
                }
            }

            controls[axis] = control_sign * att_params_.max_torque;
            U(axis, step) = static_cast<double>(controls[axis]);
        }

        rk4(current_state, controls, dt_opt_, next_state, &att_params_);

        for (int i = 0; i < n_states; i++)
        {
            current_state[i] = next_state[i];
            X(i, step + 1) = static_cast<double>(current_state[i]);
        }

        dt(step) = dt_double;
        current_time += dt_opt_;
    }

    // // Validate final state
    // real final_error = REAL(0.0);
    // real diff;
    // for (int i = 0; i < n_quat; i++)
    // {
    //     diff = current_state[i] - att_params_.target_quat[i];
    //     final_error += diff * diff;
    // }
    // for (int i = 0; i < n_vel; i++)
    // {
    //     diff = current_state[n_quat + i] - att_params_.target_omega[i];
    //     final_error += diff * diff;
    // }
    // final_error = sqrt(final_error);

    // if (final_error > REAL(1e-3))
    // {
    //     std::cerr << "Warning: Final state error: " << final_error << std::endl;
    //     return false;
    // }
    return true;
}

bool PSOOptimizer::getStats(double &final_fitness, double &setup_time, double &exec_time) const
{
    if (!results_valid_)
    {
        std::cerr << "Warning: No valid results available. Call optimize() first." << std::endl;
        return false;
    }

    final_fitness = static_cast<double>(final_fitness_);
    setup_time = static_cast<double>(setup_time_);
    exec_time = static_cast<double>(exec_time_);

    return true;
}

void PSOOptimizer::printResults() const
{
    if (!results_valid_)
    {
        std::cout << "No valid results available." << std::endl;
        return;
    }

    std::cout << "\n=== PSO Optimization Results ===" << std::endl;
    std::cout << "Final fitness: " << std::setprecision(6) << final_fitness_ << std::endl;
    std::cout << "Total maneuver time: " << total_time_ << " seconds" << std::endl;
    std::cout << "Time step: " << dt_opt_ << " seconds" << std::endl;
    std::cout << "Execution time: " << exec_time_ << " seconds" << std::endl;
    std::cout << "Setup time: " << setup_time_ << " seconds" << std::endl;
    std::cout << "Total computation time: " << (setup_time_ + exec_time_) << " seconds" << std::endl;
    std::cout << "===============================\n"
              << std::endl;
}

void PSOOptimizer::reset()
{
    cleanup();
    results_valid_ = false;
    configured_ = false;
}

void PSOOptimizer::cleanup()
{
    // Free LHS samples
    if (lhs_samples_)
    {
        for (int i = 0; i < num_particles_; i++)
        {
            if (lhs_samples_[i])
                delete[] lhs_samples_[i];
        }
        delete[] lhs_samples_;
        lhs_samples_ = nullptr;
    }

    // Free host memory
    if (particles_)
    {
        if (particles_->position)
            free(particles_->position);
        if (particles_->velocity)
            free(particles_->velocity);
        if (particles_->pbest_pos)
            free(particles_->pbest_pos);
        if (particles_->fitness)
            free(particles_->fitness);
        if (particles_->pbest_fit)
            free(particles_->pbest_fit);
        free(particles_);
        particles_ = nullptr;
    }

    // Free device memory
    if (position_d_)
    {
        cudaFree(position_d_);
        position_d_ = nullptr;
    }
    if (velocity_d_)
    {
        cudaFree(velocity_d_);
        velocity_d_ = nullptr;
    }
    if (fitness_d_)
    {
        cudaFree(fitness_d_);
        fitness_d_ = nullptr;
    }
    if (pbest_pos_d_)
    {
        cudaFree(pbest_pos_d_);
        pbest_pos_d_ = nullptr;
    }
    if (pbest_fit_d_)
    {
        cudaFree(pbest_fit_d_);
        pbest_fit_d_ = nullptr;
    }
    if (gbest_d_)
    {
        cudaFree(gbest_d_);
        gbest_d_ = nullptr;
    }
    if (aux_)
    {
        cudaFree(aux_);
        aux_ = nullptr;
    }
    if (aux_pos_)
    {
        cudaFree(aux_pos_);
        aux_pos_ = nullptr;
    }

    if (start_event_)
    {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
    }
    if (stop_event_)
    {
        cudaEventDestroy(stop_event_);
        stop_event_ = nullptr;
    }
}

bool PSOOptimizer::handleCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " in " << file << " at line " << line << std::endl;
        return false;
    }
    return true;
}