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
__constant__ my_real w_d, c1_d, c2_d;
__constant__ my_real min_w_d, min_c1_d, min_c2_d;
__constant__ bool decay_w_d, decay_c1_d, decay_c2_d;
__constant__ my_real alpha_d, saturation_d;

/** @brief Physical constraint bounds in device constant memory */
__constant__ my_real max_torque_d, min_torque_d;
__constant__ my_real max_dt_d, min_dt_d;

/** @brief PSO velocity limits in device constant memory */
__constant__ my_real max_v_torque_d, max_v_dt_d;

/** @brief Problem dimensions in device constant memory */
__constant__ int particle_cnt_d, dimensions_d;

/** @brief Complete attitude parameters structure in device constant memory */
__constant__ attitude_params att_params_d;

/*==============================================================================
 * MATHEMATICAL UTILITY FUNCTIONS (CUDA KERNELS)
 *============================================================================*/

__host__ __device__ void skew_matrix_4(my_real *w, my_real *S) {
    S[0] = REAL(0.0);     S[1] = -w[0]; S[2] = -w[1]; S[3] = -w[2];
    S[4] = w[0];  S[5] = REAL(0.0);     S[6] = w[2];  S[7] = -w[1];
    S[8] = w[1];  S[9] = -w[2]; S[10] = REAL(0.0);    S[11] = w[0];
    S[12] = w[2]; S[13] = w[1]; S[14] = -w[0]; S[15] = REAL(0.0);
}

__host__ __device__ void cross_product(my_real *a, my_real *b, my_real *result) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2]; 
    result[2] = a[0]*b[1] - a[1]*b[0];
}

__host__ __device__ my_real quaternion_norm(my_real *q) {
    return SQRT(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}

/*==============================================================================
 * DYNAMICS AND NUMERICAL INTEGRATION SCHEMES
 *============================================================================*/

__host__ __device__ void attitude_dynamics(my_real *X, my_real *U, my_real *X_dot, attitude_params *params) {
    my_real *q = X;
    my_real *w = &X[n_quat];
    
    // Quaternion kinematics: q̇ = 0.5 * S(ω) * q
    my_real S[16];
    skew_matrix_4(w, S);
    for(int i = 0; i < n_quat; i++) {
        X_dot[i] = REAL(0.5) * (S[i*4]*q[0] + S[i*4+1]*q[1] + S[i*4+2]*q[2] + S[i*4+3]*q[3]);
    }
    
    // Angular dynamics: ω̇ = I⁻¹ * (τ - ω × (I*ω))
    my_real Iw[n_vel] = {
        params->inertia[0] * w[0], 
        params->inertia[1] * w[1], 
        params->inertia[2] * w[2]
    };
    
    my_real w_cross_Iw[n_vel];
    cross_product(w, Iw, w_cross_Iw);

    for(int i = 0; i < n_vel; i++) {
        X_dot[n_quat+i] = (U[i] - w_cross_Iw[i]) / params->inertia[i];
    }
}

__host__ __device__ void rk4(my_real *X, my_real *U, my_real dt, my_real *X_next, attitude_params *params) {
    my_real k1[n_states], k2[n_states], k3[n_states], k4[n_states];
    my_real X_temp[n_states];
    
    // k1 = f(X, U)
    attitude_dynamics(X, U, k1, params);
    
    // k2 = f(X + dt/2*k1, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/REAL(2.0)*k1[i];
    attitude_dynamics(X_temp, U, k2, params);

    // k3 = f(X + dt/2*k2, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/REAL(2.0)*k2[i];
    attitude_dynamics(X_temp, U, k3, params);

    // k4 = f(X + dt*k3, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt*k3[i];
    attitude_dynamics(X_temp, U, k4, params);

    // Final integration step
    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt/REAL(6.0)*(k1[i] + REAL(2.0)*k2[i] + REAL(2.0)*k3[i] + k4[i]);
    }
    
    // Maintain quaternion unit constraint
    my_real q_norm = quaternion_norm(X_next);
    if(q_norm > REAL(1e-6)) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

__host__ __device__ void euler(my_real *X, my_real *U, my_real dt, my_real *X_next, attitude_params *params) {
    my_real X_dot[n_states];
    
    attitude_dynamics(X, U, X_dot, params);

    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt * X_dot[i];
    }
    
    my_real q_norm = quaternion_norm(X_next);
    if(q_norm > REAL(1e-6)) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

/*==============================================================================
 * FITNESS FUNCTIONS IMPLEMENTATIONS
 *==============================================================================*/

__host__ __device__ my_real fit_full(my_real *solution_vector, int particle_id, attitude_params *params) {
    my_real dt = solution_vector[PARTICLE_IDX_FULL(particle_id, DT_IDX_FULL)];
    my_real constraints_violation = REAL(0.0);
    my_real X[n_states], X_next[n_states];
    for(int i = 0; i < n_quat; i++) X[i] = params->initial_quat[i];
    for(int i = 0; i < n_vel; i++) X[n_quat+i] = params->initial_omega[i];

    int switches = 0;

    for(int step = 0; step < N_STEPS; step++) {
        my_real U[n_controls];
        for(int axis = 0; axis < n_controls; axis++) {
            int torque_idx = TORQUE_IDX(step, axis);
            U[axis] = solution_vector[PARTICLE_IDX_FULL(particle_id, torque_idx)];
            
            if (step > 0) {
                int previous_idx = TORQUE_IDX(step-1, axis);
                my_real previous_torque = solution_vector[PARTICLE_IDX_FULL(particle_id, previous_idx)];
                if (U[axis] * previous_torque < 0) {
                    switches++;
                }
            }
        }

        INTEGRATE(X, U, dt, X_next, params);

        my_real q_norm = quaternion_norm(X_next);
        constraints_violation -= QUAT_NORM_PENALTY * FABS(q_norm - REAL(1.0));

        for(int i = 0; i < n_states; i++) X[i] = X_next[i];
    }

    constraints_violation -= SWITCH_PENALTY * switches;

    my_real final_error = REAL(0.0);
    my_real diff;
    for(int i = 0; i < n_quat; i++) {
        diff = X[i] - params->target_quat[i];
        final_error += diff * diff;
    }
    for(int i = 0; i < n_vel; i++) {
        diff = X[n_quat+i] - params->target_omega[i];
        final_error += diff * diff;
    }
    final_error = SQRT(final_error);
    constraints_violation -= FINAL_STATE_PENALTY * final_error;
    my_real total_time = dt * N_STEPS;

    constraints_violation -= DT_PENALTY * total_time;
    return constraints_violation;

}

__host__ __device__ my_real fit_sto(my_real *solution_vector, int particle_id, attitude_params *params)
{
    my_real dt = solution_vector[PARTICLE_IDX_STO(particle_id, DT_IDX_STO)];
    my_real total_time = dt * N_STEPS;

    // Decode initial control signs (map [-1,1] to {-1,+1})
    my_real initial_signs[n_controls];
    for (int axis = 0; axis < n_controls; axis++)
    {
        initial_signs[axis] = solution_vector[PARTICLE_IDX_STO(particle_id, axis)];
    }

    // Decode and sort switch times for each axis (normalized to [0,1])
    my_real switch_times[n_controls][MAX_SWITCHES_PER_AXIS];
    int num_switches[n_controls];

    for (int axis = 0; axis < n_controls; axis++)
    {
        // Extract switch times for this axis
        my_real times[MAX_SWITCHES_PER_AXIS];
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++)
        {
            times[s] = solution_vector[PARTICLE_IDX_STO(particle_id, SWITCH_TIME_IDX(axis, s))];
            // Clamp to [0,1]
            times[s] = FMAX(REAL(0.0), FMIN(REAL(1.0), times[s]));
        }

        // Bubble sort (simple, efficient for small arrays)
        for (int i = 0; i < MAX_SWITCHES_PER_AXIS - 1; i++)
        {
            for (int j = 0; j < MAX_SWITCHES_PER_AXIS - i - 1; j++)
            {
                if (times[j] > times[j + 1])
                {
                    my_real temp = times[j];
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
                    FABS(times[s] - switch_times[axis][num_switches[axis] - 1]) > REAL(0.01))
                {
                    switch_times[axis][num_switches[axis]] = times[s];
                    num_switches[axis]++;
                }
            }
        }
    }

    // Simulate trajectory with bang-bang control
    my_real constraints_violation = REAL(0.0);
    my_real X[n_states], X_next[n_states];

    for (int i = 0; i < n_quat; i++)
        X[i] = params->initial_quat[i];
    for (int i = 0; i < n_vel; i++)
        X[n_quat + i] = params->initial_omega[i];

    my_real current_time = REAL(0.0);

    for (int step = 0; step < N_STEPS; step++)
    {
        my_real step_start_time = current_time / total_time; // Normalized time [0,1]
        my_real U[n_controls];

        // Determine control for each axis based on switch times
        for (int axis = 0; axis < n_controls; axis++)
        {
            my_real control_sign = initial_signs[axis];

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

        INTEGRATE(X, U, dt, X_next, params);

        my_real q_norm = quaternion_norm(X_next);
        constraints_violation -= QUAT_NORM_PENALTY * FABS(q_norm - REAL(1.0));

        for (int i = 0; i < n_states; i++)
            X[i] = X_next[i];
        current_time += dt;
    }

    // Final state error
    my_real final_error = REAL(0.0);
    my_real diff;
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
    final_error = SQRT(final_error);

    constraints_violation -= FINAL_STATE_PENALTY * final_error;
    constraints_violation -= DT_PENALTY * total_time;

    return constraints_violation;
}

/*==============================================================================
 * CUDA KERNEL IMPLEMENTATIONS
 *============================================================================*/

__global__ void move(my_real *position_d, my_real *velocity_d, my_real *fitness_d,
                     my_real *pbest_pos_d, my_real *pbest_fit_d,
                     particle_gbest *gbest_d, my_real *aux, my_real *aux_pos) {

    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    extern __shared__ my_real sharedMemory[];
    my_real *privateBestQueue = (my_real *)sharedMemory;                    
    int *privateBestParticleQueue = (int *)&sharedMemory[blockDim.x];   
    __shared__ unsigned int queue_num;
    
    if (particle_id >= particle_cnt_d) return;

    if (tidx == 0) queue_num = 0;
    __syncthreads();

    curandState state1, state2;
    curand_init((unsigned long long)clock() + particle_id * 2, 0, 0, &state1);
    curand_init((unsigned long long)clock() + particle_id * 2 + 1, 0, 0, &state2);

    my_real w = w_d, c1 = c1_d, c2 = c2_d;
    if (decay_w_d) {
        w = w_d - (w_d - min_w_d) * (my_real)particle_id / particle_cnt_d;
    }
    if (decay_c1_d) {
        c1 = c1_d - (c1_d - min_c1_d) * (my_real)particle_id / particle_cnt_d;
    }
    if (decay_c2_d) {
        c2 = c2_d - (c2_d - min_c2_d) * (my_real)particle_id / particle_cnt_d;
    }

    // Determine index macro based on method (use dimensions_d for runtime selection)
    for (int dim = 0; dim < dimensions_d; dim++) {
        int pos_idx = particle_id * dimensions_d + dim;  // Direct calculation instead of macro
        
        my_real pos = position_d[pos_idx];
        my_real vel = velocity_d[pos_idx];
        my_real pbest_pos = pbest_pos_d[pos_idx];
        my_real gbest_pos = gbest_d->position[dim];  // Access via pointer
        
        vel = w * vel +
              c1 * CURAND(state1) * (pbest_pos - pos) +
              c2 * CURAND(state2) * (gbest_pos - pos);
        
        // Handle dimension-specific constraints
        if (dimensions_d == DIMENSIONS_FULL) {
            // FULL method
            if (dim < TORQUE_DIMS) {
                vel = FMAX(-max_v_torque_d, FMIN(max_v_torque_d, vel));
                pos = pos + vel;
                pos = FMAX(min_torque_d, FMIN(max_torque_d, pos));
            } else {  // dt dimension
                vel = FMAX(-max_v_dt_d, FMIN(max_v_dt_d, vel));
                pos = pos + vel;
                pos = FMAX(min_dt_d, FMIN(max_dt_d, pos));
            }
        } else {
            // STO method
            if (dim < N_SIGNS) {  // Sign dimensions
                my_real vel_clamp = -LOG(REAL(1.0)/saturation_d - REAL(1.0))/alpha_d;
                vel = FMAX(-vel_clamp, FMIN(vel_clamp, vel));
                my_real sigmoid = REAL(1.0) / (REAL(1.0) + EXP(-alpha_d * vel));
                pos = (CURAND(state1) < sigmoid) ? REAL(1.0) : REAL(-1.0);
            } else if (dim < N_SIGNS + N_SWITCH_TIMES) {  // Switch time dimensions
                vel = FMAX(REAL(-1.0), FMIN(REAL(1.0), vel));
                pos = pos + vel;
                pos = FMAX(REAL(0.0), FMIN(REAL(1.0), pos));
            } else {  // dt dimension
                vel = FMAX(-max_v_dt_d, FMIN(max_v_dt_d, vel));
                pos = pos + vel;
                pos = FMAX(min_dt_d, FMIN(max_dt_d, pos));
            }
        }
        
        position_d[pos_idx] = pos;
        velocity_d[pos_idx] = vel;
    }
    
    // Evaluate fitness based on method
    my_real new_fitness;
    if (dimensions_d == DIMENSIONS_FULL) {
        new_fitness = fit_full(position_d, particle_id, &att_params_d);
    } else {
        new_fitness = fit_sto(position_d, particle_id, &att_params_d);
    }
    fitness_d[particle_id] = new_fitness;

    if (new_fitness > pbest_fit_d[particle_id]) {
        pbest_fit_d[particle_id] = new_fitness;
        for (int dim = 0; dim < dimensions_d; dim++) {
            int pos_idx = particle_id * dimensions_d + dim;
            pbest_pos_d[pos_idx] = position_d[pos_idx];
        }
    }
    
    __syncthreads();

    if (new_fitness > gbest_d->fitness) {
        unsigned int my_index = atomicAdd(&queue_num, 1);
        if (my_index < blockDim.x) {
            privateBestQueue[my_index] = new_fitness;
            privateBestParticleQueue[my_index] = particle_id;
        }
    }
    
    __syncthreads();

    if (tidx == 0) {
        aux[blockIdx.x] = REAL_MIN;
        aux_pos[blockIdx.x] = -1;
        
        if (queue_num > 0) {
            my_real best_fitness = privateBestQueue[0];
            int best_idx = 0;
            
            for (unsigned int i = 1; i < queue_num && i < blockDim.x; i++) {
                if (privateBestQueue[i] > best_fitness) {
                    best_fitness = privateBestQueue[i];
                    best_idx = i;
                }
            }
            aux[blockIdx.x] = best_fitness;
            aux_pos[blockIdx.x] = privateBestParticleQueue[best_idx];
        }
    }
}

__global__ void findBest(particle_gbest *gbest, my_real *aux, my_real *aux_pos, my_real *position_d) {
    int tid = threadIdx.x;
    
    my_real my_fitness = (tid < BlocksPerGrid) ? aux[tid] : REAL_MIN;
    int my_particle = (tid < BlocksPerGrid) ? (int)aux_pos[tid] : -1;
    
    for (int offset = 16; offset > 0; offset /= 2) {
        my_real other_fitness = __shfl_down_sync(0xffffffff, my_fitness, offset);
        int other_particle = __shfl_down_sync(0xffffffff, my_particle, offset);
        if (other_fitness > my_fitness) {
            my_fitness = other_fitness;
            my_particle = other_particle;
        }
    }
    
    if (tid == 0 && my_fitness > gbest->fitness) {
        gbest->fitness = my_fitness;
        if (my_particle >= 0) {
            for (int dim = 0; dim < dimensions_d; dim++) {
                int pos_idx = my_particle * dimensions_d + dim;
                gbest->position[dim] = position_d[pos_idx];  // Access via pointer
            }
        }
        __threadfence();
    }
}

/*==============================================================================
 * PSO OPTIMIZER CLASS IMPLEMENTATION
 *============================================================================*/
PSOOptimizer::PSOOptimizer(casadi::DM& state_matrix, casadi::DM& input_matrix,
                           casadi::DM& dt_matrix, PSOMethod method, bool verbose,
                           int num_particles) 
    : configured_(false)
    , results_valid_(false)
    , max_iterations_(ITERATIONS)
    , num_particles_(num_particles)
    , inertia_weight_(W)
    , cognitive_weight_(C_1)
    , social_weight_(C_2)
    , min_w_(MIN_W)
    , min_c1_(MIN_C1)
    , min_c2_(MIN_C2)
    , decay_w_(DEC_INERTIA)
    , decay_c1_(DEC_C1)
    , decay_c2_(DEC_C2)
    , particles_(nullptr)
    , position_d_(nullptr)
    , velocity_d_(nullptr)
    , fitness_d_(nullptr)
    , pbest_pos_d_(nullptr)
    , pbest_fit_d_(nullptr)
    , gbest_d_(nullptr)
    , gbest_pos_d_(nullptr)
    , aux_(nullptr)
    , aux_pos_(nullptr)
    , lhs_samples_(nullptr)
    , lhs_generated_(false)
    , verbose_(verbose)
    , X(state_matrix)
    , U(input_matrix)
    , dt(dt_matrix)
    , method_(method)
{
    // Initialize CUDA events
    if (!handleCudaError(cudaEventCreate(&start_event_), __FILE__, __LINE__) ||
        !handleCudaError(cudaEventCreate(&stop_event_), __FILE__, __LINE__)) {
        std::cerr << "Failed to create CUDA events" << std::endl;
    }

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    // Validate output matrix dimensions
    if ((X.size1() != n_states || X.size2() != N_STEPS + 1) ||
       (U.size1() != n_controls || U.size2() != N_STEPS) ||
       (dt.size1() != N_STEPS || dt.size2() != 1)){
        std::cerr << "Error: Output matrices have incorrect dimensions." << std::endl;
        std::cerr << "Expected dimensions | Given dimensions: " 
                  << "X: (" << n_states << ", " << N_STEPS + 1 << ") | (" << X.size1() << ", " << X.size2() << "), "
                  << "U: (" << n_controls << ", " << N_STEPS << ") | (" << U.size1() << ", " << U.size2() << "), "
                  << "dt: (" << N_STEPS << ", 1) | (" << dt.size1() << ", " << dt.size2() << ")" << std::endl;
        cleanup();
        return;
    }

    // Determine optimization dimensions based on method
    if (method_ == PSOMethod::FULL) {
        dimensions_ = DIMENSIONS_FULL; // Torques + dt
    } else if (method_ == PSOMethod::STO) {
        dimensions_ = DIMENSIONS_STO; // Initial signs + switch times + dt
    } else {
        std::cerr << "Error: Unknown PSO method." << std::endl;
        cleanup();
        return;
    }

    // Initialize attitude parameters with default spacecraft values
    memset(&att_params_, 0, sizeof(attitude_params));
    
    att_params_.inertia[0] = static_cast<my_real>(i_x);
    att_params_.inertia[1] = static_cast<my_real>(i_y);
    att_params_.inertia[2] = static_cast<my_real>(i_z);
    att_params_.max_torque = static_cast<my_real>(tau_max);
    att_params_.min_torque = -static_cast<my_real>(tau_max);
    att_params_.max_dt = static_cast<my_real>(dt_max);
    att_params_.min_dt = static_cast<my_real>(dt_min);
    
    // Initialize velocity limits
    max_v_torque_ = att_params_.max_torque - att_params_.min_torque;
    max_v_dt_ = att_params_.max_dt - att_params_.min_dt;

    // Allocate LHS samples storage
    lhs_samples_ = new my_real*[num_particles_];
    for (int i = 0; i < num_particles_; i++) {
        lhs_samples_[i] = new my_real[dimensions_];
    }

    // Allocate host particle structure
    particles_ = (particle*)malloc(sizeof(particle));
    if (!particles_) {
        std::cerr << "Failed to allocate particle structure" << std::endl;
        cleanup();
        return;
    }
    
    particles_->position = (my_real*)malloc(sizeof(my_real) * num_particles_ * dimensions_);
    particles_->velocity = (my_real*)malloc(sizeof(my_real) * num_particles_ * dimensions_);
    particles_->pbest_pos = (my_real*)malloc(sizeof(my_real) * num_particles_ * dimensions_);
    particles_->fitness = (my_real*)malloc(sizeof(my_real) * num_particles_);
    particles_->pbest_fit = (my_real*)malloc(sizeof(my_real) * num_particles_);
    
    if (!particles_->position || !particles_->velocity || !particles_->pbest_pos || 
        !particles_->fitness || !particles_->pbest_fit) {
        std::cerr << "Failed to allocate particle arrays" << std::endl;
        cleanup();
        return;
    }

    // Allocate gbest structure - NOW WITH POINTER
    gbest_ = (particle_gbest*)malloc(sizeof(particle_gbest));
    if (!gbest_) {
        std::cerr << "Failed to allocate global best particle" << std::endl;
        cleanup();
        return;
    }
    // Allocate the position array
    gbest_->position = (my_real*)malloc(sizeof(my_real) * dimensions_);
    if (!gbest_->position) {
        std::cerr << "Failed to allocate global best position array" << std::endl;
        cleanup();
        return;
    }
    gbest_->fitness = -REAL_MAX;

    // One-time GPU initialization
    if (!allocateDeviceMemory() || !copyImmutableConstants()) {
        std::cerr << "GPU initialization failed" << std::endl;
        cleanup();
        return;
    }

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&setup_time_, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ /= 1000.0f;
}

PSOOptimizer::~PSOOptimizer() {
    cleanup();
}

bool PSOOptimizer::allocateDeviceMemory() {
    size_t particle_data_size = sizeof(my_real) * num_particles_ * dimensions_;
    
    if (!handleCudaError(cudaMalloc((void**)&position_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&velocity_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&pbest_pos_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&fitness_d_, sizeof(my_real) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&pbest_fit_d_, sizeof(my_real) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&aux_, sizeof(my_real) * BlocksPerGrid), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&aux_pos_, sizeof(my_real) * BlocksPerGrid), __FILE__, __LINE__)) {
        return false;
    }
    
    // Allocate device memory for gbest structure
    if (!handleCudaError(cudaMalloc((void**)&gbest_d_, sizeof(particle_gbest)), __FILE__, __LINE__)) {
        return false;
    }
    
    // Allocate device memory for gbest position array
    if (!handleCudaError(cudaMalloc((void**)&gbest_pos_d_, sizeof(my_real) * dimensions_), __FILE__, __LINE__)) {
        return false;
    }
    
    return true;
}

void PSOOptimizer::setPSOParameters(int max_iterations, double inertia_weight, 
    double cognitive_weight, double social_weight, bool decay_inertia, 
    bool decay_cognitive, bool decay_social, double min_inertia, double min_cognitive, 
    double min_social,  double sigmoid_alpha, double sigmoid_saturation) {

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    if (max_iterations > 0) max_iterations_ = max_iterations;
    if (inertia_weight >= 0) inertia_weight_ = static_cast<my_real>(inertia_weight);
    if (cognitive_weight >= 0) cognitive_weight_ = static_cast<my_real>(cognitive_weight);
    if (social_weight >= 0) social_weight_ = static_cast<my_real>(social_weight);
    decay_w_ = decay_inertia;   
    decay_c1_ = decay_cognitive;
    decay_c2_ = decay_social;
    if (min_inertia >= 0 && min_inertia <= inertia_weight) min_w_ = static_cast<my_real>(min_inertia);
    if (min_cognitive >= 0 && min_cognitive <= cognitive_weight) min_c1_ = static_cast<my_real>(min_cognitive);
    if (min_social >= 0 && min_social <= social_weight) min_c2_ = static_cast<my_real>(min_social);
    if (sigmoid_alpha > 0) sigmoid_alpha_ = static_cast<my_real>(sigmoid_alpha);
    if (sigmoid_saturation >= 0.5 && sigmoid_saturation <= 1.0) sigmoid_saturation_ = static_cast<my_real>(sigmoid_saturation);

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

bool PSOOptimizer::copyImmutableConstants() {
    // Copy PSO parameters
    if (!handleCudaError(cudaMemcpyToSymbol(w_d, &inertia_weight_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c1_d, &cognitive_weight_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c2_d, &social_weight_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(decay_w_d, &decay_w_, sizeof(bool)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(decay_c1_d, &decay_c1_, sizeof(bool)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(decay_c2_d, &decay_c2_, sizeof(bool)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_w_d, &min_w_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_c1_d, &min_c1_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_c2_d, &min_c2_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(alpha_d, &sigmoid_alpha_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(saturation_d, &sigmoid_saturation_, sizeof(my_real)), __FILE__, __LINE__)) {
        return false;
    }
    
    // Copy physical constraints (immutable)
    if (!handleCudaError(cudaMemcpyToSymbol(max_torque_d, &att_params_.max_torque, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_torque_d, &att_params_.min_torque, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_dt_d, &att_params_.max_dt, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_dt_d, &att_params_.min_dt, sizeof(my_real)), __FILE__, __LINE__)) {
        return false;
    }
    
    // Copy velocity limits
    if (!handleCudaError(cudaMemcpyToSymbol(max_v_torque_d, &max_v_torque_, sizeof(my_real)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_v_dt_d, &max_v_dt_, sizeof(my_real)), __FILE__, __LINE__)) {
        return false;
    }
    
    // Copy dimensions
    if (!handleCudaError(cudaMemcpyToSymbol(particle_cnt_d, &num_particles_, sizeof(int)), __FILE__, __LINE__)) {
        return false;
    }
    
    if (!handleCudaError(cudaMemcpyToSymbol(dimensions_d, &dimensions_, sizeof(int)), __FILE__, __LINE__)) {
        return false;
    }
    
    return true;
}

bool PSOOptimizer::copyMutableStateParameters() {
    // Only copy the att_params structure (contains initial/target states)
    return handleCudaError(cudaMemcpyToSymbol(att_params_d, &att_params_, sizeof(attitude_params)), __FILE__, __LINE__);
}

void PSOOptimizer::setStates(const double* initial_state, const double* target_state) {
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);
    // Update initial and target states
    for (int i = 0; i < n_quat; i++) {
        att_params_.initial_quat[i] = static_cast<my_real>(initial_state[i]);
        att_params_.target_quat[i] = static_cast<my_real>(target_state[i]);
    }
    for (int i = 0; i < n_vel; i++) {
        att_params_.initial_omega[i] = static_cast<my_real>(initial_state[i + n_quat]);
        att_params_.target_omega[i] = static_cast<my_real>(target_state[i + n_quat]);
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

bool PSOOptimizer::initializeParticles_sto(bool regenerate_lhs) {
    srand((unsigned)time(NULL));
    
    // Generate or reuse LHS samples
    if (regenerate_lhs || !lhs_generated_) {
        generateLHSSamples(lhs_samples_);
        lhs_generated_ = true;
    }
    
    // Initialize global best
    gbest_->fitness = -REAL_MAX;
    
    int best_particle = 0;
    
    for (int i = 0; i < num_particles_; i++) {
        // Initialize sign variables
        for (int dim = 0; dim < N_SIGNS; dim++) {
            int idx = i * dimensions_ + dim;
            particles_->position[idx] = (lhs_samples_[i][dim] > REAL(0.5)) ? REAL(1.0) : REAL(-1.0);
            particles_->velocity[idx] = (lhs_samples_[i][dim] - REAL(0.5)) * REAL(4.0);
            particles_->pbest_pos[idx] = particles_->position[idx];
        }

        // Initialize switch time variables
        for (int dim = N_SIGNS; dim < N_SIGNS + N_SWITCH_TIMES; dim++) {
            int idx = i * dimensions_ + dim;
            particles_->position[idx] = lhs_samples_[i][dim];
            particles_->velocity[idx] = (lhs_samples_[i][dim] - REAL(0.5)) * REAL(2.0);
            particles_->pbest_pos[idx] = particles_->position[idx];
        }

        // Initialize dt
        int dt_idx = i * dimensions_ + (dimensions_ - 1);
        my_real dt_range = att_params_.max_dt - att_params_.min_dt;
        particles_->position[dt_idx] = lhs_samples_[i][dimensions_ - 1] * dt_range + att_params_.min_dt;
        particles_->velocity[dt_idx] = (lhs_samples_[i][dimensions_ - 1] - REAL(0.5)) * REAL(2.0) * max_v_dt_;
        particles_->pbest_pos[dt_idx] = particles_->position[dt_idx];

        // Evaluate fitness
        particles_->fitness[i] = fit_sto(particles_->position, i, &att_params_);
        particles_->pbest_fit[i] = particles_->fitness[i];

        if (i == 0 || particles_->pbest_fit[i] > gbest_->fitness) {
            best_particle = i;
            gbest_->fitness = particles_->fitness[i];
        }
    }
    
    // Copy global best position
    for (int dim = 0; dim < dimensions_; dim++) {
        int idx = best_particle * dimensions_ + dim;
        gbest_->position[dim] = particles_->position[idx];
    }
    
    return true;
}

bool PSOOptimizer::initializeParticles_full(bool regenerate_lhs) {
    srand((unsigned)time(NULL));
    
    // Generate or reuse LHS samples
    if (regenerate_lhs || !lhs_generated_) {
        generateLHSSamples(lhs_samples_);
        lhs_generated_ = true;
    }
    
    // Initialize global best
    gbest_->fitness = -REAL_MAX;
    
    int best_particle = 0;
    
    for (int i = 0; i < num_particles_; i++) {
        // Initialize torque variables
        for (int dim = 0; dim < TORQUE_DIMS; dim++) {
            int idx = i * dimensions_ + dim;
            my_real torque_range = att_params_.max_torque - att_params_.min_torque;
            
            particles_->position[idx] = lhs_samples_[i][dim] * torque_range + att_params_.min_torque;
            particles_->velocity[idx] = (lhs_samples_[i][dim] - REAL(0.5)) * REAL(2.0) * max_v_torque_;
            particles_->pbest_pos[idx] = particles_->position[idx];
        }
        
        // Initialize dt
        int dt_idx = i * dimensions_ + (dimensions_ - 1);
        my_real dt_range = att_params_.max_dt - att_params_.min_dt;
        
        particles_->position[dt_idx] = lhs_samples_[i][dimensions_ - 1] * dt_range + att_params_.min_dt;
        particles_->velocity[dt_idx] = (lhs_samples_[i][dimensions_ - 1] - REAL(0.5)) * REAL(2.0) * max_v_dt_;
        particles_->pbest_pos[dt_idx] = particles_->position[dt_idx];
        
        // Evaluate fitness
        particles_->fitness[i] = fit_full(particles_->position, i, &att_params_);
        particles_->pbest_fit[i] = particles_->fitness[i];
        
        if (i == 0 || particles_->pbest_fit[i] > gbest_->fitness) {
            best_particle = i;
            gbest_->fitness = particles_->fitness[i];
        }
    }
    
    
    // Copy global best position
    for (int dim = 0; dim < dimensions_; dim++) {
        int idx = best_particle * dimensions_ + dim;
        gbest_->position[dim] = particles_->position[idx];
    }
    
    return true;
}

void PSOOptimizer::generateLHSSamples(my_real** samples) {
    // Generate LHS samples in [0,1] for each dimension
    for (int dim = 0; dim < dimensions_; dim++) {
        std::vector<my_real> intervals(num_particles_);
        
        // Create stratified intervals
        for (int i = 0; i < num_particles_; i++) {
            my_real interval_start = static_cast<my_real>(i) / num_particles_;
            my_real interval_width = REAL(1.0) / num_particles_;
            intervals[i] = interval_start + RND() * interval_width;
        }
        
        // Shuffle to break correlations
        for (int i = num_particles_ - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(intervals[i], intervals[j]);
        }
        
        // Assign to samples matrix
        for (int i = 0; i < num_particles_; i++) {
            samples[i][dim] = intervals[i];
        }
    }
}

bool PSOOptimizer::optimize(bool regenerate_lhs) {

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);
    
    // Initialize particles
    if (method_ == PSOMethod::FULL) {
        if (!initializeParticles_full(regenerate_lhs)) {
            std::cerr << "Particle initialization failed" << std::endl;
            return false;
        }
    } else  if (method_ == PSOMethod::STO) {
        if (!initializeParticles_sto(regenerate_lhs)) {
            std::cerr << "Particle initialization failed" << std::endl;
            return false;
        }
    }
    
    // Copy gbest position pointer to device
    // First copy the position array, then update the struct pointer
    if (!handleCudaError(cudaMemcpy(gbest_pos_d_, gbest_->position, sizeof(my_real) * dimensions_, cudaMemcpyHostToDevice), __FILE__, __LINE__)) {
        return false;
    }
    
    // Create temporary host struct with device pointer
    particle_gbest temp_gbest;
    temp_gbest.position = gbest_pos_d_;  // Point to device memory
    temp_gbest.fitness = gbest_->fitness;
    
    // Copy struct to device
    if (!handleCudaError(cudaMemcpy(gbest_d_, &temp_gbest, sizeof(particle_gbest), cudaMemcpyHostToDevice), __FILE__, __LINE__)) {
        return false;
    }
    
    // Copy initial data to device
    size_t particle_data_size = sizeof(my_real) * num_particles_ * dimensions_;
    if (!handleCudaError(cudaMemcpy(position_d_, particles_->position, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(velocity_d_, particles_->velocity, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_pos_d_, particles_->pbest_pos, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(fitness_d_, particles_->fitness, sizeof(my_real) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_fit_d_, particles_->pbest_fit, sizeof(my_real) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__)) {
        return false;
    }
    
    if (verbose_) {
        std::cout << "Starting PSO optimization..." << std::endl;
        std::cout << "LHS: " << (regenerate_lhs ? "Regenerated" : "Reused") << std::endl;
        std::cout << "Initial best fitness: " << gbest_->fitness << std::endl;
    }
    
    // Main optimization loop
    int shared_mem_size = sizeof(my_real) * ThreadsPerBlock + sizeof(int) * ThreadsPerBlock;

    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time / 1000.0f;  // Convert ms to seconds
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    for (int iter = 0; iter < max_iterations_; iter++) {
        move<<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>(
            position_d_, velocity_d_, fitness_d_, pbest_pos_d_, pbest_fit_d_,
            gbest_d_, aux_, aux_pos_);
        
        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__)) {
            return false;
        }
        
        findBest<<<1, 32>>>(gbest_d_, aux_, aux_pos_, position_d_);
        
        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__)) {
            return false;
        }
        
        if (verbose_ && (iter % 100 == 0 || iter == max_iterations_ - 1)) {
            particle_gbest current_best;
            if (handleCudaError(cudaMemcpy(&current_best, gbest_d_, sizeof(particle_gbest), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) {
                std::cout << "Iteration " << iter << ": Best fitness = " << current_best.fitness << std::endl;
            }
        }
    }
    
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&exec_time_, start_event_, stop_event_), __FILE__, __LINE__);
    
    // Copy final results
    if (!handleCudaError(cudaMemcpy(gbest_->position, gbest_pos_d_, sizeof(my_real) * dimensions_, cudaMemcpyDeviceToHost), __FILE__, __LINE__)) {
        std::cerr << "Error copying final gbest position" << std::endl;
        return false;
    }
    
    // Copy fitness value
    if (!handleCudaError(cudaMemcpy(&gbest_->fitness, &gbest_d_->fitness, sizeof(my_real), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) {
        std::cerr << "Error copying final gbest fitness" << std::endl;
        return false;
    }

    dt_opt_ = gbest_->position[dimensions_ - 1];  // Last dimension is dt
    total_time_ = dt_opt_ * N_STEPS;
    final_fitness_ = gbest_->fitness;
    exec_time_ = exec_time_ / 1000.0f;

    if (method_ == PSOMethod::STO) {
        if (!extractResults_sto()) {
            std::cerr << "Error extracting STO results" << std::endl;
            return false;
        }
    } else if (method_ == PSOMethod::FULL) {
        if (!extractResults_full()) {
            std::cerr << "Error extracting FULL results" << std::endl;
            return false;
        }
    }

    results_valid_ = true;
    configured_ = true;

    if (verbose_) {
        printResults();
    }

    return true;
}

bool PSOOptimizer::extractResults_sto() {
    double dt_double = static_cast<double>(dt_opt_);
    my_real current_state[n_states], next_state[n_states];
    
    // Initialize with initial conditions
    for (int i = 0; i < n_quat; i++) {
        current_state[i] = att_params_.initial_quat[i];
        X(i, 0) = static_cast<double>(current_state[i]);
    }
    for (int i = 0; i < n_vel; i++) {
        current_state[n_quat + i] = att_params_.initial_omega[i];
        X(n_quat + i, 0) = static_cast<double>(current_state[n_quat + i]);
    }
    // Extract STO bang-bang trajectory
    my_real initial_signs[n_controls];
    for (int axis = 0; axis < n_controls; axis++) {
        initial_signs[axis] = gbest_->position[axis];
    }

    my_real switch_times[n_controls][MAX_SWITCHES_PER_AXIS];
    int num_switches[n_controls];

    for (int axis = 0; axis < n_controls; axis++) {
        my_real times[MAX_SWITCHES_PER_AXIS];
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++) {
            times[s] = gbest_->position[N_SIGNS + axis * MAX_SWITCHES_PER_AXIS + s];
            times[s] = std::max(REAL(0.0), std::min(REAL(1.0), times[s]));
        }

        // Sort times
        for (int i = 0; i < MAX_SWITCHES_PER_AXIS - 1; i++) {
            for (int j = 0; j < MAX_SWITCHES_PER_AXIS - i - 1; j++) {
                if (times[j] > times[j + 1]) {
                    my_real temp = times[j];
                    times[j] = times[j + 1];
                    times[j + 1] = temp;
                }
            }
        }

        num_switches[axis] = 0;
        for (int s = 0; s < MAX_SWITCHES_PER_AXIS; s++) {
            if (times[s] > REAL(0.01) && times[s] < REAL(0.99)) {
                if (num_switches[axis] == 0 ||
                    std::abs(times[s] - switch_times[axis][num_switches[axis] - 1]) > REAL(0.01)) {
                    switch_times[axis][num_switches[axis]] = times[s];
                    num_switches[axis]++;
                }
            }
        }
    }

    my_real total_time = dt_opt_ * N_STEPS;
    my_real current_time = REAL(0.0);

    for (int step = 0; step < N_STEPS; step++) {
        my_real step_start_time = current_time / total_time;
        my_real controls[n_controls];

        for (int axis = 0; axis < n_controls; axis++) {
            my_real control_sign = initial_signs[axis];

            for (int s = 0; s < num_switches[axis]; s++) {
                if (step_start_time >= switch_times[axis][s]) {
                    control_sign *= REAL(-1.0);
                }
            }

            controls[axis] = control_sign * att_params_.max_torque;
            U(axis, step) = static_cast<double>(controls[axis]);
        }

        INTEGRATE(current_state, controls, dt_opt_, next_state, &att_params_);

        for (int i = 0; i < n_states; i++) {
            current_state[i] = next_state[i];
            X(i, step + 1) = static_cast<double>(current_state[i]);
        }

        dt(step) = dt_double;
        current_time += dt_opt_;
    }

    // Validate final state
    my_real final_error = REAL(0.0);
    my_real diff;
    for(int i = 0; i < n_quat; i++) {
        diff = current_state[i] - att_params_.target_quat[i];
        final_error += diff * diff;
    }
    for(int i = 0; i < n_vel; i++) {
        diff = current_state[n_quat+i] - att_params_.target_omega[i];
        final_error += diff * diff;
    }
    final_error = SQRT(final_error);
    
    if (final_error > REAL(1e-3) && verbose_) {
        std::cerr << "Warning: Final state deviates significantly from target state. Final error: " 
                  << final_error << std::endl;
    }
    
    return true;
}

bool PSOOptimizer::extractResults_full() {
    double dt_double = static_cast<double>(dt_opt_);
    my_real current_state[n_states], next_state[n_states];
    
    // Initialize with initial conditions
    for (int i = 0; i < n_quat; i++) {
        current_state[i] = att_params_.initial_quat[i];
        X(i, 0) = static_cast<double>(current_state[i]);
    }
    for (int i = 0; i < n_vel; i++) {
        current_state[n_quat + i] = att_params_.initial_omega[i];
        X(n_quat + i, 0) = static_cast<double>(current_state[n_quat + i]);
    }

    // Extract FULL control trajectory
    for (int step = 0; step < N_STEPS; step++) {
        my_real controls[n_controls];
        for (int axis = 0; axis < n_controls; axis++) {
            controls[axis] = gbest_->position[step * n_controls + axis];
            U(axis, step) = static_cast<double>(controls[axis]);
        }

        INTEGRATE(current_state, controls, dt_opt_, next_state, &att_params_);

        for (int i = 0; i < n_states; i++) {
            current_state[i] = next_state[i];
            X(i, step + 1) = static_cast<double>(current_state[i]);
        }

        dt(step) = dt_double;
    }
    
    // Validate final state
    my_real final_error = REAL(0.0);
    my_real diff;
    for(int i = 0; i < n_quat; i++) {
        diff = current_state[i] - att_params_.target_quat[i];
        final_error += diff * diff;
    }
    for(int i = 0; i < n_vel; i++) {
        diff = current_state[n_quat+i] - att_params_.target_omega[i];
        final_error += diff * diff;
    }
    final_error = SQRT(final_error);
    
    if (final_error > REAL(1e-3) && verbose_) {
        std::cerr << "Warning: Final state deviates significantly from target state. Final error: " 
                  << final_error << std::endl;
    }
    
    return true;
}

void PSOOptimizer::cleanup() {
    // Free LHS samples
    if (lhs_samples_) {
        for (int i = 0; i < num_particles_; i++) {
            if (lhs_samples_[i]) delete[] lhs_samples_[i];
        }
        delete[] lhs_samples_;
        lhs_samples_ = nullptr;
    }
    
    // Free host memory
    if (particles_) {
        if (particles_->position) free(particles_->position);
        if (particles_->velocity) free(particles_->velocity);
        if (particles_->pbest_pos) free(particles_->pbest_pos);
        if (particles_->fitness) free(particles_->fitness);
        if (particles_->pbest_fit) free(particles_->pbest_fit);
        free(particles_);
        particles_ = nullptr;
    }
    
    // Free gbest - FREE THE POSITION ARRAY FIRST
    if (gbest_) {
        if (gbest_->position) {
            free(gbest_->position);
            gbest_->position = nullptr;
        }
        free(gbest_);
        gbest_ = nullptr;
    }
    
    // Free device memory
    if (position_d_) { cudaFree(position_d_); position_d_ = nullptr; }
    if (velocity_d_) { cudaFree(velocity_d_); velocity_d_ = nullptr; }
    if (fitness_d_) { cudaFree(fitness_d_); fitness_d_ = nullptr; }
    if (pbest_pos_d_) { cudaFree(pbest_pos_d_); pbest_pos_d_ = nullptr; }
    if (pbest_fit_d_) { cudaFree(pbest_fit_d_); pbest_fit_d_ = nullptr; }
    if (gbest_pos_d_) { cudaFree(gbest_pos_d_); gbest_pos_d_ = nullptr; }  // Free position array
    if (gbest_d_) { cudaFree(gbest_d_); gbest_d_ = nullptr; }  // Free struct
    if (aux_) { cudaFree(aux_); aux_ = nullptr; }
    if (aux_pos_) { cudaFree(aux_pos_); aux_pos_ = nullptr; }

    if (start_event_) { cudaEventDestroy(start_event_); start_event_ = nullptr; }
    if (stop_event_) { cudaEventDestroy(stop_event_); stop_event_ = nullptr; }
}

bool PSOOptimizer::handleCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
        return false;
    }
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