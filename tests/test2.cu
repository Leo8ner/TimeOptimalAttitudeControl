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
#include "test2.h"
using ::erfinv;
/*==============================================================================
 * CUDA CONSTANT MEMORY DECLARATIONS
 *============================================================================*/

/** @brief PSO algorithm parameters in device constant memory */
__constant__ float w_d, c1_d, c2_d;

/** @brief Physical constraint bounds in device constant memory */
__constant__ float max_torque_d, min_torque_d;
__constant__ float max_dt_d, min_dt_d;

/** @brief PSO velocity limits in device constant memory */
__constant__ float max_v_torque_d, max_v_dt_d;

/** @brief Problem dimensions in device constant memory */
__constant__ int particle_cnt_d, dimensions_d;

/** @brief Complete attitude parameters structure in device constant memory */
__constant__ attitude_params att_params_d;

/*==============================================================================
 * MATHEMATICAL UTILITY FUNCTIONS (CUDA KERNELS)
 *============================================================================*/

__host__ __device__ void skew_matrix_4(float *w, float *S) {
    S[0] = 0;     S[1] = -w[0]; S[2] = -w[1]; S[3] = -w[2];
    S[4] = w[0];  S[5] = 0;     S[6] = w[2];  S[7] = -w[1];
    S[8] = w[1];  S[9] = -w[2]; S[10] = 0;    S[11] = w[0];
    S[12] = w[2]; S[13] = w[1]; S[14] = -w[0]; S[15] = 0;
}

__host__ __device__ void cross_product(float *a, float *b, float *result) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2]; 
    result[2] = a[0]*b[1] - a[1]*b[0];
}

__host__ __device__ float quaternion_norm(float *q) {
    return sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}

__host__ __device__ void attitude_dynamics(float *X, float *U, float *X_dot, attitude_params *params) {
    float *q = X;
    float *w = &X[n_quat];
    
    // Quaternion kinematics: q̇ = 0.5 * S(ω) * q
    float S[16];
    skew_matrix_4(w, S);
    for(int i = 0; i < n_quat; i++) {
        X_dot[i] = 0.5f * (S[i*4]*q[0] + S[i*4+1]*q[1] + S[i*4+2]*q[2] + S[i*4+3]*q[3]);
    }
    
    // Angular dynamics: ω̇ = I⁻¹ * (τ - ω × (I*ω))
    float Iw[n_vel] = {
        params->inertia[0] * w[0], 
        params->inertia[1] * w[1], 
        params->inertia[2] * w[2]
    };
    
    float w_cross_Iw[n_vel];
    cross_product(w, Iw, w_cross_Iw);

    for(int i = 0; i < n_vel; i++) {
        X_dot[n_quat+i] = (U[i] - w_cross_Iw[i]) / params->inertia[i];
    }
}

__host__ __device__ void rk4(float *X, float *U, float dt, float *X_next, attitude_params *params) {
    float k1[n_states], k2[n_states], k3[n_states], k4[n_states];
    float X_temp[n_states];
    
    attitude_dynamics(X, U, k1, params);
    
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/2.0f*k1[i];
    attitude_dynamics(X_temp, U, k2, params);

    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/2.0f*k2[i];
    attitude_dynamics(X_temp, U, k3, params);

    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt*k3[i];
    attitude_dynamics(X_temp, U, k4, params);

    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt/6.0f*(k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }
    
    float q_norm = quaternion_norm(X_next);
    if(q_norm > 1e-6f) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

__host__ __device__ void euler(float *X, float *U, float dt, float *X_next, attitude_params *params) {
    float X_dot[n_states];
    
    attitude_dynamics(X, U, X_dot, params);

    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt * X_dot[i];
    }
    
    float q_norm = quaternion_norm(X_next);
    if(q_norm > 1e-6f) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

__host__ __device__ float fit(float *solution_vector, int particle_id, attitude_params *params) {
    float dt = solution_vector[PARTICLE_POS_IDX(particle_id, DT_IDX)];
    
    float X[n_states], X_next[n_states];
    for(int i = 0; i < n_quat; i++) X[i] = params->initial_quat[i];
    for(int i = 0; i < n_vel; i++) X[n_quat+i] = params->initial_omega[i];

    float constraint_violation = 0.0f;
    int switches = 0;

    for(int step = 0; step < N_STEPS; step++) {
        float U[n_controls];
        for(int axis = 0; axis < n_controls; axis++) {
            int torque_idx = TORQUE_IDX(step, axis);
            U[axis] = solution_vector[PARTICLE_POS_IDX(particle_id, torque_idx)];
            
            if (step > 0) {
                int previous_idx = TORQUE_IDX(step-1, axis);
                float previous_torque = solution_vector[PARTICLE_POS_IDX(particle_id, previous_idx)];
                if (U[axis] * previous_torque < 0) {
                    switches++;
                }
            }
        }
        
        euler(X, U, dt, X_next, params);

        float q_norm = quaternion_norm(X_next);
        constraint_violation += QUAT_NORM_PENALTY * fabsf(q_norm - 1.0f);

        for(int i = 0; i < n_states; i++) X[i] = X_next[i];
    }

    constraint_violation += SWITCH_PENALTY * switches;
    
    float final_error = 0.0f;
    for(int i = 0; i < n_quat; i++) {
        final_error += pow(X[i] - params->target_quat[i], 2);
    }
    for(int i = 0; i < n_vel; i++) {
        final_error += pow(X[n_quat+i] - params->target_omega[i], 2);
    }
    final_error = sqrt(final_error);
    constraint_violation += FINAL_STATE_PENALTY * final_error;
    printf("Particle %d final error: %f\n", particle_id, final_error);
    float total_time = dt * N_STEPS;
    
    return -DT_PENALTY * total_time - constraint_violation;
}

/*==============================================================================
 * CUDA KERNEL IMPLEMENTATIONS
 *============================================================================*/

__global__ void move(float *position_d, float *velocity_d, float *fitness_d,
                     float *pbest_pos_d, float *pbest_fit_d, 
                     particle_gbest *gbest_d, float *aux, float *aux_pos) {
    
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    extern __shared__ float sharedMemory[];
    float *privateBestQueue = (float *)sharedMemory;                    
    int *privateBestParticleQueue = (int *)&sharedMemory[blockDim.x];   
    __shared__ unsigned int queue_num;
    
    if (particle_id >= particle_cnt_d) return;

    if (tidx == 0) queue_num = 0;
    __syncthreads();

    curandState state1, state2;
    curand_init((unsigned long long)clock() + particle_id * 2, 0, 0, &state1);
    curand_init((unsigned long long)clock() + particle_id * 2 + 1, 0, 0, &state2);

    float w = w_d;
    if (DEC_INERTIA) {
        w = w_d - (w_d - MIN_W) * particle_id / N_PARTICLES;
    }
    
    for (int dim = 0; dim < dimensions_d; dim++) {
        int pos_idx = PARTICLE_POS_IDX(particle_id, dim);
        
        float pos = position_d[pos_idx];
        float vel = velocity_d[pos_idx];
        float pbest_pos = pbest_pos_d[pos_idx];
        float gbest_pos = gbest_d->position[dim];
        
        vel = w * vel +
              c1_d * curand_uniform(&state1) * (pbest_pos - pos) +
              c2_d * curand_uniform(&state2) * (gbest_pos - pos);
        
        if (dim < TORQUE_DIMS) {
            vel = fmax(-max_v_torque_d, fmin(max_v_torque_d, vel));
            pos = pos + vel;
            pos = fmax(min_torque_d, fmin(max_torque_d, pos));
        } else {
            vel = fmax(-max_v_dt_d, fmin(max_v_dt_d, vel));
            pos = pos + vel;
            pos = fmax(min_dt_d, fmin(max_dt_d, pos));
        }
        
        position_d[pos_idx] = pos;
        velocity_d[pos_idx] = vel;
    }
    
    float new_fitness = fit(position_d, particle_id, &att_params_d);
    fitness_d[particle_id] = new_fitness;
    
    if (new_fitness > pbest_fit_d[particle_id]) {
        pbest_fit_d[particle_id] = new_fitness;
        for (int dim = 0; dim < dimensions_d; dim++) {
            pbest_pos_d[PARTICLE_POS_IDX(particle_id, dim)] = 
                position_d[PARTICLE_POS_IDX(particle_id, dim)];
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
        aux[blockIdx.x] = -FLT_MAX;
        aux_pos[blockIdx.x] = -1;
        
        if (queue_num > 0) {
            float best_fitness = privateBestQueue[0];
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

__global__ void findBest(particle_gbest *gbest, float *aux, float *aux_pos, float *position_d) {
    int tid = threadIdx.x;
    
    float my_fitness = (tid < BlocksPerGrid) ? aux[tid] : -FLT_MAX;
    int my_particle = (tid < BlocksPerGrid) ? (int)aux_pos[tid] : -1;
    
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_fitness = __shfl_down_sync(0xffffffff, my_fitness, offset);
        int other_particle = __shfl_down_sync(0xffffffff, my_particle, offset);
        if (other_fitness > my_fitness) {
            my_fitness = other_fitness;
            my_particle = other_particle;
        }
    }
    
    if (tid == 0 && my_fitness > gbest->fitness) {
        gbest->fitness = my_fitness;
        if (my_particle >= 0) {
            for (int dim = 0; dim < DIMENSIONS; dim++) {
                gbest->position[dim] = position_d[PARTICLE_POS_IDX(my_particle, dim)];
            }
        }
        __threadfence();
    }
}

/*==============================================================================
 * PSO OPTIMIZER CLASS IMPLEMENTATION
 *============================================================================*/

PSOOptimizer::PSOOptimizer(const double* initial_state, const double* target_state, bool verbose) 
    : configured_(false)
    , results_valid_(false)
    , max_iterations_(MAX_ITERA)
    , num_particles_(N_PARTICLES)
    , inertia_weight_(W)
    , cognitive_weight_(C1)
    , social_weight_(C2)
    , particles_(nullptr)
    , position_d_(nullptr)
    , velocity_d_(nullptr)
    , fitness_d_(nullptr)
    , pbest_pos_d_(nullptr)
    , pbest_fit_d_(nullptr)
    , gbest_d_(nullptr)
    , aux_(nullptr)
    , aux_pos_(nullptr)
    , verbose_(verbose)
{
    // Initialize CUDA events
    if (!handleCudaError(cudaEventCreate(&start_event_), __FILE__, __LINE__) ||
        !handleCudaError(cudaEventCreate(&stop_event_), __FILE__, __LINE__)) {
        std::cerr << "Failed to create CUDA events" << std::endl;
    }

    // Start timing
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    // Initialize attitude parameters with default values
    memset(&att_params_, 0, sizeof(attitude_params));
    
    // Set default spacecraft parameters (will be overridden by setSpacecraftParameters)
    att_params_.inertia[0] = static_cast<float>(i_x);
    att_params_.inertia[1] = static_cast<float>(i_y);
    att_params_.inertia[2] = static_cast<float>(i_z);
    att_params_.max_torque = static_cast<float>(tau_max);
    att_params_.min_torque = -static_cast<float>(tau_max);
    att_params_.max_dt = static_cast<float>(dt_max);
    att_params_.min_dt = static_cast<float>(dt_min);
    for (int i = 0; i < n_quat; i++) {
        att_params_.initial_quat[i] = static_cast<float>(initial_state[i]);
        att_params_.target_quat[i] = static_cast<float>(target_state[i]);
        printf("Initial quat[%d]: %f, Target quat[%d]: %f\n", i, att_params_.initial_quat[i], i, att_params_.target_quat[i]);
    }
    for (int i = 0; i < n_vel; i++) {
        att_params_.initial_omega[i] = static_cast<float>(initial_state[i + n_quat]);
        att_params_.target_omega[i] = static_cast<float>(target_state[i + n_quat]);
        printf("Initial omega[%d]: %f, Target omega[%d]: %f\n", i, att_params_.initial_omega[i], i, att_params_.target_omega[i]);
    }
    
    // Initialize velocity limits
    max_v_torque_ = 2.0f * att_params_.max_torque;
    max_v_dt_ = att_params_.max_dt - att_params_.min_dt;

    // Validate configuration
    if (!validateConfiguration()) {
        std::cerr << "Configuration validation failed" << std::endl;
        cleanup();
    }

        // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    
    // Calculate execution time
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&setup_time_, start_event_, stop_event_), __FILE__, __LINE__);
}

PSOOptimizer::~PSOOptimizer() {
    cleanup();
}

void PSOOptimizer::setPSOParameters(int max_iterations, int num_particles,
                                   double inertia_weight, double cognitive_weight, double social_weight) {
    
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    num_particles_ = num_particles;
    inertia_weight_ = static_cast<float>(inertia_weight);
    cognitive_weight_ = static_cast<float>(cognitive_weight);
    social_weight_ = static_cast<float>(social_weight);

    results_valid_ = false;
    // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    
    // Calculate execution time
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time; // Accumulate
}

bool PSOOptimizer::validateConfiguration() const {
    // Check that all required parameters are set
    bool valid = true;
    
    // Check inertia values
    if (att_params_.inertia[0] <= 0 || att_params_.inertia[1] <= 0 || att_params_.inertia[2] <= 0) {
        std::cerr << "Error: Inertia values must be positive" << std::endl;
        valid = false;
    }
    
    // Check torque limits
    if (att_params_.max_torque <= 0) {
        std::cerr << "Error: Maximum torque must be positive" << std::endl;
        valid = false;
    }
    
    // Check time constraints
    if (att_params_.min_dt < 0 || att_params_.max_dt <= att_params_.min_dt) {
        std::cerr << "Error: Invalid time step constraints" << std::endl;
        valid = false;
    }
    
    // Check quaternion normalization
    float q_norm_initial = sqrt(att_params_.initial_quat[0]*att_params_.initial_quat[0] + 
                               att_params_.initial_quat[1]*att_params_.initial_quat[1] + 
                               att_params_.initial_quat[2]*att_params_.initial_quat[2] + 
                               att_params_.initial_quat[3]*att_params_.initial_quat[3]);
    
    float q_norm_target = sqrt(att_params_.target_quat[0]*att_params_.target_quat[0] + 
                              att_params_.target_quat[1]*att_params_.target_quat[1] + 
                              att_params_.target_quat[2]*att_params_.target_quat[2] + 
                              att_params_.target_quat[3]*att_params_.target_quat[3]);
    
    if (fabs(q_norm_initial - 1.0f) > 1e-3f) {
        std::cerr << "Warning: Initial quaternion is not normalized (norm = " << q_norm_initial << ")" << std::endl;
    }
    
    if (fabs(q_norm_target - 1.0f) > 1e-3f) {
        std::cerr << "Warning: Target quaternion is not normalized (norm = " << q_norm_target << ")" << std::endl;
    }
    
    return valid;
}

bool PSOOptimizer::initializeCUDA() {
    // Allocate device memory
    size_t particle_data_size = sizeof(float) * num_particles_ * DIMENSIONS;
    
    if (!handleCudaError(cudaMalloc((void**)&position_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&velocity_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&pbest_pos_d_, particle_data_size), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&fitness_d_, sizeof(float) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&pbest_fit_d_, sizeof(float) * num_particles_), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&gbest_d_, sizeof(particle_gbest)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&aux_, sizeof(float) * BlocksPerGrid), __FILE__, __LINE__) ||
        !handleCudaError(cudaMalloc((void**)&aux_pos_, sizeof(float) * BlocksPerGrid), __FILE__, __LINE__)) {
        return false;
    }
    
    // Copy constants to device constant memory
    if (!handleCudaError(cudaMemcpyToSymbol(w_d, &inertia_weight_, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c1_d, &cognitive_weight_, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(c2_d, &social_weight_, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_torque_d, &att_params_.max_torque, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_torque_d, &att_params_.min_torque, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_dt_d, &att_params_.max_dt, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(min_dt_d, &att_params_.min_dt, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_v_torque_d, &max_v_torque_, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(max_v_dt_d, &max_v_dt_, sizeof(float)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(particle_cnt_d, &num_particles_, sizeof(int)), __FILE__, __LINE__)) {
        return false;
    }
    
    int dimensions = DIMENSIONS;
    if (!handleCudaError(cudaMemcpyToSymbol(dimensions_d, &dimensions, sizeof(int)), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpyToSymbol(att_params_d, &att_params_, sizeof(attitude_params)), __FILE__, __LINE__)) {
        return false;
    }
    
    return true;
}

bool PSOOptimizer::initializeParticles() {
    srand((unsigned)time(NULL));
    
    // Allocate host memory
    particles_ = (particle*)malloc(sizeof(particle));
    if (!particles_) {
        std::cerr << "Failed to allocate particle structure" << std::endl;
        return false;
    }
    
    particles_->position = (float*)malloc(sizeof(float) * num_particles_ * DIMENSIONS);
    particles_->velocity = (float*)malloc(sizeof(float) * num_particles_ * DIMENSIONS);
    particles_->pbest_pos = (float*)malloc(sizeof(float) * num_particles_ * DIMENSIONS);
    particles_->fitness = (float*)malloc(sizeof(float) * num_particles_);
    particles_->pbest_fit = (float*)malloc(sizeof(float) * num_particles_);
    
    if (!particles_->position || !particles_->velocity || !particles_->pbest_pos || 
        !particles_->fitness || !particles_->pbest_fit) {
        std::cerr << "Failed to allocate particle arrays" << std::endl;
        return false;
    }
    
    // Initialize global best
    gbest_.fitness = -FLT_MAX;
    
    // Initialize each particle
    int best_particle = 0;
    for (int i = 0; i < num_particles_; i++) {
        // Initialize torque variables
        for (int dim = 0; dim < TORQUE_DIMS; dim++) {
            int idx = PARTICLE_POS_IDX(i, dim);
            float torque_range = att_params_.max_torque - att_params_.min_torque;
            
            particles_->position[idx] = RND() * torque_range + att_params_.min_torque;
            particles_->velocity[idx] = (RND() - 0.5f) * 2.0f * max_v_torque_;
            particles_->pbest_pos[idx] = particles_->position[idx];
        }
        
        // Initialize time step variable
        int dt_idx = PARTICLE_POS_IDX(i, DT_IDX);
        float dt_range = att_params_.max_dt - att_params_.min_dt;
        
        particles_->position[dt_idx] = RND() * dt_range + att_params_.min_dt;
        particles_->velocity[dt_idx] = (RND() - 0.5f) * 2.0f * max_v_dt_;
        particles_->pbest_pos[dt_idx] = particles_->position[dt_idx];
        
        // Evaluate initial fitness
        particles_->fitness[i] = fit(particles_->position, i, &att_params_);
        particles_->pbest_fit[i] = particles_->fitness[i];
        
        // Track global best
        if (i == 0 || particles_->pbest_fit[i] > gbest_.fitness) {
            best_particle = i;
            gbest_.fitness = particles_->fitness[i];
        }
    }
    
    // Copy global best position
    for (int dim = 0; dim < DIMENSIONS; dim++) {
        int idx = PARTICLE_POS_IDX(best_particle, dim);
        gbest_.position[dim] = particles_->position[idx];
    }
    
    return true;
}

bool PSOOptimizer::optimize(double* X, double* U, double* dt) {

    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);
    
    // Initialize CUDA and particles
    if (!initializeCUDA() || !initializeParticles()) {
        std::cerr << "Initialization failed" << std::endl;
        cleanup();
        return false;
    }
    
    // Copy initial data to device
    size_t particle_data_size = sizeof(float) * num_particles_ * DIMENSIONS;
    if (!handleCudaError(cudaMemcpy(position_d_, particles_->position, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(velocity_d_, particles_->velocity, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_pos_d_, particles_->pbest_pos, particle_data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(fitness_d_, particles_->fitness, sizeof(float) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(pbest_fit_d_, particles_->pbest_fit, sizeof(float) * num_particles_, cudaMemcpyHostToDevice), __FILE__, __LINE__) ||
        !handleCudaError(cudaMemcpy(gbest_d_, &gbest_, sizeof(particle_gbest), cudaMemcpyHostToDevice), __FILE__, __LINE__)) {
        cleanup();
        return false;
    }
    
    if (verbose_) {
        std::cout << "Starting PSO optimization..." << std::endl;
        std::cout << "Particles: " << num_particles_ << ", Dimensions: " << DIMENSIONS << std::endl;
        std::cout << "Max iterations: " << max_iterations_ << std::endl;
        std::cout << "Initial best fitness: " << gbest_.fitness << std::endl;
    }
    
    // Main optimization loop
    int shared_mem_size = sizeof(float) * ThreadsPerBlock + sizeof(int) * ThreadsPerBlock;

    // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    
    // Calculate execution time
    float temp_time;
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time; // Accumulate

    // Start timing
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    for (int iter = 0; iter < max_iterations_; iter++) {
        // Launch PSO update kernel
        move<<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>(
            position_d_, velocity_d_, fitness_d_, pbest_pos_d_, pbest_fit_d_,
            gbest_d_, aux_, aux_pos_);
        
        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__)) {
            cleanup();
            return false;
        }
        
        // Launch global best finding kernel
        findBest<<<1, 32>>>(gbest_d_, aux_, aux_pos_, position_d_);
        
        if (!handleCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__)) {
            cleanup();
            return false;
        }
        
        // Progress reporting
        if (verbose_ && (iter % 100 == 0 || iter == max_iterations_ - 1)) {
            particle_gbest current_best;
            if (handleCudaError(cudaMemcpy(&current_best, gbest_d_, sizeof(particle_gbest), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) {
                std::cout << "Iteration " << iter << ": Best fitness = " << current_best.fitness << std::endl;
            }
        }
    }
    
    // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);

    // Calculate execution time
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&exec_time_, start_event_, stop_event_), __FILE__, __LINE__);
    
    handleCudaError(cudaEventRecord(start_event_), __FILE__, __LINE__);

    // Copy final results
    if (!handleCudaError(cudaMemcpy(&gbest_, gbest_d_, sizeof(particle_gbest), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) {
        cleanup();
        return false;
    }

    extractResults(X, U, dt);

    dt_opt_ = gbest_.position[DT_IDX];
    total_time_ = dt_opt_ * N_STEPS;
    final_fitness_ = gbest_.fitness;
    exec_time_ = exec_time_ / 1000.0f; // Convert to seconds

    results_valid_ = true;
    configured_ = true;
    
    if (verbose_) {
        std::cout << "Optimization completed!" << std::endl;
        std::cout << "Final fitness: " << final_fitness_ << std::endl;
        std::cout << "Execution time: " << exec_time_ << " seconds" << std::endl;
    }
    
    // Stop timing
    handleCudaError(cudaEventRecord(stop_event_), __FILE__, __LINE__);
    
    // Calculate execution time
    handleCudaError(cudaEventSynchronize(stop_event_), __FILE__, __LINE__);
    handleCudaError(cudaEventElapsedTime(&temp_time, start_event_, stop_event_), __FILE__, __LINE__);
    setup_time_ += temp_time; // Accumulate
    setup_time_ /= 1000.0f; // Convert to seconds

    return true;
}

void PSOOptimizer::extractResults(double* X, double* U, double* dt) {

    double dt_double = static_cast<double>(dt_opt_);

    // Simulate trajectory using RK4 for high accuracy
    float current_state[n_states], next_state[n_states];
    
    // Initialize with initial conditions
    for (int i = 0; i < n_quat; i++) {
        current_state[i] = att_params_.initial_quat[i];
        X[i] = static_cast<double>(current_state[i]);
    }
    for (int i = 0; i < n_vel; i++) {
        current_state[n_quat + i] = att_params_.initial_omega[i];
        X[n_quat + i] = static_cast<double>(current_state[n_quat + i]);
    }

    for (int step = 0; step < N_STEPS; step++) {
        float controls[n_controls];
        int torque_idx = TORQUE_IDX(step, 0);
        for (int axis = 0; axis < n_controls; axis++) {
            controls[axis] = gbest_.position[torque_idx + axis];
        }
        
        euler(current_state, controls, dt_opt_, next_state, &att_params_);
        
        // Store next state
        for (int i = 0; i < n_states; i++) {
            current_state[i] = next_state[i];
            X[n_states * (step + 1) + i] = static_cast<double>(current_state[i]);
        }

        for (int axis = 0; axis < n_controls; axis++) {
            U[torque_idx + axis] = static_cast<double>(gbest_.position[torque_idx + axis]);
        }

        dt[step] = dt_double;
    }
}

bool PSOOptimizer::getStats(double& final_fitness, double& setup_time, double& exec_time) const {
    if (!results_valid_) {
        std::cerr << "Warning: No valid results available. Call optimize() first." << std::endl;
        return false;
    }

    final_fitness = static_cast<double>(final_fitness_);
    setup_time = static_cast<double>(setup_time_);
    exec_time = static_cast<double>(exec_time_);

    return true;
}

void PSOOptimizer::printResults() const {
    if (!results_valid_) {
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
    std::cout << "===============================\n" << std::endl;

}

void PSOOptimizer::reset() {
    cleanup();
    results_valid_ = false;
    configured_ = false;
}

void PSOOptimizer::cleanup() {
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
    
    // Free device memory
    if (position_d_) { cudaFree(position_d_); position_d_ = nullptr; }
    if (velocity_d_) { cudaFree(velocity_d_); velocity_d_ = nullptr; }
    if (fitness_d_) { cudaFree(fitness_d_); fitness_d_ = nullptr; }
    if (pbest_pos_d_) { cudaFree(pbest_pos_d_); pbest_pos_d_ = nullptr; }
    if (pbest_fit_d_) { cudaFree(pbest_fit_d_); pbest_fit_d_ = nullptr; }
    if (gbest_d_) { cudaFree(gbest_d_); gbest_d_ = nullptr; }
    if (aux_) { cudaFree(aux_); aux_ = nullptr; }
    if (aux_pos_) { cudaFree(aux_pos_); aux_pos_ = nullptr; }

    // Destroy CUDA events
    if (start_event_) { cudaEventDestroy(start_event_); start_event_ = nullptr; }
    if (stop_event_) { cudaEventDestroy(stop_event_); stop_event_ = nullptr; }
}

bool PSOOptimizer::handleCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) 
                  << " in " << file << " at line " << line << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int result = mainish(argc, argv);
    return result;
}