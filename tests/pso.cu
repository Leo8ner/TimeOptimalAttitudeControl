/**
 * @file pso.cu
 * @brief CUDA-accelerated Particle Swarm Optimization for Spacecraft Attitude Control
 * 
 * This implementation provides a GPU-accelerated PSO algorithm for optimizing
 * spacecraft attitude maneuvers. The algorithm finds optimal torque sequences
 * and time steps to minimize maneuver time while satisfying physical constraints.
 * 
 * Key features:
 * - Structure-of-arrays memory layout for efficient GPU memory access
 * - Shared memory optimization for global best finding
 * - Adaptive inertia weight for improved convergence
 * - Quaternion-based attitude representation with automatic normalization
 * - Multiple integration schemes (Euler for speed, RK4 for accuracy)
 * 
 * @author [Your Name]
 * @date [Date]
 * @version 1.0
 */

/*==============================================================================
 * INCLUDES
 *============================================================================*/
#include "pso.h"

/*==============================================================================
 * GLOBAL VARIABLE DEFINITIONS
 *============================================================================*/

/** @brief Maximum velocity for torque variables in PSO updates */
float max_v_torque = (float)(2 * tau_max);

/** @brief Maximum velocity for time step variable in PSO updates */
float max_v_dt = (float)(dt_max - dt_min);

/** @brief Runtime particle count (set from command line arguments) */
unsigned int particle_cnt;

/** @brief Global best particle across all iterations */
particle_gbest gbest;

/** @brief Spacecraft attitude dynamics and optimization parameters */
attitude_params att_params;

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
 * MATHEMATICAL UTILITY FUNCTIONS
 *============================================================================*/

/**
 * @brief Construct 4×4 skew-symmetric matrix for quaternion kinematics
 * 
 * Builds the skew-symmetric matrix S(ω) used in quaternion kinematic equations:
 * q̇ = 0.5 * S(ω) * q, where ω is the angular velocity vector.
 * 
 * Matrix structure:
 * S = [ 0   -ωx  -ωy  -ωz]
 *     [ωx    0   ωz  -ωy]
 *     [ωy  -ωz    0   ωx]
 *     [ωz   ωy  -ωx    0]
 * 
 * @param w Angular velocity vector [ωx, ωy, ωz] (rad/s)
 * @param S Output 4×4 matrix stored in row-major order (16 elements)
 */
__host__ __device__ void skew_matrix_4(float *w, float *S) {
    S[0] = 0;     S[1] = -w[0]; S[2] = -w[1]; S[3] = -w[2];
    S[4] = w[0];  S[5] = 0;     S[6] = w[2];  S[7] = -w[1];
    S[8] = w[1];  S[9] = -w[2]; S[10] = 0;    S[11] = w[0];
    S[12] = w[2]; S[13] = w[1]; S[14] = -w[0]; S[15] = 0;
}

/**
 * @brief Compute cross product of two 3D vectors
 * 
 * Calculates result = a × b using the standard cross product formula.
 * Essential for computing angular momentum and torque interactions.
 * 
 * @param a First vector [ax, ay, az]
 * @param b Second vector [bx, by, bz]  
 * @param result Output vector [ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx]
 */
__host__ __device__ void cross_product(float *a, float *b, float *result) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2]; 
    result[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * @brief Calculate Euclidean norm of quaternion
 * 
 * Computes ||q|| = √(w² + x² + y² + z²). Used for quaternion normalization
 * to maintain unit quaternion constraint for valid rotations.
 * 
 * @param q Quaternion [w, x, y, z]
 * @return Quaternion magnitude (should be 1.0 for unit quaternions)
 */
__host__ __device__ float quaternion_norm(float *q) {
    return sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}

/*==============================================================================
 * SPACECRAFT ATTITUDE DYNAMICS
 *============================================================================*/

/**
 * @brief Compute spacecraft attitude dynamics derivatives
 * 
 * Implements the coupled quaternion kinematics and rigid body dynamics:
 * 
 * Quaternion kinematics: q̇ = 0.5 * S(ω) * q
 * Angular dynamics: ω̇ = I⁻¹ * (τ - ω × (I*ω))
 * 
 * Where:
 * - q = [w,x,y,z] is the attitude quaternion
 * - ω = [ωx,ωy,ωz] is the angular velocity vector
 * - τ = [τx,τy,τz] is the applied torque vector
 * - I is the spacecraft inertia tensor (diagonal)
 * 
 * @param X State vector [q0,q1,q2,q3,ω0,ω1,ω2] (7 elements)
 * @param U Control torque vector [τx,τy,τz] (N⋅m)
 * @param X_dot Output derivative vector (7 elements)
 * @param params Spacecraft physical parameters
 */
__host__ __device__ void attitude_dynamics(float *X, float *U, float *X_dot, attitude_params *params) {
    float *q = X;                // Quaternion [w,x,y,z]
    float *w = &X[n_quat];       // Angular velocity [ωx,ωy,ωz]
    
    // Quaternion kinematics: q̇ = 0.5 * S(ω) * q
    float S[16];
    skew_matrix_4(w, S);
    for(int i = 0; i < n_quat; i++) {
        X_dot[i] = 0.5f * (S[i*4]*q[0] + S[i*4+1]*q[1] + S[i*4+2]*q[2] + S[i*4+3]*q[3]);
    }
    
    // Angular dynamics: ω̇ = I⁻¹ * (τ - ω × (I*ω))
    // First compute I*ω (inertia is diagonal)
    float Iw[n_vel] = {
        params->inertia[0] * w[0], 
        params->inertia[1] * w[1], 
        params->inertia[2] * w[2]
    };
    
    // Compute gyroscopic torque ω × (I*ω)
    float w_cross_Iw[n_vel];
    cross_product(w, Iw, w_cross_Iw);

    // Apply Euler's rotation equation
    for(int i = 0; i < n_vel; i++) {
        X_dot[n_quat+i] = (U[i] - w_cross_Iw[i]) / params->inertia[i];
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
__host__ __device__ void rk4(float *X, float *U, float dt, float *X_next, attitude_params *params) {
    float k1[n_states], k2[n_states], k3[n_states], k4[n_states];
    float X_temp[n_states];
    
    // k1 = f(X, U)
    attitude_dynamics(X, U, k1, params);
    
    // k2 = f(X + dt/2*k1, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/2.0f*k1[i];
    attitude_dynamics(X_temp, U, k2, params);

    // k3 = f(X + dt/2*k2, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt/2.0f*k2[i];
    attitude_dynamics(X_temp, U, k3, params);

    // k4 = f(X + dt*k3, U)
    for(int i = 0; i < n_states; i++) X_temp[i] = X[i] + dt*k3[i];
    attitude_dynamics(X_temp, U, k4, params);

    // Final integration step
    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt/6.0f*(k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }
    
    // Maintain quaternion unit constraint
    float q_norm = quaternion_norm(X_next);
    if(q_norm > 1e-6f) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

/**
 * @brief Forward Euler integration for attitude dynamics
 * 
 * Provides fast, first-order numerical integration:
 * y_{n+1} = y_n + h * f(t_n, y_n)
 * 
 * Less accurate than RK4 but much faster, making it suitable for fitness
 * evaluation during PSO optimization where speed is critical.
 * Automatically normalizes quaternion to maintain unit constraint.
 * 
 * @param X Current state vector [q,ω] (7 elements)
 * @param U Control input vector [τx,τy,τz] (3 elements)
 * @param dt Integration time step (seconds)
 * @param X_next Output next state vector (7 elements)
 * @param params Spacecraft parameters
 */
__host__ __device__ void euler(float *X, float *U, float dt, float *X_next, attitude_params *params) {
    float X_dot[n_states];
    
    // Compute derivatives at current state
    attitude_dynamics(X, U, X_dot, params);

    // Forward Euler step
    for(int i = 0; i < n_states; i++) {
        X_next[i] = X[i] + dt * X_dot[i];
    }
    
    // Maintain quaternion unit constraint
    float q_norm = quaternion_norm(X_next);
    if(q_norm > 1e-6f) {
        for(int i = 0; i < n_quat; i++) X_next[i] /= q_norm;
    }
}

/*==============================================================================
 * FITNESS EVALUATION FUNCTION
 *============================================================================*/

/**
 * @brief Evaluate fitness of candidate PSO solution
 * 
 * Simulates spacecraft attitude maneuver using candidate torque sequence and
 * time step, then computes fitness based on maneuver time and constraint violations.
 * 
 * Fitness components:
 * 1. Time penalty: Minimize total maneuver time
 * 2. Final state error: Penalize deviation from target attitude
 * 3. Quaternion norm violation: Penalize non-unit quaternions
 * 4. Control switching: Penalize excessive torque reversals
 * 
 * Uses Euler integration for computational efficiency during optimization.
 * Higher fitness values indicate better solutions.
 * 
 * @param solution_vector Complete solution [τ₀,τ₁,...,τₙ,dt] (151 elements)
 * @param particle_id Particle index for array indexing in SoA layout
 * @param params Spacecraft and optimization parameters
 * @return Fitness value (higher = better, negative indicates violations)
 */
__host__ __device__ float fit(float *solution_vector, int particle_id, attitude_params *params) {
    // Extract time step from solution vector
    float dt = solution_vector[PARTICLE_POS_IDX(particle_id, DT_IDX)];
    
    // Initialize state with initial conditions
    float X[n_states], X_next[n_states];
    for(int i = 0; i < n_quat; i++) X[i] = params->initial_quat[i];
    for(int i = 0; i < n_vel; i++) X[n_quat+i] = params->initial_omega[i];

    float constraint_violation = 0.0f;
    int switches = 0;

    // Forward simulate trajectory through all time steps
    for(int step = 0; step < N_STEPS; step++) {
        // Extract control torques for current time step
        float U[n_controls];
        for(int axis = 0; axis < n_controls; axis++) {
            int torque_idx = TORQUE_IDX(step, axis);
            U[axis] = solution_vector[PARTICLE_POS_IDX(particle_id, torque_idx)];
            
            // Count torque direction reversals (undesirable for actuators)
            if (step > 0) {
                int previous_idx = TORQUE_IDX(step-1, axis);
                float previous_torque = solution_vector[PARTICLE_POS_IDX(particle_id, previous_idx)];
                if (U[axis] * previous_torque < 0) {
                    switches++;
                }
            }
        }
        
        // Integrate dynamics (Euler for speed during optimization)
        euler(X, U, dt, X_next, params);

        // Penalize quaternion norm deviations
        float q_norm = quaternion_norm(X_next);
        constraint_violation += QUAT_NORM_PENALTY * fabsf(q_norm - 1.0f);

        // Update state for next iteration
        for(int i = 0; i < n_states; i++) X[i] = X_next[i];
    }

    // Penalize excessive control switching
    constraint_violation += SWITCH_PENALTY * switches;
    
    // Compute final state error from target
    float final_error = 0.0f;
    for(int i = 0; i < n_quat; i++) {
        final_error += pow(X[i] - params->target_quat[i], 2);
    }
    for(int i = 0; i < n_vel; i++) {
        final_error += pow(X[n_quat+i] - params->target_omega[i], 2);
    }
    final_error = sqrt(final_error);
    constraint_violation += FINAL_STATE_PENALTY * final_error;

    // Calculate total maneuver time
    float total_time = dt * N_STEPS;
    
    // Return negative of objective (PSO maximizes fitness)
    return -DT_PENALTY * total_time - constraint_violation;
}

/*==============================================================================
 * CUDA KERNEL IMPLEMENTATIONS
 *============================================================================*/

/**
 * @brief Main PSO particle update kernel
 * 
 * Each thread handles one particle and performs:
 * 1. PSO velocity and position updates for all dimensions
 * 2. Fitness evaluation using attitude dynamics simulation
 * 3. Personal best updates
 * 4. Shared memory reduction to find block-level best particles
 * 
 * Uses adaptive inertia weight and separate velocity limits for torque/time variables.
 * Implements queue-based optimization: only particles better than global best
 * are considered for global best updates.
 * 
 * Shared memory layout:
 * - [0, blockDim.x): Fitness values of candidate particles
 * - [blockDim.x, 2*blockDim.x): Particle IDs of candidates
 * 
 * @param position_d All particle positions [N_PARTICLES × DIMENSIONS]
 * @param velocity_d All particle velocities [N_PARTICLES × DIMENSIONS]
 * @param fitness_d Current fitness values [N_PARTICLES]
 * @param pbest_pos_d Personal best positions [N_PARTICLES × DIMENSIONS]
 * @param pbest_fit_d Personal best fitness values [N_PARTICLES]
 * @param gbest_d Global best particle (read/write)
 * @param aux Output array for block-level best fitness [BlocksPerGrid]
 * @param aux_pos Output array for block-level best particle IDs [BlocksPerGrid]
 */
__global__ void move(float *position_d, float *velocity_d, float *fitness_d,
                     float *pbest_pos_d, float *pbest_fit_d, 
                     particle_gbest *gbest_d, float *aux, float *aux_pos) {
    
    // Thread and block identification
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    // Shared memory allocation for block-level reduction
    extern __shared__ float sharedMemory[];
    float *privateBestQueue = (float *)sharedMemory;                    
    int *privateBestParticleQueue = (int *)&sharedMemory[blockDim.x];   
    __shared__ unsigned int queue_num;
    
    // Bounds check for particle count
    if (particle_id >= particle_cnt_d) return;

    // Initialize shared memory queue
    if (tidx == 0) queue_num = 0;
    __syncthreads();

    // Initialize random number generators (one per thread)
    curandState state1, state2;
    curand_init((unsigned long long)clock() + particle_id * 2, 0, 0, &state1);
    curand_init((unsigned long long)clock() + particle_id * 2 + 1, 0, 0, &state2);

    // Apply adaptive inertia weight
    float w = w_d;
    if (DEC_INERTIA) {
        w = w_d - (w_d - MIN_W) * particle_id / N_PARTICLES; // Linear decrease
    }
    
    // Update all dimensions for this particle
    for (int dim = 0; dim < dimensions_d; dim++) {
        int pos_idx = PARTICLE_POS_IDX(particle_id, dim);
        
        // Load current values
        float pos = position_d[pos_idx];
        float vel = velocity_d[pos_idx];
        float pbest_pos = pbest_pos_d[pos_idx];
        float gbest_pos = gbest_d->position[dim];
        
        // PSO velocity update equation
        vel = w * vel +
              c1_d * curand_uniform(&state1) * (pbest_pos - pos) +
              c2_d * curand_uniform(&state2) * (gbest_pos - pos);
        
        // Apply dimension-specific velocity and position constraints
        if (dim < TORQUE_DIMS) {
            // Torque variables: apply torque velocity limits and bounds
            vel = fmax(-max_v_torque_d, fmin(max_v_torque_d, vel));
            pos = pos + vel;
            pos = fmax(min_torque_d, fmin(max_torque_d, pos));
        } else {
            // Time step variable: apply dt velocity limits and bounds  
            vel = fmax(-max_v_dt_d, fmin(max_v_dt_d, vel));
            pos = pos + vel;
            pos = fmax(min_dt_d, fmin(max_dt_d, pos));
        }
        
        // Store updated values
        position_d[pos_idx] = pos;
        velocity_d[pos_idx] = vel;
    }
    
    // Evaluate fitness of updated position
    float new_fitness = fit(position_d, particle_id, &att_params_d);
    fitness_d[particle_id] = new_fitness;
    
    // Update personal best if improved
    if (new_fitness > pbest_fit_d[particle_id]) {
        pbest_fit_d[particle_id] = new_fitness;
        for (int dim = 0; dim < dimensions_d; dim++) {
            pbest_pos_d[PARTICLE_POS_IDX(particle_id, dim)] = 
                position_d[PARTICLE_POS_IDX(particle_id, dim)];
        }
    }
    
    __syncthreads();

    // Queue optimization: only consider particles better than global best
    if (new_fitness > gbest_d->fitness) {
        unsigned int my_index = atomicAdd(&queue_num, 1);
        if (my_index < blockDim.x) { // Prevent queue overflow
            privateBestQueue[my_index] = new_fitness;
            privateBestParticleQueue[my_index] = particle_id;
        }
    }
    
    __syncthreads();

    // Block-level reduction: find best particle in queue
    if (tidx == 0) {
        aux[blockIdx.x] = -FLT_MAX;
        aux_pos[blockIdx.x] = -1;
        
        if (queue_num > 0) {
            float best_fitness = privateBestQueue[0];
            int best_idx = 0;
            
            // Linear search through queued particles
            for (unsigned int i = 1; i < queue_num && i < blockDim.x; i++) {
                if (privateBestQueue[i] > best_fitness) {
                    best_fitness = privateBestQueue[i];
                    best_idx = i;
                }
            }
            
            // Store block-level results
            aux[blockIdx.x] = best_fitness;
            aux_pos[blockIdx.x] = (float)privateBestParticleQueue[best_idx];
        }
    }
}

/**
 * @brief Global best reduction kernel using warp-level primitives
 * 
 * Performs final reduction across all block-level results to identify the
 * single global best particle. Uses efficient warp shuffle operations for
 * high-speed parallel reduction within a single warp.
 * 
 * Algorithm:
 * 1. Load block-level results into warp (32 threads)
 * 2. Perform butterfly reduction using __shfl_down_sync
 * 3. Thread 0 updates global best if improvement found
 * 4. Copy best particle position if global best updated
 * 
 * @param gbest Global best particle structure (updated in-place)
 * @param aux Block-level best fitness values [BlocksPerGrid]
 * @param aux_pos Block-level best particle IDs [BlocksPerGrid]  
 * @param position_d All particle positions (for copying best position)
 */
__global__ void findBest(particle_gbest *gbest, float *aux, float *aux_pos,
                         float *position_d) {
    int tid = threadIdx.x;
    
    // Load data into registers (pad with sentinel values for inactive threads)
    float my_fitness = (tid < BlocksPerGrid) ? aux[tid] : -FLT_MAX;
    int my_particle = (tid < BlocksPerGrid) ? (int)aux_pos[tid] : -1;
    
    // Warp-level butterfly reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_fitness = __shfl_down_sync(0xffffffff, my_fitness, offset);
        int other_particle = __shfl_down_sync(0xffffffff, my_particle, offset);
        if (other_fitness > my_fitness) {
            my_fitness = other_fitness;
            my_particle = other_particle;
        }
    }
    
    // Thread 0 holds the global result after reduction
    if (tid == 0 && my_fitness > gbest->fitness) {
        gbest->fitness = my_fitness;
        if (my_particle >= 0) {
            // Copy best particle position to global best
            for (int dim = 0; dim < DIMENSIONS; dim++) {
                gbest->position[dim] = position_d[PARTICLE_POS_IDX(my_particle, dim)];
            }
        }
        __threadfence(); // Ensure global visibility
    }
}

/*==============================================================================
 * HOST FUNCTION IMPLEMENTATIONS
 *============================================================================*/

/**
 * @brief Initialize particle swarm with random positions and velocities
 * 
 * Allocates memory for all particle data arrays and initializes each particle
 * with random positions within physical bounds and random velocities within
 * PSO velocity limits. Evaluates initial fitness and identifies initial global best.
 * 
 * Memory allocation pattern (Structure-of-Arrays):
 * - All positions in contiguous block: [p0_d0, p0_d1, ..., p1_d0, p1_d1, ...]
 * - All velocities in contiguous block: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...]
 * - Personal bests in similar layout
 * 
 * @param p Particle structure to initialize (allocated but arrays uninitialized)
 * @post All particle arrays allocated and filled with random valid values
 * @post Global best particle identified and stored
 * @post Random seed initialized from system time
 */
void ParticleInit(particle *p) {
    // Initialize random number generator
    srand((unsigned)time(NULL));
    
    // Allocate structure-of-arrays memory layout
    p->position = (float *)malloc(sizeof(float) * particle_cnt * DIMENSIONS);
    p->velocity = (float *)malloc(sizeof(float) * particle_cnt * DIMENSIONS);
    p->pbest_pos = (float *)malloc(sizeof(float) * particle_cnt * DIMENSIONS);
    p->fitness = (float *)malloc(sizeof(float) * particle_cnt);
    p->pbest_fit = (float *)malloc(sizeof(float) * particle_cnt);
    
    // Check memory allocation success
    if (!p->position || !p->velocity || !p->pbest_pos || !p->fitness || !p->pbest_fit) {
        printf("Error: Failed to allocate particle memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize global best to worst possible value
    gbest.fitness = -FLT_MAX;
    
    // Initialize each particle
    int best_particle = 0;
    for (int i = 0; i < particle_cnt; i++) {
        // Initialize torque control variables [dimensions 0-149]
        for (int dim = 0; dim < TORQUE_DIMS; dim++) {
            int idx = PARTICLE_POS_IDX(i, dim);
            float torque_range = att_params.max_torque - att_params.min_torque;
            
            // Random position within torque bounds
            p->position[idx] = RND() * torque_range + att_params.min_torque;
            
            // Random velocity within velocity limits  
            p->velocity[idx] = (RND() - 0.5f) * 2.0f * max_v_torque;
            
            // Initialize personal best to current position
            p->pbest_pos[idx] = p->position[idx];
        }
        
        // Initialize time step variable [dimension 150]
        int dt_idx = PARTICLE_POS_IDX(i, DT_IDX);
        float dt_range = att_params.max_dt - att_params.min_dt;
        
        // Random time step within bounds
        p->position[dt_idx] = RND() * dt_range + att_params.min_dt;
        
        // Random velocity for time step
        p->velocity[dt_idx] = (RND() - 0.5f) * 2.0f * max_v_dt;
        
        // Initialize personal best
        p->pbest_pos[dt_idx] = p->position[dt_idx];
        
        // Evaluate initial fitness
        p->fitness[i] = fit(p->position, i, &att_params);       
        p->pbest_fit[i] = p->fitness[i]; 
        
        // Track global best particle
        if(i == 0 || p->pbest_fit[i] > gbest.fitness) {
            best_particle = i;                 
            gbest.fitness = p->fitness[i];                    
        } 
    }

    // Copy global best position
    for (int dim = 0; dim < DIMENSIONS; dim++) {
        int idx = PARTICLE_POS_IDX(best_particle, dim);
        gbest.position[dim] = p->position[idx];
    }
    
    printf("Particle initialization complete. Best initial fitness: %.6f\n", gbest.fitness);
}

/**
 * @brief CUDA error handling with file and line information
 * 
 * Checks CUDA API return codes and provides detailed error reporting.
 * Terminates program execution on any CUDA error to prevent silent failures.
 * 
 * @param err CUDA error code from API call
 * @param file Source filename where error occurred  
 * @param line Line number where error occurred
 * @post Program terminates if err != cudaSuccess
 */
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {        
        printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);        
        exit(EXIT_FAILURE);    
    }
}

/**
 * @brief Write optimized trajectory to CSV file for analysis
 * 
 * Simulates the complete trajectory using the best solution found by PSO
 * and writes state history, control history, and timing data to CSV format.
 * Uses RK4 integration for maximum accuracy in final trajectory generation.
 * 
 * CSV file structure:
 * - X matrix: State history [n_states × (N_STEPS+1)]
 * - U matrix: Control history [n_controls × N_STEPS]  
 * - T: Total maneuver time
 * - dt: Time step array
 * 
 * @param filename Output CSV file path
 * @param params Spacecraft parameters used for simulation
 * @param best_solution Optimal solution found by PSO
 * @param execution_time PSO execution time (for metadata)
 * @param iterations Number of PSO iterations completed
 */
void writeTrajectoryCSV(const char* filename, attitude_params params, 
                       particle_gbest* best_solution, float execution_time, int iterations) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not create CSV file %s\n", filename);
        return;
    }
    
    // Extract optimal time step
    float dt_opt = best_solution->position[DT_IDX];
    
    // Allocate trajectory storage
    float X[n_states][N_STEPS+1], U[n_controls][N_STEPS];
    float current_state[n_states], next_state[n_states];
    
    // Initialize with initial conditions
    for(int i = 0; i < n_quat; i++) current_state[i] = params.initial_quat[i];
    for(int i = 0; i < n_vel; i++) current_state[n_quat+i] = params.initial_omega[i];
    
    // Simulate complete trajectory using RK4 for accuracy
    for(int step = 0; step < N_STEPS+1; step++) {
        // Store current state
        for(int i = 0; i < n_states; i++) X[i][step] = current_state[i];
        
        if(step == N_STEPS) break; // No control action at final time
        
        // Extract optimal control sequence
        for(int axis = 0; axis < n_controls; axis++) {
            int torque_idx = TORQUE_IDX(step, axis);
            U[axis][step] = best_solution->position[torque_idx];
        }
        
        // Integrate to next state using high-accuracy method
        float controls[n_controls];
        for(int axis = 0; axis < n_controls; axis++) controls[axis] = U[axis][step];
        rk4(current_state, controls, dt_opt, next_state, &params);
        for(int i = 0; i < n_states; i++) current_state[i] = next_state[i];
    }
    
    // Write state trajectory matrix
    fprintf(file, "X\n");
    for(int state = 0; state < n_states; state++) {
        for(int step = 0; step < N_STEPS+1; step++) {
            fprintf(file, "%.6g", X[state][step]);
            if(step < N_STEPS) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    
    // Write control trajectory matrix
    fprintf(file, "U\n");
    for(int control = 0; control < n_controls; control++) {
        for(int step = 0; step < N_STEPS; step++) {
            fprintf(file, "%.6g", U[control][step]);
            if(step < N_STEPS-1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    
    // Write total maneuver time
    fprintf(file, "T\n%.6g\n\n", dt_opt * N_STEPS);
    
    // Write time step array (constant for this implementation)
    fprintf(file, "dt\n");
    for(int step = 0; step < N_STEPS; step++) {
        fprintf(file, "%.6g", dt_opt);
        if(step < N_STEPS-1) fprintf(file, ",");
    }
    fprintf(file, "\n");
    
    fclose(file);
    printf("Trajectory written to %s\n", filename);
    printf("Maneuver time: %.6f seconds, Final fitness: %.6f\n", 
           dt_opt * N_STEPS, best_solution->fitness);
}

/*==============================================================================
 * MAIN FUNCTION
 *============================================================================*/

/**
 * @brief Main PSO optimization routine for spacecraft attitude control
 * 
 * Orchestrates the complete PSO optimization process:
 * 1. Parameter initialization and validation
 * 2. Host and device memory allocation  
 * 3. Particle swarm initialization
 * 4. CUDA constant memory setup
 * 5. Main PSO iteration loop
 * 6. Results extraction and analysis
 * 7. Trajectory generation and output
 * 8. Memory cleanup
 * 
 * @return 0 on successful completion, EXIT_FAILURE on errors
 */
int main() {
    // Initialize runtime arguments with default values
    arguments args = {MAX_ITERA, N_PARTICLES, ThreadsPerBlock, BlocksPerGrid, VERBOSE};
    
    /*----------------------------------------------------------------------
     * SPACECRAFT PARAMETER INITIALIZATION
     *--------------------------------------------------------------------*/
    
    // Physical constraint bounds
    att_params.max_torque = (float)tau_max;   // Maximum torque [N⋅m]
    att_params.min_torque = (float)-tau_max;  // Minimum torque [N⋅m] 
    att_params.max_dt = (float)dt_max;        // Maximum time step [s]
    att_params.min_dt = (float)dt_min;        // Minimum time step [s]

    // Target attitude: 180° rotation about x-axis
    att_params.target_quat[0] = 0.0f; // w component
    att_params.target_quat[1] = 1.0f; // x component  
    att_params.target_quat[2] = 0.0f; // y component
    att_params.target_quat[3] = 0.0f; // z component

    // Target angular velocity: rest state
    for (int i = 0; i < n_vel; i++) att_params.target_omega[i] = 0.0f;

    // Initial conditions: identity quaternion (no initial rotation)
    att_params.initial_quat[0] = 1.0f; // w (identity quaternion)
    att_params.initial_quat[1] = 0.0f; // x
    att_params.initial_quat[2] = 0.0f; // y
    att_params.initial_quat[3] = 0.0f; // z
    
    // Initial angular velocity: zero (starting from rest)
    for (int i = 0; i < n_vel; i++) att_params.initial_omega[i] = 0.0f;
    
    // Spacecraft inertia tensor (diagonal elements)
    att_params.inertia[0] = (float)i_x; // Ixx [kg⋅m²]
    att_params.inertia[1] = (float)i_y; // Iyy [kg⋅m²]
    att_params.inertia[2] = (float)i_z; // Izz [kg⋅m²]

    // Set runtime particle count
    particle_cnt = args.particle_cnt;
    
    /*----------------------------------------------------------------------
     * TIMING AND PERFORMANCE MEASUREMENT SETUP
     *--------------------------------------------------------------------*/
    
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float exe_time;
    
    clock_t begin_app = clock();
    clock_t begin_init = begin_app;
    
    /*----------------------------------------------------------------------
     * HOST MEMORY ALLOCATION AND PARTICLE INITIALIZATION
     *--------------------------------------------------------------------*/
    
    printf("Initializing %d particles with %d dimensions...\n", particle_cnt, DIMENSIONS);
    particle *p = (particle*)malloc(sizeof(particle));
    if (!p) {
        printf("Error: Failed to allocate particle structure\n");
        return EXIT_FAILURE;
    }
    
    ParticleInit(p);
    
    /*----------------------------------------------------------------------
     * DEVICE MEMORY ALLOCATION
     *--------------------------------------------------------------------*/
    
    float *position_d, *velocity_d, *fitness_d, *pbest_pos_d, *pbest_fit_d;
    particle_gbest *gbest_d;
    float *aux, *aux_pos;
    
    // Allocate particle data arrays (structure-of-arrays layout)
    size_t particle_data_size = sizeof(float) * particle_cnt * DIMENSIONS;
    HANDLE_ERROR(cudaMalloc((void **)&position_d, particle_data_size));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, particle_data_size));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, particle_data_size));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(float) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(float) * particle_cnt));
    
    // Allocate global best particle
    HANDLE_ERROR(cudaMalloc((void **)&gbest_d, sizeof(particle_gbest)));
    HANDLE_ERROR(cudaMemcpy(gbest_d, &gbest, sizeof(particle_gbest), cudaMemcpyHostToDevice));

    // Allocate auxiliary arrays for block-level reduction
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(float) * BlocksPerGrid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(float) * BlocksPerGrid));
        
    /*----------------------------------------------------------------------
     * HOST-TO-DEVICE DATA TRANSFER
     *--------------------------------------------------------------------*/
    
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, particle_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, particle_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, particle_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(float) * particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(float) * particle_cnt, cudaMemcpyHostToDevice));
    
    /*----------------------------------------------------------------------
     * CONSTANT MEMORY INITIALIZATION
     *--------------------------------------------------------------------*/
    
    // PSO algorithm parameters
    float w = W, c1 = C1, c2 = C2;
    HANDLE_ERROR(cudaMemcpyToSymbol(w_d, &w, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c1_d, &c1, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c2_d, &c2, sizeof(float)));
    
    // Physical bounds
    HANDLE_ERROR(cudaMemcpyToSymbol(max_torque_d, &att_params.max_torque, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(min_torque_d, &att_params.min_torque, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_dt_d, &att_params.max_dt, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(min_dt_d, &att_params.min_dt, sizeof(float)));
    
    // PSO velocity limits
    HANDLE_ERROR(cudaMemcpyToSymbol(max_v_torque_d, &max_v_torque, sizeof(float)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_v_dt_d, &max_v_dt, sizeof(float)));
    
    // Problem dimensions
    HANDLE_ERROR(cudaMemcpyToSymbol(particle_cnt_d, &particle_cnt, sizeof(int)));
    int dimensions_host = DIMENSIONS;
    HANDLE_ERROR(cudaMemcpyToSymbol(dimensions_d, &dimensions_host, sizeof(int)));
    
    // Complete parameter structure
    HANDLE_ERROR(cudaMemcpyToSymbol(att_params_d, &att_params, sizeof(attitude_params)));
        
    clock_t end_init = clock();
    
    /*----------------------------------------------------------------------
     * MAIN PSO OPTIMIZATION LOOP  
     *--------------------------------------------------------------------*/
    
    printf("Starting PSO optimization with %d particles, %d dimensions\n", 
           particle_cnt, DIMENSIONS);
    printf("Target: 180° rotation about x-axis\n");
    printf("Max iterations: %d\n", args.max_iter);
    
    HANDLE_ERROR(cudaEventRecord(start));

    // Calculate shared memory requirements
    int shared_mem_size = sizeof(float) * args.threads_per_block +  // Fitness queue
                         sizeof(int) * args.threads_per_block;     // Particle ID queue

    // Main optimization loop
    for (unsigned int iter = 0; iter < args.max_iter; iter++) {
        // Launch particle update kernel
        move<<<args.blocks_per_grid, args.threads_per_block, shared_mem_size>>>
            (position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d, 
             gbest_d, aux, aux_pos);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Launch global best reduction kernel
        findBest<<<1, 32>>>(gbest_d, aux, aux_pos, position_d);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Optional progress reporting
        if (args.verbose && (iter % 100 == 0 || iter == args.max_iter - 1)) {
            // Copy current best fitness for progress monitoring
            particle_gbest current_best;
            HANDLE_ERROR(cudaMemcpy(&current_best, gbest_d, sizeof(particle_gbest), cudaMemcpyDeviceToHost));
            printf("Iteration %d: Best fitness = %.6f\n", iter, current_best.fitness);
        }
    }

    HANDLE_ERROR(cudaEventRecord(stop));
    
    /*----------------------------------------------------------------------
     * RESULTS EXTRACTION AND ANALYSIS
     *--------------------------------------------------------------------*/
    
    // Copy final results back to host
    HANDLE_ERROR(cudaMemcpy(&gbest, gbest_d, sizeof(particle_gbest), cudaMemcpyDeviceToHost));

    // Calculate timing results
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&exe_time, start, stop));
    clock_t end_app = clock();
    
    /*----------------------------------------------------------------------
     * RESULTS REPORTING
     *--------------------------------------------------------------------*/
    
    printf("\n=== PSO Optimization Results ===\n");
    printf("Best fitness: %10.6f\n", gbest.fitness);
    printf("Optimal dt: %10.6f seconds\n", gbest.position[DT_IDX]);
    printf("Total maneuver time: %10.6f seconds\n", gbest.position[DT_IDX] * N_STEPS);
    
    // Analyze control effort
    float max_torque_used = 0.0f;
    for (int i = 0; i < TORQUE_DIMS; i++) {
        float torque_mag = fabsf(gbest.position[i]);
        if (torque_mag > max_torque_used) max_torque_used = torque_mag;
    }
    printf("Maximum torque used: %10.6f N⋅m\n", max_torque_used);
    
    printf("\n=== Timing Results ===\n");
    printf("[Initialization]: %lf sec\n", (float)(end_init - begin_init) / CLOCKS_PER_SEC);
    printf("[CUDA Execution]: %f sec\n", exe_time / 1000);
    printf("[Total Time    ]: %lf sec\n", (float)(end_app - begin_app) / CLOCKS_PER_SEC);
    printf("[Iterations/sec]: %.2f\n", args.max_iter / (exe_time / 1000));

    /*----------------------------------------------------------------------
     * TRAJECTORY GENERATION AND OUTPUT
     *--------------------------------------------------------------------*/
    
    writeTrajectoryCSV("trajectory.csv", att_params, &gbest, exe_time / 1000, args.max_iter);
    
    /*----------------------------------------------------------------------
     * MEMORY CLEANUP
     *--------------------------------------------------------------------*/
    
    // Free host memory
    free(p->position);
    free(p->velocity);
    free(p->pbest_pos);
    free(p->fitness);
    free(p->pbest_fit);
    free(p);
    
    // Free device memory
    cudaFree(position_d);
    cudaFree(velocity_d);
    cudaFree(fitness_d);
    cudaFree(pbest_pos_d);
    cudaFree(pbest_fit_d);
    cudaFree(gbest_d);
    cudaFree(aux);
    cudaFree(aux_pos);
    
    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nOptimization completed successfully.\n");
    return 0;
}