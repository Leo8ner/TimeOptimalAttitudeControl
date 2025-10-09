/**
 * @file pso.h
 * @brief C++ Class for CUDA-accelerated PSO Spacecraft Attitude Control Optimization
 * 
 * This header defines a C++ class that encapsulates the PSO optimization functionality
 * for spacecraft attitude control. The class provides a clean interface for setting
 * initial/target states and retrieving optimized trajectories.
 * 
 * @author Leonardo Eitner
 * @date 11/09/2025
 * @version 2.0
 */

#ifndef PSO_H
#define PSO_H

/*==============================================================================
 * SYSTEM INCLUDES
 *============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <memory>

/*==============================================================================
 * EXTERNAL LIBRARY INCLUDES  
 *============================================================================*/
#include <curand_kernel.h>
#include <casadi/casadi.hpp>
/*==============================================================================
 * LOCAL INCLUDES
 *============================================================================*/
#include <toac/symmetric_spacecraft.h>

/*==============================================================================
 * PROBLEM CONFIGURATION CONSTANTS
 *============================================================================*/

/** @brief Total optimization dimensions: torque variables + time step */
#define DIMENSIONS (n_stp * n_controls + 1)

/** @brief Number of discrete time steps in trajectory */
#define N_STEPS n_stp

/** @brief Number of torque control variables (3 axes × N_STEPS) */
#define TORQUE_DIMS (n_stp * n_controls)

/*==============================================================================
 * CUDA CONFIGURATION PARAMETERS
 *============================================================================*/

/** @brief Maximum PSO iterations for optimization convergence */
#define MAX_ITERA 200

/** @brief Total number of particles in swarm */
#define N_PARTICLES 1280

/** @brief CUDA threads per block (must be multiple of 32, ≤ 1024) */
#define ThreadsPerBlock 128

/** @brief CUDA blocks per grid (must divide N_PARTICLES evenly) */
#define BlocksPerGrid (N_PARTICLES / ThreadsPerBlock)

/*==============================================================================
 * PSO ALGORITHM PARAMETERS
 *============================================================================*/

/** @brief Inertia weight - controls particle momentum */
#define W 5.0f

/** @brief Cognitive weight - attraction to personal best */
#define C1 2.0f

/** @brief Social weight - attraction to global best */
#define C2 1.5f

/** @brief Minimum inertia weight for adaptive inertia */
#define MIN_W 0.4f

/** @brief Enable inertia weight decay over iterations (1=enable, 0=disable) */
#define DEC_INERTIA 1

/*==============================================================================
 * CONSTRAINT PENALTY COEFFICIENTS
 *============================================================================*/

/** @brief Penalty for quaternion normalization violations */
#define QUAT_NORM_PENALTY 1000.0f

/** @brief Penalty for final state error from target */
#define FINAL_STATE_PENALTY 1000.0f

/** @brief Penalty for excessive torque switching */
#define SWITCH_PENALTY 0.0f

/** @brief Penalty coefficient for maneuver time minimization */
#define DT_PENALTY 10.0f

/*==============================================================================
 * UTILITY MACROS
 *============================================================================*/

/** @brief Generate random float in [0,1] */
#define RND() ((float)rand() / RAND_MAX)

/** @brief CUDA error handling wrapper */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/*==============================================================================
 * ARRAY INDEXING MACROS
 *============================================================================*/

/** 
 * @brief Calculate flat array index for particle position/velocity
 * @param particle_id Particle index [0, N_PARTICLES-1]
 * @param dim Dimension index [0, DIMENSIONS-1]
 * @return Flat array index for structure-of-arrays layout
 */
#define PARTICLE_POS_IDX(particle_id, dim) ((particle_id) * DIMENSIONS + (dim))

/** 
 * @brief Calculate dimension index for torque variable
 * @param step Time step index [0, N_STEPS-1]
 * @param axis Control axis index [0, n_controls-1] (typically 0=x, 1=y, 2=z)
 * @return Dimension index for torque variable
 */
#define TORQUE_IDX(step, axis) ((step) * n_controls + (axis))

/** @brief Dimension index for the time step variable */
#define DT_IDX (n_controls * n_stp)

/*==============================================================================
 * DATA STRUCTURE DEFINITIONS
 *============================================================================*/

/**
 * @brief Host-side particle swarm structure using structure-of-arrays layout
 */
typedef struct tag_particle {
    float *position;    /**< Particle positions [particle_cnt × DIMENSIONS] */
    float *velocity;    /**< Particle velocities [particle_cnt × DIMENSIONS] */
    float *fitness;     /**< Current fitness values [particle_cnt] */
    float *pbest_pos;   /**< Personal best positions [particle_cnt × DIMENSIONS] */
    float *pbest_fit;   /**< Personal best fitness values [particle_cnt] */
} particle;

/**
 * @brief Global best particle structure for device memory
 */
typedef struct tag_particle_gbest {
    float position[DIMENSIONS];  /**< Global best position vector */
    float fitness;               /**< Global best fitness value */
} particle_gbest;

/**
 * @brief Spacecraft attitude dynamics parameters
 */
typedef struct tag_attitude_params {
    float max_torque;                /**< Maximum torque magnitude [N⋅m] */
    float min_torque;                /**< Minimum torque magnitude [N⋅m] */
    float max_dt;                    /**< Maximum time step [seconds] */
    float min_dt;                    /**< Minimum time step [seconds] */
    float target_quat[n_quat];       /**< Target quaternion [w,x,y,z] */
    float target_omega[n_vel];       /**< Target angular velocity [rad/s] */
    float initial_quat[n_quat];      /**< Initial quaternion [w,x,y,z] */
    float initial_omega[n_vel];      /**< Initial angular velocity [rad/s] */
    float inertia[n_vel];            /**< Spacecraft inertia diagonal [kg⋅m²] */
} attitude_params;

/*==============================================================================
 * CUDA KERNEL DECLARATIONS (same as before)
 *============================================================================*/

// Math utility functions
__host__ __device__ void skew_matrix_4(float *w, float *S);
__host__ __device__ void cross_product(float *a, float *b, float *result);
__host__ __device__ float quaternion_norm(float *q);

// Dynamics and integration
__host__ __device__ void attitude_dynamics(float *X, float *U, float *X_dot, attitude_params *params);
__host__ __device__ void euler(float *X, float *U, float dt, float *X_next, attitude_params *params);

// Fitness function
__host__ __device__ float fit(float *solution_vector, int particle_id, attitude_params *params);

// CUDA kernels
__global__ void move(float *position_d, float *velocity_d, float *fitness_d,
                     float *pbest_pos_d, float *pbest_fit_d, 
                     particle_gbest *gbest_d, float *aux, float *aux_pos);

__global__ void findBest(particle_gbest *gbest, float *aux, float *aux_pos, float *position_d);

/*==============================================================================
 * PSO OPTIMIZER CLASS
 *============================================================================*/

/**
 * @brief CUDA-accelerated Particle Swarm Optimization class for spacecraft attitude control
 * 
 * This class encapsulates all PSO functionality and provides a clean interface for
 * optimizing spacecraft attitude maneuvers. It manages CUDA memory, handles the
 * optimization process, and provides methods to extract results.
 * 
 * Usage example:
 * @code
 * PSOOptimizer optimizer;
 * optimizer.setSpacecraftParameters(inertia_x, inertia_y, inertia_z, max_torque);
 * optimizer.setInitialState(initial_quat, initial_omega);
 * optimizer.setTargetState(target_quat, target_omega);
 * optimizer.setTimeConstraints(min_dt, max_dt);
 * 
 * if (optimizer.optimize()) {
 *     auto results = optimizer.getResults();
 *     // Use results.U, results.X, results.dt
 * }
 * @endcode
 */
class PSOOptimizer {
public:
    /*==========================================================================
     * CONSTRUCTOR AND DESTRUCTOR
     *========================================================================*/
    
    /**
     * @brief Constructor - initializes PSO optimizer with default parameters
     * @param[out] state_matrix  Discrete state trajectory matrix (size: n_states x (N+1)).
     * @param[out] input_matrix  Discrete control input trajectory matrix (size: n_controls x N).
     * @param[out] dt_matrix Time step durations (vector or 1 x N DM depending on formulation).
     * @param verbose Enable progress output during optimization
     */
    PSOOptimizer(casadi::DM& state_matrix, casadi::DM& input_matrix, casadi::DM& dt_matrix, bool verbose = false);
    
    /**
     * @brief Destructor - cleans up allocated memory
     */
    ~PSOOptimizer();

    /*==========================================================================
     * CONFIGURATION METHODS
     *========================================================================*/
    
    /**
     * @brief Set PSO algorithm parameters
     * @param max_iterations Maximum number of PSO iterations
     * @param num_particles Number of particles in swarm
     * @param inertia_weight PSO inertia weight
     * @param cognitive_weight PSO cognitive coefficient (c1)
     * @param social_weight PSO social coefficient (c2)
     */
    void setPSOParameters(int max_iterations = MAX_ITERA, 
                         double inertia_weight = W,
                         double cognitive_weight = C1, 
                         double social_weight = C2);

    /**
     * @brief Set spacecraft initial and target states
     * @param initial_state Pointer to initial state array [q0, q1, q2, q3, wx, wy, wz]
     * @param target_state Pointer to target state array [q0, q1, q2, q3, wx, wy, wz]
     */
    void setStates(const double* initial_state, const double* target_state);

    /*==========================================================================
     * OPTIMIZATION METHODS
     *========================================================================*/
    
    /**
     * @brief Run PSO optimization to find optimal attitude maneuver
     * @param regenerate_lhs If true, regenerate Latin Hypercube Samples for particle initialization
     * @return true if optimization completed successfully, false on error
     */
    bool optimize(bool regenerate_lhs = true);
   
    /**
     * @brief Get summary statistics of the last optimization run.
     *
     * Provides basic performance and convergence metrics gathered during the
     * most recent optimize() execution.
     *
     * @param[out] final_fitness   Best (lowest) objective function value achieved.
     * @param[out] setup_time      Time taken for setup (seconds).
     * @param[out] exec_time       Pure execution time excluding setup / teardown (seconds) if tracked.
     * @return true if statistics are available (i.e., an optimization has completed),
     *         false otherwise.
     */
    bool getStats(double& final_fitness, double& setup_time, double& exec_time) const;

    /*==========================================================================
     * UTILITY METHODS
     *========================================================================*/
    
    /**
     * @brief Print optimization summary to console
     */
    void printResults() const;
    
    /**
     * @brief Reset optimizer state for new optimization run
     */
    void reset();

private:
    /*==========================================================================
     * PRIVATE MEMBER VARIABLES
     *========================================================================*/
    
    // Configuration parameters
    attitude_params att_params_;        /**< Spacecraft and optimization parameters */
    int max_iterations_;                /**< Maximum PSO iterations */
    int num_particles_;                 /**< Number of particles in swarm */
    float inertia_weight_;              /**< PSO inertia weight */
    float cognitive_weight_;            /**< PSO cognitive coefficient */
    float social_weight_;               /**< PSO social coefficient */
    
    // PSO velocity limits
    float max_v_torque_;                /**< Maximum velocity for torque variables */
    float max_v_dt_;                    /**< Maximum velocity for time step variable */
    
    // Optimization state
    bool configured_;                   /**< True if all parameters are set */
    bool results_valid_;                /**< True if optimization completed successfully */

    // Store LHS samples for reuse
    float** lhs_samples_;
    bool lhs_generated_;
    
    // Host memory pointers
    particle* particles_;               /**< Host particle data structure */
    particle_gbest gbest_;              /**< Global best particle */
    
    // Device memory pointers
    float *position_d_, *velocity_d_, *fitness_d_;
    float *pbest_pos_d_, *pbest_fit_d_;
    particle_gbest *gbest_d_;
    float *aux_, *aux_pos_;

    // Output references
    casadi::DM& X;             /**< Reference to output state trajectory DM */
    casadi::DM& U;             /**< Reference to output control trajectory DM */
    casadi::DM& dt;            /**< Reference to output time step DM */
    
    // CUDA timing
    cudaEvent_t start_event_, stop_event_;

    // Performance metrics
    float exec_time_;                   /**< Pure execution time excluding setup/teardown */
    float total_time_;                  /**< Total maneuver time */
    float final_fitness_;               /**< Final fitness value */
    float dt_opt_;                      /**< Optimized time step */
    float setup_time_;                  /**< Time spent in setup (seconds) */
    bool verbose_;                     /**< Enable progress output during optimization */

    // Results storage
    casadi::DM X_opt_;               /**< Optimized state trajectory */
    casadi::DM U_opt_;               /**< Optimized control trajectory */
    casadi::DM dt_opt_vec_;          /**< Optimized time step vector */
    
    /*==========================================================================
     * PRIVATE METHODS
     *========================================================================*/
    
    /**
     * @brief Allocate all necessary device memory
     * @return true on success, false on error
     */
    bool allocateDeviceMemory();
    
    /**
     * @brief Copy immutable constants to device memory
     * @return true on success, false on error
     */
    bool copyImmutableConstants();

    /**
     * @brief Copy mutable state parameters to device memory
     * @return true on success, false on error
     */
    bool copyMutableStateParameters();

    /**
     * @brief Clean up all allocated memory (host and device)
     */
    void cleanup();
    
    /**
     * @brief Initialize particle swarm with random positions and velocities
     * @param regenerate_lhs If true, regenerate Latin Hypercube Samples
     * @return true on success, false on error
     */
    bool initializeParticles(bool regenerate_lhs);

    /**
     * @brief Generate Latin Hypercube Samples for particle initialization
     * @param samples Output matrix [num_particles_][DIMENSIONS]
     */
    void generateLHSSamples(float** samples);
    
    /**
     * @brief Extract results from optimization and populate output DMs
     * @return true on success, false on error and on invalid results
     */
    bool extractResults();
    
    /**
     * @brief CUDA error handling function
     * @param err CUDA error code
     * @param file Source file name
     * @param line Line number
     * @return true if no error, false if error occurred
     */
    static bool handleCudaError(cudaError_t err, const char* file, int line);
};

#endif /* PSO_H */