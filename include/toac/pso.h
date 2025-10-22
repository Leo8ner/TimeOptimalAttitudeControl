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

#define MAX_SWITCHES_PER_AXIS 3   // Typical for spacecraft: 1-3 switches optimal
#define N_SIGNS n_controls                 // Initial control direction per axis
#define N_SWITCH_TIMES 7          // Switch times for all axes
#define DIMENSIONS_STO (N_SIGNS + N_SWITCH_TIMES + 1)  // Total dimensions for STO formulation

// Index helpers for new parameterization
#define SWITCH_TIME_IDX(axis, switch_num) (N_SIGNS + (axis)*MAX_SWITCHES_PER_AXIS + (switch_num))

/** @brief Number of discrete time steps in trajectory */
#define N_STEPS n_stp

/** @brief Number of torque control variables (3 axes × N_STEPS) */
#define TORQUE_DIMS (n_stp * n_controls)
#define DIMENSIONS_FULL (TORQUE_DIMS + 1)

/*==============================================================================
 * CUDA CONFIGURATION PARAMETERS
 *============================================================================*/

/** @brief Maximum PSO iterations for optimization convergence */
#define ITERATIONS 400

/** @brief Total number of particles in swarm */
#define N_PARTICLES 1920

/** @brief CUDA threads per block (must be multiple of 32, ≤ 1024) */
#define ThreadsPerBlock 128

/** @brief CUDA blocks per grid (must divide N_PARTICLES evenly) */
#define BlocksPerGrid (N_PARTICLES / ThreadsPerBlock)

/*==============================================================================
 * PRECISION CONFIGURATION
 *============================================================================*/

/** 
 * @brief Precision control: 0 = single (float), 1 = double precision
 * WARNING: Double precision is 2-32× slower on most GPUs and uses 2× memory
 */
#define USE_DOUBLE_PRECISION 0

#if USE_DOUBLE_PRECISION
    typedef double my_real;
    #define REAL_MAX DBL_MAX
    #define REAL_MIN -DBL_MAX
    
    // Math functions
    #define CURAND(x) curand_uniform_double(&x)
    
    // Constants
    #define REAL(x) x
#else
    typedef float my_real;
    #define REAL_MAX FLT_MAX
    #define REAL_MIN -FLT_MAX

    // Math functions
    #define CURAND(x) curand_uniform(&x)
    
    // Constants
    #define REAL(x) x##f
#endif

/*==============================================================================
 * INTEGRATION METHOD SELECTION
 *============================================================================*/

#define USE_RK4_INTEGRATION 1  // Use RK4 integration (0=Euler, 1=RK4)

#if USE_RK4_INTEGRATION
    #define INTEGRATE(X, U, dt, X_next, params) rk4(X, U, dt, X_next, params)
#else
    #define INTEGRATE(X, U, dt, X_next, params) euler(X, U, dt, X_next, params)
#endif

/*==============================================================================
 * PSO ALGORITHM PARAMETERS
 *============================================================================*/

/** @brief Inertia weight - controls particle momentum */
#define W REAL(2.0)

/** @brief Cognitive weight - attraction to personal best */
#define C_1 REAL(3.0)

/** @brief Social weight - attraction to global best */
#define C_2 REAL(1.0)

/** @brief Enable inertia weight decay over iterations (1=enable, 0=disable) */
#define DEC_INERTIA 1

/** @brief Minimum inertia weight for adaptive inertia */
#define MIN_W REAL(0.1)

/** @brief Enable cognitive weight decay over iterations (1=enable, 0=disable) */
#define DEC_C1 1

/** @brief Minimum cognitive weight for adaptive cognitive */
#define MIN_C1 REAL(0.5)

/** @brief Enable social weight decrease over iterations (1=enable, 0=disable) */
#define DEC_C2 1

/** @brief Minimum social weight for adaptive social */
#define MIN_C2 REAL(0.2)

/** @brief Sigmoid alpha parameter for smooth control transitions */
#define SIGMOID_ALPHA REAL(6.0)  // Higher = sharper sigmoid

/*==============================================================================
 * CONSTRAINT PENALTY COEFFICIENTS
 *============================================================================*/

/** @brief Penalty for quaternion normalization violations */
#define QUAT_NORM_PENALTY REAL(1000.0)

/** @brief Penalty for final state error from target */
#define FINAL_STATE_PENALTY REAL(1000.0)

/** @brief Penalty for excessive torque switching */
#define SWITCH_PENALTY REAL(0.0)

/** @brief Penalty coefficient for maneuver time minimization */
#define DT_PENALTY REAL(1.0)

/*==============================================================================
 * UTILITY MACROS
 *============================================================================*/

/** @brief Generate random my_real in [0,1] */
#define RND() ((my_real)rand() / RAND_MAX)

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
#define PARTICLE_IDX_STO(particle_id, dim) ((particle_id) * DIMENSIONS_STO  + (dim))
#define PARTICLE_IDX_FULL(particle_id, dim) ((particle_id) * DIMENSIONS_FULL  + (dim))

/** 
 * @brief Calculate dimension index for torque variable
 * @param step Time step index [0, N_STEPS-1]
 * @param axis Control axis index [0, n_controls-1] (typically 0=x, 1=y, 2=z)
 * @return Dimension index for torque variable
 */
#define TORQUE_IDX(step, axis) ((step) * n_controls + (axis))

/** @brief Dimension index for the time step variable */
#define DT_IDX_FULL (DIMENSIONS_FULL - 1)
#define DT_IDX_STO (DIMENSIONS_STO - 1)

/*==============================================================================
 * DATA STRUCTURE DEFINITIONS
 *============================================================================*/

/**
 * @brief Host-side particle swarm structure using structure-of-arrays layout
 */
typedef struct tag_particle {
    my_real *position;    /**< Particle positions [particle_cnt × DIMENSIONS] */
    my_real *velocity;    /**< Particle velocities [particle_cnt × DIMENSIONS] */
    my_real *fitness;     /**< Current fitness values [particle_cnt] */
    my_real *pbest_pos;   /**< Personal best positions [particle_cnt × DIMENSIONS] */
    my_real *pbest_fit;   /**< Personal best fitness values [particle_cnt] */
} particle;

/**
 * @brief Global best particle structure for device memory
 */
typedef struct tag_particle_gbest {
    my_real *position;      /**< Global best position vector */
    my_real fitness;               /**< Global best fitness value */
} particle_gbest;

/**
 * @brief Spacecraft attitude dynamics parameters
 */
typedef struct tag_attitude_params {
    my_real max_torque;                /**< Maximum torque magnitude [N⋅m] */
    my_real min_torque;                /**< Minimum torque magnitude [N⋅m] */
    my_real max_dt;                    /**< Maximum time step [seconds] */
    my_real min_dt;                    /**< Minimum time step [seconds] */
    my_real target_quat[n_quat];       /**< Target quaternion [w,x,y,z] */
    my_real target_omega[n_vel];       /**< Target angular velocity [rad/s] */
    my_real initial_quat[n_quat];      /**< Initial quaternion [w,x,y,z] */
    my_real initial_omega[n_vel];      /**< Initial angular velocity [rad/s] */
    my_real inertia[n_vel];            /**< Spacecraft inertia diagonal [kg⋅m²] */
} attitude_params;

/**
 * @brief Enumeration for PSO optimization method
 */
enum class PSOMethod {
    FULL,    // Optimize all control points
    STO      // Optimize switch times for bang-bang control
};

/*==============================================================================
 * CUDA KERNEL DECLARATIONS (same as before)
 *============================================================================*/

// Math utility functions
__host__ __device__ void skew_matrix_4(my_real *w, my_real *S);
__host__ __device__ void cross_product(my_real *a, my_real *b, my_real *result);
__host__ __device__ my_real quaternion_norm(my_real *q);

// Dynamics and integration
__host__ __device__ void attitude_dynamics(my_real *X, my_real *U, my_real *X_dot, attitude_params *params);
__host__ __device__ void euler(my_real *X, my_real *U, my_real dt, my_real *X_next, attitude_params *params);
__host__ __device__ void rk4(my_real *X, my_real *U, my_real dt, my_real *X_next, attitude_params *params);

// Fitness function
__host__ __device__ my_real fit_full(my_real *solution_vector, int particle_id, attitude_params *params);
__host__ __device__ my_real fit_sto(my_real *solution_vector, int particle_id, attitude_params *params);

// CUDA kernels
__global__ void move(my_real *position_d, my_real *velocity_d, my_real *fitness_d,
                     my_real *pbest_pos_d, my_real *pbest_fit_d, 
                     particle_gbest *gbest_d, my_real *aux, my_real *aux_pos);

__global__ void findBest(particle_gbest *gbest, my_real *aux, my_real *aux_pos, my_real *position_d);

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
     * @param method PSO optimization method (FULL or STO)
     * @param verbose Enable progress output during optimization
     */
    PSOOptimizer(casadi::DM& state_matrix, casadi::DM& input_matrix, casadi::DM& dt_matrix, PSOMethod method = PSOMethod::FULL, bool verbose = false);
    
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
    void setPSOParameters(int max_iterations = ITERATIONS, 
                         double inertia_weight = W,
                         double cognitive_weight = C_1, 
                         double social_weight = C_2);

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
    
    PSOMethod method_;                  /**< PSO optimization method (FULL or STO) */
    int dimensions_;                   /**< Total optimization dimensions */
    
    // Configuration parameters
    attitude_params att_params_;        /**< Spacecraft and optimization parameters */
    int max_iterations_;                /**< Maximum PSO iterations */
    int num_particles_;                 /**< Number of particles in swarm */
    my_real inertia_weight_;              /**< PSO inertia weight */
    my_real cognitive_weight_;            /**< PSO cognitive coefficient */
    my_real social_weight_;               /**< PSO social coefficient */
    
    // PSO velocity limits
    my_real max_v_torque_;                /**< Maximum velocity for torque variables */
    my_real max_v_dt_;                    /**< Maximum velocity for time step variable */
    
    // Optimization state
    bool configured_;                   /**< True if all parameters are set */
    bool results_valid_;                /**< True if optimization completed successfully */

    // Store LHS samples for reuse
    my_real** lhs_samples_;
    bool lhs_generated_;
    
    // Host memory pointers
    particle* particles_;               /**< Host particle data structure */
    particle_gbest* gbest_;              /**< Global best particle */
    
    // Device memory pointers
    my_real *position_d_, *velocity_d_, *fitness_d_;
    my_real *pbest_pos_d_, *pbest_fit_d_;
    my_real *gbest_pos_d_;
    particle_gbest *gbest_d_;
    my_real *aux_, *aux_pos_;

    // Output references
    casadi::DM& X;             /**< Reference to output state trajectory DM */
    casadi::DM& U;             /**< Reference to output control trajectory DM */
    casadi::DM& dt;            /**< Reference to output time step DM */
    
    // CUDA timing
    cudaEvent_t start_event_, stop_event_;

    // Performance metrics
    float exec_time_;                   /**< Pure execution time excluding setup/teardown */
    float total_time_;                  /**< Total maneuver time */
    my_real final_fitness_;               /**< Final fitness value */
    my_real dt_opt_;                      /**< Optimized time step */
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
    bool initializeParticles_full(bool regenerate_lhs);
    bool initializeParticles_sto(bool regenerate_lhs);

    /**
     * @brief Generate Latin Hypercube Samples for particle initialization
     * @param samples Output matrix [num_particles_][DIMENSIONS]
     */
    void generateLHSSamples(my_real** samples);
    
    /**
     * @brief Extract results from optimization and populate output DMs
     * @return true on success, false on error and on invalid results
     */
    bool extractResults_full();
    bool extractResults_sto();
    
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