/**
 * @file pso.h
 * @brief Particle Swarm Optimization for Spacecraft Attitude Control
 * 
 * This header defines the CUDA-accelerated PSO implementation for optimizing
 * spacecraft attitude maneuvers. The optimization finds optimal torque sequences
 * and time steps to minimize maneuver time while satisfying attitude constraints.
 * 
 * @author [Leonardo Eitner]
 * @date [11/09/2025]
 * @version 1.0
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

/*==============================================================================
 * EXTERNAL LIBRARY INCLUDES  
 *============================================================================*/
#include <curand_kernel.h>

/*==============================================================================
 * LOCAL INCLUDES
 *============================================================================*/
#include "symmetric_spacecraft.h"

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
#define MAX_ITERA 500

/** @brief Total number of particles in swarm */
#define N_PARTICLES 1280

/** @brief CUDA threads per block (must be multiple of 32, ≤ 1024) */
#define ThreadsPerBlock 128

/** @brief CUDA blocks per grid (must divide N_PARTICLES evenly) */
#define BlocksPerGrid (N_PARTICLES / ThreadsPerBlock)

/** @brief Verbosity level for debug output (0=silent, 1=verbose) */
#define VERBOSE 0

/*==============================================================================
 * PSO ALGORITHM PARAMETERS
 *============================================================================*/

/** @brief Inertia weight - controls particle momentum */
#define W 10.0f

/** @brief Cognitive weight - attraction to personal best */
#define C1 3.0f

/** @brief Social weight - attraction to global best */
#define C2 1.5f

/** @brief Minimum inertia weight for adaptive inertia */
#define MIN_W 0.1f

/** @brief Enable inertia weight decay over iterations (1=enable, 0=disable) */
#define DEC_INERTIA 1

/*==============================================================================
 * CONSTRAINT PENALTY COEFFICIENTS
 *============================================================================*/

/** @brief Penalty for quaternion normalization violations */
#define QUAT_NORM_PENALTY 1000.0f

/** @brief Penalty for final state error from target */
#define FINAL_STATE_PENALTY 10.0f

/** @brief Penalty for excessive torque switching */
#define SWITCH_PENALTY 0.05f

/** @brief Penalty coefficient for maneuver time minimization */
#define DT_PENALTY 1.0f

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
 * BOUNDS CLAMPING MACROS
 *============================================================================*/

/** @brief Clamp torque value to allowable range */
#define CLAMP_TORQUE(val) (fmax(att_params.min_torque, fmin(att_params.max_torque, val)))

/** @brief Clamp time step value to allowable range */
#define CLAMP_DT(val) (fmax(att_params.min_dt, fmin(att_params.max_dt, val)))

/*==============================================================================
 * DATA STRUCTURE DEFINITIONS
 *============================================================================*/

/**
 * @brief Host-side particle swarm structure using structure-of-arrays layout
 * 
 * This structure stores particle data in separate arrays for each property,
 * which is more efficient for CUDA memory transfers and kernel access patterns.
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
 * 
 * Stores the single best solution found across all particles and iterations.
 * Uses array-of-structures layout since only one instance exists.
 */
typedef struct tag_particle_gbest {
    float position[DIMENSIONS];  /**< Global best position vector */
    float fitness;               /**< Global best fitness value */
} particle_gbest;

/**
 * @brief Command line arguments and runtime configuration
 */
typedef struct tag_arguments {
    int max_iter;           /**< Maximum PSO iterations */
    int particle_cnt;       /**< Number of particles in swarm */
    int threads_per_block;  /**< CUDA threads per block */
    int blocks_per_grid;    /**< CUDA blocks per grid */
    int verbose;            /**< Verbosity level for output */
} arguments;

/**
 * @brief Spacecraft attitude dynamics parameters
 * 
 * Contains all physical parameters and constraints needed for attitude
 * dynamics simulation and optimization bounds.
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
 * GLOBAL VARIABLE DECLARATIONS
 *============================================================================*/

/** @brief Maximum velocity for torque variables in PSO */
extern float max_v_torque;

/** @brief Maximum velocity for time step variable in PSO */
extern float max_v_dt;

/** @brief Total number of particles (runtime configurable) */
extern unsigned int particle_cnt;

/** @brief Global best particle across all iterations */
extern particle_gbest gbest;

/** @brief Spacecraft attitude dynamics parameters */
extern attitude_params att_params;

/*==============================================================================
 * FUNCTION DECLARATIONS
 *============================================================================*/

/**
 * @brief Initialize particle swarm with random positions and velocities
 * 
 * Allocates memory for particle data structures and initializes each particle
 * with random positions within bounds and random velocities. Also evaluates
 * initial fitness values and sets personal bests.
 * 
 * @param p Pointer to particle structure to initialize
 * @pre p must be allocated but member arrays can be uninitialized
 * @post All particle arrays allocated and initialized with random values
 */
void ParticleInit(particle *p);

/**
 * @brief CUDA error handling and reporting function
 * 
 * Checks CUDA API return codes and prints error messages with file/line info
 * before terminating program execution on errors.
 * 
 * @param err CUDA error code to check
 * @param file Source file name (typically __FILE__)
 * @param line Source line number (typically __LINE__)
 * @post Program exits if err != cudaSuccess
 */
static void HandleError(cudaError_t err, const char *file, int line);

/*==============================================================================
 * CUDA KERNEL DECLARATIONS
 *============================================================================*/

/**
 * @brief Main PSO particle update kernel
 * 
 * Updates particle positions and velocities according to PSO equations,
 * evaluates fitness, updates personal bests, and identifies candidates
 * for global best using shared memory optimization.
 * 
 * @param position_d Device array of particle positions [N_PARTICLES × DIMENSIONS]
 * @param velocity_d Device array of particle velocities [N_PARTICLES × DIMENSIONS]
 * @param fitness_d Device array of current fitness values [N_PARTICLES]
 * @param pbest_pos_d Device array of personal best positions [N_PARTICLES × DIMENSIONS]
 * @param pbest_fit_d Device array of personal best fitness values [N_PARTICLES]
 * @param gbest_d Device pointer to global best particle
 * @param aux Device array for block-level reduction results [BlocksPerGrid]
 * @param aux_pos Device array for best particle IDs per block [BlocksPerGrid]
 * 
 * @note Requires shared memory: sizeof(float) * ThreadsPerBlock + sizeof(int) * ThreadsPerBlock
 * @note One thread per particle, launch with <<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>
 */
__global__ void move(
    float *position_d,
    float *velocity_d,
    float *fitness_d,
    float *pbest_pos_d,
    float *pbest_fit_d,
    particle_gbest *gbest_d,
    float *aux,
    float *aux_pos
);

/**
 * @brief Global best reduction kernel
 * 
 * Performs final reduction across block-level results to find the single
 * global best particle and updates the global best structure.
 * 
 * @param gbest Device pointer to global best particle (updated in-place)
 * @param aux Device array of block-level best fitness values [BlocksPerGrid]
 * @param aux_pos Device array of block-level best particle IDs [BlocksPerGrid]
 * @param position_d Device array of all particle positions (for copying best position)
 * 
 * @note Launch with <<<1, 32>>> - single block with warp-level reduction
 */
__global__ void findBest(
    particle_gbest *gbest,
    float *aux,
    float *aux_pos,
    float *position_d
);

/**
 * @brief Spacecraft attitude control fitness function
 * 
 * Evaluates the quality of a candidate solution by simulating the spacecraft
 * attitude dynamics and computing penalties for constraint violations and
 * maneuver time.
 * 
 * @param solution_vector Array containing torque sequence and time step [DIMENSIONS]
 * @param particle_id Particle index for proper array indexing
 * @param params Pointer to attitude dynamics parameters
 * @return Fitness value (higher is better, negative indicates constraint violations)
 * 
 * @note Uses Euler integration for computational efficiency during optimization
 * @note Fitness = -time_penalty - constraint_penalties (minimize time + violations)
 */
__host__ __device__ float fit(
    float *solution_vector,
    int particle_id,
    attitude_params *params
);

#endif /* PSO_H */
