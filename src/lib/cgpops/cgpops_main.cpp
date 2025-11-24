/**
 * @file cgpops_main_test.cpp
 * @brief CGPOPS problem setup for time-optimal spacecraft attitude maneuver
 * 
 * PROBLEM DESCRIPTION:
 * ====================
 * This file configures a time-optimal control problem for spacecraft attitude
 * reorientation using quaternion representation. The problem seeks the minimum-time
 * trajectory to rotate the spacecraft from an initial attitude to a target attitude
 * while respecting torque constraints and ensuring quaternion normalization.
 * 
 * MATHEMATICAL FORMULATION:
 * =========================
 * 
 * Objective:
 *   Minimize J = tf (final time)
 * 
 * State Vector (7 components):
 *   x = [q₀, q₁, q₂, q₃, ωx, ωy, ωz]ᵀ
 *   where:
 *     q = [q₀, q₁, q₂, q₃]ᵀ : Unit quaternion (scalar-first convention)
 *     ω = [ωx, ωy, ωz]ᵀ    : Angular velocity in body frame (rad/s)
 * 
 * Control Vector (3 components):
 *   u = [τx, τy, τz]ᵀ : Applied torques in body frame (N⋅m)
 * 
 * Dynamics:
 *   q̇ = 0.5 * Ω(ω) * q           (Quaternion kinematics)
 *   I·ω̇ + ω × (I·ω) = τ          (Euler's equations)
 * 
 * Path Constraints:
 *   ||q||² = 1  (quaternion normalization)
 * 
 * Boundary Conditions:
 *   x(t₀) = x₀  (specified initial state)
 *   x(tf) = xf  (specified final state)
 * 
 * COLLOCATION METHOD:
 * ===================
 * Legendre-Gauss-Radau (LGR) pseudospectral collocation is used to transcribe
 * the continuous-time optimal control problem into a nonlinear programming (NLP)
 * problem, which is then solved using IPOPT.
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 */

#include <cgpops/cgpops_main.hpp>
#include <cmath>
#include <algorithm>

// =============================================================================
// STATE AND CONTROL VECTOR INDEXING CONSTANTS
// =============================================================================
// These constants provide meaningful names for array indices, improving code
// readability and reducing errors from magic numbers.

namespace StateIndex {
    constexpr int Q0 = 0;  ///< Quaternion scalar component (q₀)
    constexpr int Q1 = 1;  ///< Quaternion vector x-component (q₁)
    constexpr int Q2 = 2;  ///< Quaternion vector y-component (q₂)
    constexpr int Q3 = 3;  ///< Quaternion vector z-component (q₃)
    constexpr int WX = 4;  ///< Angular velocity x-component (ωx, rad/s)
    constexpr int WY = 5;  ///< Angular velocity y-component (ωy, rad/s)
    constexpr int WZ = 6;  ///< Angular velocity z-component (ωz, rad/s)
    constexpr int TOTAL = 7;  ///< Total number of state components
}

namespace ControlIndex {
    constexpr int TX = 0;  ///< Torque x-component (τx, N⋅m)
    constexpr int TY = 1;  ///< Torque y-component (τy, N⋅m)
    constexpr int TZ = 2;  ///< Torque z-component (τz, N⋅m)
    constexpr int TOTAL = 3;  ///< Total number of control components
}

namespace ConstraintIndex {
    constexpr int QUAT_NORM = 0;  ///< Quaternion normalization constraint
    constexpr int TOTAL = 1;  ///< Total number of path constraints
}

// =============================================================================
// PROBLEM DIMENSION CONSTANTS
// =============================================================================

namespace ProblemDimensions {
    constexpr int NUM_PHASES = 1;           ///< Single-phase problem
    constexpr int NUM_STATIC_PARAMS = 0;    ///< No static parameters
    constexpr int NUM_EVENT_CONSTRAINTS = 0; ///< No linkage constraints
    constexpr int NUM_ENDPOINT_PARAMS = 0;   ///< No endpoint parameters
    constexpr int NUM_INTEGRAL_COSTS = 0;    ///< No integral costs (Mayer problem)
    constexpr int NUM_PHASE_PARAMS = 0;      ///< No phase parameters
}

namespace DefaultValues {
    constexpr double QUAT_NORM_EXACT = 1.0;  ///< Required quaternion norm value
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Validates that a quaternion is properly normalized
 * 
 * @param q Quaternion components [q0, q1, q2, q3]
 * @param tolerance Allowable deviation from unit norm
 * @return true if ||q|| ≈ 1 within tolerance, false otherwise
 */
bool isQuaternionValid(const std::vector<double>& q, double tolerance = 1e-6) {
    if (q.size() < 4) return false;
    double norm_sq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    return std::abs(norm_sq - 1.0) < tolerance;
}

/**
 * @brief Initializes an array with uniform values
 * 
 * @tparam T Array element type
 * @param arr Pointer to array
 * @param size Number of elements
 * @param value Value to set all elements to
 */
template<typename T>
void initializeArray(T* arr, int size, T value) {
    for (int i = 0; i < size; i++) {
        arr[i] = value;
    }
}

/**
 * @brief Copies state vector components to bound arrays
 * 
 * Helper function to reduce repetitive code when setting initial/final
 * state bounds that are equality constraints.
 * 
 * @param dest Destination array (bound array)
 * @param src Source vector (state values)
 */
void copyStateToBounds(double* dest, const std::vector<double>& src) {
    for (size_t i = 0; i < src.size() && i < StateIndex::TOTAL; i++) {
        dest[i] = src[i];
    }
}

/**
 * @brief Sets up uniform mesh grid for LGR collocation
 * 
 * Creates a mesh with equal-sized intervals and uniform collocation points.
 * This is the initial mesh; adaptive mesh refinement may adjust it during
 * optimization.
 * 
 * @param phase Phase index (0 for single-phase problem)
 * @param num_intervals Number of mesh intervals
 * @param collocation_points Number of collocation points per interval
 */
void setupUniformMesh(int phase, int num_intervals, int collocation_points) {
    double fraction[num_intervals];
    int colpoints[num_intervals];
    
    // Create uniform mesh: each interval spans equal fraction of time domain
    double interval_fraction = 1.0 / static_cast<double>(num_intervals);
    
    for (int m = 0; m < num_intervals; m++) {
        fraction[m] = interval_fraction;
        colpoints[m] = collocation_points;
    }
    
    setRPMDG(phase, num_intervals, fraction, colpoints);
}

// =============================================================================
// MAIN CGPOPS SETUP FUNCTION
// =============================================================================

/**
 * @brief Configures and solves time-optimal spacecraft attitude maneuver problem
 * 
 * This function sets up the complete optimal control problem including:
 * - Problem dimensions and phase structure
 * - State and control bounds
 * - Initial and final boundary conditions
 * - Path constraints (quaternion normalization)
 * - Initial guess for NLP solver
 * - Collocation mesh configuration
 * 
 * The problem is then transcribed to NLP form using LGR pseudospectral
 * collocation and solved using the IPOPT optimizer.
 * 
 * @param[out] cgpopsResults Matrix containing optimization results including:
 *   - Optimal state trajectory
 *   - Optimal control trajectory
 *   - Time grid points
 *   - Objective function value (minimum time)
 * @param[in] initial_state Initial state vector [q₀, q₁, q₂, q₃, ωx, ωy, ωz]
 *   Must be a valid unit quaternion with angular velocity components
 * @param[in] final_state Final (target) state vector [q₀, q₁, q₂, q₃, ωx, ωy, ωz]
 *   Must be a valid unit quaternion with angular velocity components
 * 
 * @note External constants must be defined before calling this function:
 *   - T_min, T_max: Time bounds (seconds)
 *   - q_min, q_max: Quaternion component bounds (typically [-1, 1])
 *   - w_min, w_max: Angular velocity bounds (rad/s)
 *   - tau_min, tau_max: Torque bounds (N⋅m)
 *   - T_0: Initial time guess (seconds)
 *   - numintervalsG: Number of mesh intervals
 *   - initcolptsG: Initial collocation points per interval
 *   - i_x, i_y, i_z: Spacecraft principal moments of inertia (kg⋅m²)
 * 
 * @pre initial_state.size() == 7 and final_state.size() == 7
 * @pre Quaternion components of both states must be normalized
 * 
 * @throw May throw exceptions from CGPOPS framework if problem setup is invalid
 */
void cgpops_go(doubleMat& cgpopsResults, 
               const std::vector<double>& initial_state, 
               const std::vector<double>& final_state)
{
    // =========================================================================
    // PROBLEM DIMENSION CONFIGURATION
    // =========================================================================
    // Define the structure of the optimal control problem including number
    // of phases, parameters, and constraints.
    
    PG   = ProblemDimensions::NUM_PHASES;
    nsG  = ProblemDimensions::NUM_STATIC_PARAMS;
    nbG  = ProblemDimensions::NUM_EVENT_CONSTRAINTS;
    nepG = ProblemDimensions::NUM_ENDPOINT_PARAMS;
    
    // Allocate memory for phase-specific arrays
    initGlobalVars();
    
    // =========================================================================
    // PHASE 1 CONFIGURATION
    // =========================================================================
    // Define the number of states, controls, and constraints for the single
    // phase of this problem.
    
    constexpr int phase = 0;  // Phase index (zero-based)
    
    nxG[phase]  = StateIndex::TOTAL;        // 7 states: quaternion + angular velocity
    nuG[phase]  = ControlIndex::TOTAL;      // 3 controls: torque components
    nqG[phase]  = ProblemDimensions::NUM_INTEGRAL_COSTS;  // 0: Mayer formulation (no running cost)
    ncG[phase]  = ConstraintIndex::TOTAL;   // 1 path constraint: ||q|| = 1
    nppG[phase] = ProblemDimensions::NUM_PHASE_PARAMS;    // 0: no phase parameters
    
    // Load any global tabular data (none required for analytical equations)
    setGlobalTabularData();
    
    // =========================================================================
    // MESH GRID CONFIGURATION
    // =========================================================================
    // Configure the LGR collocation mesh. The initial mesh uses uniform
    // intervals; adaptive mesh refinement may adjust this during optimization.
    //
    // Strategy: Start with a moderate number of intervals (e.g., 10-20) with
    // 3-5 collocation points each. This provides good initial resolution while
    // allowing the adaptive refinement to add points where needed for accuracy.
    
    setupUniformMesh(phase, numintervalsG, initcolptsG);
    
    // Set information for the transcribed NLP problem
    setInfoNLPG();
    
    // =========================================================================
    // INPUT VALIDATION
    // =========================================================================
    // Verify that input states have correct dimensions and valid quaternions
    
    if (initial_state.size() != StateIndex::TOTAL || 
        final_state.size() != StateIndex::TOTAL) {
        throw std::invalid_argument(
            "State vectors must have 7 components: [q0, q1, q2, q3, wx, wy, wz]"
        );
    }
    
    if (!isQuaternionValid(initial_state) || !isQuaternionValid(final_state)) {
        throw std::invalid_argument(
            "Quaternion components must be normalized: ||q|| = 1"
        );
    }
    
    // =========================================================================
    // PROBLEM BOUNDS DEFINITION
    // =========================================================================
    // Define bounds on states, controls, time, and path constraints. These
    // bounds define the feasible region for the optimization.
    
    // -------------------------------------------------------------------------
    // Time Bounds
    // -------------------------------------------------------------------------
    // Initial time is fixed at zero; final time is optimized within bounds
    const double t0min = 0.0;
    const double t0max = 0.0;
    const double tfmin = T_min;  // Minimum maneuver time (prevents singular solutions)
    const double tfmax = T_max;  // Maximum maneuver time (physical/operational limit)
    
    // -------------------------------------------------------------------------
    // State Bounds During Trajectory
    // -------------------------------------------------------------------------
    // Quaternion components: typically [-1, 1] but may be tighter
    const double qmin = q_min;
    const double qmax = q_max;
    
    // Angular velocity bounds: based on spacecraft dynamics and physical limits
    const double wmin = w_min;  // Maximum rotation rate in any direction (rad/s)
    const double wmax = w_max;
    
    // -------------------------------------------------------------------------
    // Control Bounds
    // -------------------------------------------------------------------------
    // Torque limits: determined by actuator capabilities (reaction wheels, thrusters)
    const double torquemin = tau_min;  // Maximum available torque (N⋅m)
    const double torquemax = tau_max;
    
    // -------------------------------------------------------------------------
    // Path Constraint Bounds
    // -------------------------------------------------------------------------
    // Quaternion normalization: strictly enforced as equality constraint
    const double normmin = DefaultValues::QUAT_NORM_EXACT;
    const double normmax = DefaultValues::QUAT_NORM_EXACT;
    
    // -------------------------------------------------------------------------
    // Boundary Condition Setup
    // -------------------------------------------------------------------------
    // Initial and final states are typically fixed (equality constraints) for
    // rest-to-rest maneuvers or specified attitude changes.
    
    // Declare bound arrays for phase 1
    double x0l1[nxG[phase]], x0u1[nxG[phase]];  // Initial state bounds
    double xfl1[nxG[phase]], xfu1[nxG[phase]];  // Final state bounds
    double xl1[nxG[phase]],  xu1[nxG[phase]];   // State bounds during trajectory
    double ul1[nuG[phase]],  uu1[nuG[phase]];   // Control bounds
    double ql1[nqG[phase]],  qu1[nqG[phase]];   // Integral constraint bounds (unused)
    double cl1[ncG[phase]],  cu1[ncG[phase]];   // Path constraint bounds
    double t0l1, t0u1;  // Initial time bounds
    double tfl1, tfu1;  // Final time bounds
    
    // -------------------------------------------------------------------------
    // Initial State Bounds (Equality Constraints)
    // -------------------------------------------------------------------------
    // The spacecraft starts at a specified initial attitude and angular velocity
    copyStateToBounds(x0l1, initial_state);
    copyStateToBounds(x0u1, initial_state);
    
    // -------------------------------------------------------------------------
    // Final State Bounds (Equality Constraints)
    // -------------------------------------------------------------------------
    // The spacecraft must reach the target attitude and angular velocity
    copyStateToBounds(xfl1, final_state);
    copyStateToBounds(xfu1, final_state);
    
    // -------------------------------------------------------------------------
    // State Bounds During Trajectory (Inequality Constraints)
    // -------------------------------------------------------------------------
    // Quaternion components: allowed to vary within physical bounds
    xl1[StateIndex::Q0] = qmin;  xu1[StateIndex::Q0] = qmax;
    xl1[StateIndex::Q1] = qmin;  xu1[StateIndex::Q1] = qmax;
    xl1[StateIndex::Q2] = qmin;  xu1[StateIndex::Q2] = qmax;
    xl1[StateIndex::Q3] = qmin;  xu1[StateIndex::Q3] = qmax;
    
    // Angular velocities: bounded by maximum rotation rates
    xl1[StateIndex::WX] = wmin;  xu1[StateIndex::WX] = wmax;
    xl1[StateIndex::WY] = wmin;  xu1[StateIndex::WY] = wmax;
    xl1[StateIndex::WZ] = wmin;  xu1[StateIndex::WZ] = wmax;
    
    // -------------------------------------------------------------------------
    // Control Bounds (Actuator Limits)
    // -------------------------------------------------------------------------
    // Torques: symmetric bounds based on actuator capabilities
    ul1[ControlIndex::TX] = torquemin;  uu1[ControlIndex::TX] = torquemax;
    ul1[ControlIndex::TY] = torquemin;  uu1[ControlIndex::TY] = torquemax;
    ul1[ControlIndex::TZ] = torquemin;  uu1[ControlIndex::TZ] = torquemax;
    
    // -------------------------------------------------------------------------
    // Path Constraint Bounds (Quaternion Normalization)
    // -------------------------------------------------------------------------
    // Enforce ||q||² = 1 as an equality constraint along the trajectory
    cl1[ConstraintIndex::QUAT_NORM] = normmin;
    cu1[ConstraintIndex::QUAT_NORM] = normmax;
    
    // -------------------------------------------------------------------------
    // Time Bounds
    // -------------------------------------------------------------------------
    t0l1 = t0min;  t0u1 = t0max;  // Initial time fixed at zero
    tfl1 = tfmin;  tfu1 = tfmax;  // Final time optimized within bounds
    
    // Register phase bounds with CGPOPS framework
    setNLPPBG(phase, x0l1, x0u1, xfl1, xfu1, xl1, xu1, ul1, uu1, 
              ql1, qu1, cl1, cu1, t0l1, t0u1, tfl1, tfu1);
    
    // -------------------------------------------------------------------------
    // Whole Problem Bounds (Unused for Single-Phase Problem)
    // -------------------------------------------------------------------------
    double sl[nsG], su[nsG];  // Static parameter bounds (none)
    double bl[nbG], bu[nbG];  // Event constraint bounds (none)
    
    setNLPWBG(sl, su, bl, bu);
    
    // =========================================================================
    // INITIAL GUESS DEFINITION
    // =========================================================================
    // Provide an initial guess for the NLP solver. A good initial guess can
    // significantly improve convergence speed and solution quality.
    //
    // Strategy: Linear interpolation between initial and final states provides
    // a kinematically feasible (though dynamically infeasible) starting point.
    // Controls are guessed to transition from positive to negative, providing
    // a bang-bang-like initial profile.
    
    double x0g1[nxG[phase]], xfg1[nxG[phase]];  // Initial/final state guess
    double u0g1[nuG[phase]], ufg1[nuG[phase]];  // Initial/final control guess
    double qg1[nqG[phase]];  // Integral cost guess (unused)
    double t0g1, tfg1;  // Time guess
    
    // -------------------------------------------------------------------------
    // State Guess: Use Boundary Values
    // -------------------------------------------------------------------------
    // Copy initial and final states as guess endpoints. CGPOPS will interpolate
    // intermediate points along the trajectory.
    copyStateToBounds(x0g1, initial_state);
    copyStateToBounds(xfg1, final_state);
    
    // -------------------------------------------------------------------------
    // Control Guess: Simple Bang-Bang Profile
    // -------------------------------------------------------------------------
    // Initial guess: positive torques at start, negative at end
    // This approximates a bang-bang control profile (common in time-optimal problems)
    u0g1[ControlIndex::TX] =  1.0;  ufg1[ControlIndex::TX] = -1.0;
    u0g1[ControlIndex::TY] =  1.0;  ufg1[ControlIndex::TY] = -1.0;
    u0g1[ControlIndex::TZ] =  1.0;  ufg1[ControlIndex::TZ] = -1.0;
    
    // -------------------------------------------------------------------------
    // Time Guess
    // -------------------------------------------------------------------------
    // Initial time fixed at zero
    t0g1 = 0.0;
    
    // Final time guess: use provided estimate (e.g., from analytical approximation
    // or previous solution). For a 180° rotation, typical times are 1-5 seconds
    // depending on inertia and torque limits.
    tfg1 = T_0;
    
    // Register phase guess with CGPOPS framework
    setNLPPGG(phase, x0g1, xfg1, u0g1, ufg1, qg1, t0g1, tfg1);
    
    // -------------------------------------------------------------------------
    // Whole Problem Guess (Unused for Single-Phase Problem)
    // -------------------------------------------------------------------------
    double sg[nsG];  // Static parameter guess (none)
    
    setNLPWGG(sg);
    
    // =========================================================================
    // OPTIMAL CONTROL PROBLEM FUNCTION REGISTRATION
    // =========================================================================
    // Register function objects that define the optimal control problem:
    // - Objective function (minimize final time)
    // - Path constraints (quaternion normalization)
    // - Differential equations (quaternion kinematics + Euler's equations)
    
    // Objective: minimize final time (Mayer formulation)
    objEqG = new MinTf;
    
    // Path constraint: quaternion normalization ||q||² = 1
    pthEqVecG = {{new QuatNormConstraint}};
    
    // Differential equations: [q̇₀, q̇₁, q̇₂, q̇₃, ω̇x, ω̇y, ω̇z]
    odeEqVecG = {{
        new Q0Dot,   // q̇₀ (quaternion scalar rate)
        new Q1Dot,   // q̇₁ (quaternion x-component rate)
        new Q2Dot,   // q̇₂ (quaternion y-component rate)
        new Q3Dot,   // q̇₃ (quaternion z-component rate)
        new WXDot,   // ω̇x (angular acceleration about x-axis)
        new WYDot,   // ω̇y (angular acceleration about y-axis)
        new WZDot    // ω̇z (angular acceleration about z-axis)
    }};
    
    // =========================================================================
    // SOLVE OPTIMAL CONTROL PROBLEM
    // =========================================================================
    // Call the CGPOPS-IPOPT interface to transcribe the continuous-time optimal
    // control problem into an NLP and solve it using the IPOPT optimizer.
    //
    // The results include:
    // - Optimal state trajectory (time history of quaternion and angular velocity)
    // - Optimal control trajectory (time history of applied torques)
    // - Time grid points (LGR collocation nodes)
    // - Minimum time achieved (objective function value)
    // - Lagrange multipliers (for constraint analysis)
    
    CGPOPS_IPOPT_caller(cgpopsResults);
}