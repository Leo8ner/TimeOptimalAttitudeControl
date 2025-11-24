/**
 * @file cgpops_main.hpp
 * @brief Main interface for CGPOPS time-optimal spacecraft attitude control problem
 * 
 * OVERVIEW:
 * =========
 * This header provides the primary interface function for configuring and solving
 * time-optimal spacecraft attitude maneuver problems using the CGPOPS (C++ General
 * Pseudospectral Optimal Control Software) framework. The implementation uses
 * Legendre-Gauss-Radau (LGR) pseudospectral collocation to transcribe continuous-time
 * optimal control problems into nonlinear programming (NLP) problems, which are
 * then solved using the IPOPT optimizer.
 * 
 * PROBLEM FORMULATION:
 * ====================
 * The interface handles time-optimal attitude reorientation problems with:
 * - 7-dimensional state: quaternion (4) + angular velocity (3)
 * - 3-dimensional control: torque components
 * - Path constraint: quaternion normalization ||q|| = 1
 * - Objective: minimize maneuver time
 * - Dynamics: quaternion kinematics + Euler's rotational equations
 * 
 * 
 * EXTERNAL DEPENDENCIES:
 * ======================
 * Before calling cgpops_go(), the following global variables must be defined:
 * 
 * Time Bounds:
 *   - T_min, T_max : Bounds on final time (seconds)
 *   - T_0          : Initial guess for final time (seconds)
 * 
 * State Bounds:
 *   - q_min, q_max : Bounds on quaternion components (typically [-1, 1])
 *   - w_min, w_max : Bounds on angular velocity (rad/s)
 * 
 * Control Bounds:
 *   - tau_min, tau_max : Bounds on applied torque (N⋅m)
 * 
 * Physical Parameters:
 *   - i_x, i_y, i_z : Principal moments of inertia (kg⋅m²)
 * 
 * Mesh Configuration:
 *   - numintervalsG : Number of mesh intervals for LGR collocation
 *   - initcolptsG   : Initial number of collocation points per interval
 * 
 * FRAMEWORK COMPONENTS:
 * =====================
 * The CGPOPS framework consists of several interconnected components:
 * - nlpGlobVarExt.hpp  : Global NLP variable declarations
 * - cgpopsAuxExt.hpp   : Auxiliary function declarations
 * - cgpopsAuxDec.hpp   : Auxiliary class declarations
 * - cgpops_gov.hpp     : Governing equations (dynamics, constraints, objective)
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 * 
 * @see cgpops_go() Main interface function
 * @see cgpops_gov.hpp Dynamics and constraint equations
 * 
 * @author Yunus M. Agamawi, Anil Vithala Rao
 */

#ifndef __CGPOPS_MAIN_HPP__
#define __CGPOPS_MAIN_HPP__

// =============================================================================
// REQUIRED FRAMEWORK HEADERS
// =============================================================================
// These headers provide the core CGPOPS framework functionality required for
// optimal control problem setup and solution.

#include "nlpGlobVarExt.hpp"        ///< Global NLP variables and types (doubleMat, etc.)
#include <cgpops/cgpopsAuxExt.hpp>  ///< CGPOPS auxiliary function declarations
#include <cgpops/cgpopsAuxDec.hpp>  ///< CGPOPS auxiliary class declarations
#include <cgpops/cgpops_gov.hpp>    ///< Problem-specific governing equations

// =============================================================================
// STANDARD LIBRARY HEADERS
// =============================================================================

#include <vector>  ///< STL vector for state representation

// =============================================================================
// MAIN INTERFACE FUNCTION
// =============================================================================

/**
 * @brief Configures and solves time-optimal spacecraft attitude maneuver problem
 * 
 * This function sets up a complete optimal control problem for minimum-time
 * spacecraft attitude reorientation using quaternion representation. It configures
 * the problem dimensions, bounds, initial guess, collocation mesh, and governing
 * equations, then solves the resulting NLP using IPOPT through the CGPOPS framework.
 * 
 * PROBLEM CHARACTERISTICS:
 * ------------------------
 * - Formulation: Continuous-time optimal control (Bolza problem, Mayer form)
 * - Transcription: Legendre-Gauss-Radau (LGR) pseudospectral collocation
 * - NLP Solver: IPOPT (Interior Point OPTimizer)
 * - State dimension: 7 (quaternion q0,q1,q2,q3 + angular velocity wx,wy,wz)
 * - Control dimension: 3 (torque components tx,ty,tz)
 * - Path constraints: 1 (quaternion normalization ||q||² = 1)
 * - Objective: Minimize final time tf
 * 
 * STATE VECTOR FORMAT:
 * --------------------
 * The 7-element state vector must be ordered as:
 *   x = [q0, q1, q2, q3, wx, wy, wz]
 * where:
 *   - q0, q1, q2, q3 : Quaternion components (scalar-first convention)
 *                      Must satisfy ||q|| = 1
 *   - wx, wy, wz     : Angular velocity in body frame (rad/s)
 * 
 * QUATERNION CONVENTION:
 * ----------------------
 * This implementation uses the scalar-first quaternion convention:
 *   q = [q0, q1, q2, q3] = [w, x, y, z]
 * 
 * The quaternion represents the rotation from inertial frame to body frame.
 * 
 * REQUIRED EXTERNAL PARAMETERS:
 * ------------------------------
 * The following global variables must be defined before calling this function:
 * 
 * Time Parameters:
 *   @pre T_min > 0        Minimum final time (prevents singular solutions)
 *   @pre T_max > T_min    Maximum final time (physical/operational constraint)
 *   @pre T_0 > 0          Initial time guess (aids convergence)
 * 
 * State Bounds:
 *   @pre -1 ≤ q_min < q_max ≤ 1   Quaternion component bounds
 *   @pre w_min < 0 < w_max         Angular velocity bounds (rad/s)
 * 
 * Control Bounds:
 *   @pre tau_min < 0 < tau_max     Torque bounds (N⋅m, symmetric recommended)
 * 
 * Physical Properties:
 *   @pre i_x, i_y, i_z > 0         Principal moments of inertia (kg⋅m²)
 * 
 * Mesh Configuration:
 *   @pre numintervalsG ≥ 1         Number of mesh intervals (typically 5-20)
 *   @pre initcolptsG ≥ 2           Collocation points per interval (typically 3-5)
 * 
 * @param[out] cgpopsResults Output matrix containing optimization results.
 *   Structure (implementation-dependent, refer to CGPOPS documentation):
 *   - Optimal state trajectory at collocation points
 *   - Optimal control trajectory at collocation points
 *   - Time values at collocation points
 *   - Objective function value (minimum time achieved)
 *   - Lagrange multipliers for constraints
 *   - Solver statistics and convergence information
 * 
 * @param[in] initial_state Initial state vector [q0, q1, q2, q3, wx, wy, wz]
 *   Specifies the spacecraft's starting attitude and angular velocity.
 *   The quaternion components [q0, q1, q2, q3] must form a unit quaternion:
 *   q0² + q1² + q2² + q3² = 1
 * 
 * @param[in] final_state Target state vector [q0, q1, q2, q3, wx, wy, wz]
 *   Specifies the desired final attitude and angular velocity.
 *   The quaternion components [q0, q1, q2, q3] must form a unit quaternion:
 *   q0² + q1² + q2² + q3² = 1
 *   For rest-to-rest maneuvers, final angular velocities should be zero.
 * 
 * @pre initial_state.size() == 7
 * @pre final_state.size() == 7
 * @pre ||initial_state[0:3]|| = 1  (quaternion normalized)
 * @pre ||final_state[0:3]|| = 1    (quaternion normalized)
 * @pre All required external parameters are properly initialized
 * 
 * @post cgpopsResults contains the optimal trajectory if solver converged
 * @post Returns without throwing if problem setup is valid
 * 
 * @throws std::invalid_argument if state vector dimensions are incorrect
 * @throws std::invalid_argument if quaternions are not normalized
 * @throws May propagate exceptions from CGPOPS/IPOPT framework
 * 
 * @note This function modifies global CGPOPS state. It is not thread-safe.
 * @note Convergence depends on problem scaling, bounds, and initial guess quality.
 * @note For difficult problems, consider:
 *       - Tightening time bounds (T_min, T_max)
 *       - Improving initial guess (T_0)
 *       - Increasing mesh resolution (numintervalsG, initcolptsG)
 *       - Relaxing state/control bounds if physically justified
 * 
 * @see cgpops_gov.hpp for governing equation implementations
 * @see nlpGlobVarExt.hpp for global variable declarations
 * 
 * @example
 * // Rest-to-rest 180° rotation about z-axis
 * std::vector<double> x0 = {1, 0, 0, 0, 0, 0, 0};  // Identity quaternion
 * std::vector<double> xf = {0, 0, 0, 1, 0, 0, 0};  // 180° about z
 * doubleMat results;
 * cgpops_go(results, x0, xf);
 * 
 * @example
 * // General rotation with initial angular velocity
 * std::vector<double> x0 = {0.924, 0.383, 0, 0, 0.1, 0, 0};  // 45° about x
 * std::vector<double> xf = {0.707, 0, 0.707, 0, 0, 0, 0};    // 90° about y
 * doubleMat results;
 * cgpops_go(results, x0, xf);
 */
void cgpops_go(doubleMat& cgpopsResults, 
               const std::vector<double>& initial_state, 
               const std::vector<double>& final_state);

// =============================================================================
// END OF HEADER
// =============================================================================

#endif  // __CGPOPS_MAIN_HPP__