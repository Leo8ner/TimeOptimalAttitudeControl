/**
 * @file helper_functions.h
 * @brief Comprehensive utility library for spacecraft attitude control optimization
 * 
 * OVERVIEW:
 * =========
 * This header provides a complete suite of utility functions for solving time-optimal
 * spacecraft attitude maneuver problems using both CasADi-based direct transcription
 * methods and CGPOPS pseudospectral collocation. The library bridges multiple optimization
 * frameworks and provides essential conversions, I/O operations, and mathematical utilities.
 * 
 * LIBRARY ARCHITECTURE:
 * =====================
 * The library is organized into seven functional categories:
 * 
 * 1. QUATERNION MATHEMATICS
 *    - Unit quaternion operations (multiplication, conjugation, normalization)
 *    - Conversions: Euler angles ↔ quaternions
 *    - Angle unwrapping and normalization
 * 
 * 2. STATE VECTOR PARSING & CONVERSION
 *    - Command-line argument parsing
 *    - Euler angle + angular velocity → quaternion state vectors
 *    - Dual implementations: CasADi DM and std::vector
 * 
 * 3. TRAJECTORY I/O
 *    - CSV import/export for trajectories
 *    - MATLAB .m file parsing (CGPOPS output)
 *    - Initial guess extraction from existing solutions
 * 
 * 4. OPTIMIZATION UTILITIES
 *    - Result post-processing and display
 *    - Solution value extraction
 *    - Status checking and validation
 * 
 * 5. SOLVER INTERFACE
 *    - Dynamic loading of compiled CasADi solvers
 *    - Solver log management
 *    - Output redirection utilities
 * 
 * 6. RESULT FILE MANAGEMENT
 *    - Crash-recoverable result file writing
 *    - Appending new results to existing files
 *    - Progress reporting and time estimation
 * 
 * 7. NUMERICAL INTEGRATION
 *    - RK4 (Runge-Kutta 4th order) integration
 *    - Skew-symmetric matrix construction
 *    - Discrete-time dynamics
 * 
 * 8. SYSTEM UTILITIES
 *    - File I/O helpers
 *    - Output redirection
 *    - Numerical formatting
 * 
 * NAMING CONVENTIONS:
 * ===================
 * 
 * Function Naming Patterns:
 * - Standard functions: Use CasADi DM type for symbolic/numeric computation
 * - V-prefixed functions: Use std::vector<double> for C++ standard library compatibility
 *   Examples: euler2quat() uses DM, Veuler2quat() uses vector<double>
 * 
 * Variable Naming:
 * - X, U, T: State trajectory, control sequence, time (uppercase for matrices)
 * - x, u, t: Current state, control, time (lowercase for vectors/scalars)
 * - q: Quaternion [q0, q1, q2, q3] = [w, x, y, z] (scalar-first convention)
 * - euler/angles: [roll, pitch, yaw] = [φ, θ, ψ] (ZYX Euler sequence)
 * - w/omega: Angular velocity [ωx, ωy, ωz] in body frame (rad/s)
 * 
 * QUATERNION CONVENTION:
 * ======================
 * All functions use the scalar-first quaternion convention:
 *   q = [q0, q1, q2, q3] = [w, x, y, z]
 * where:
 *   - q0 (w): Scalar component
 *   - q1, q2, q3 (x, y, z): Vector components
 *   - Constraint: ||q|| = 1 (unit quaternion)
 * 
 * Quaternions represent rotation from inertial frame to body frame.
 * 
 * EULER ANGLE CONVENTION:
 * =======================
 * ZYX (3-2-1) Euler sequence used throughout:
 *   1. Yaw (ψ) about Z-axis
 *   2. Pitch (θ) about Y-axis  
 *   3. Roll (φ) about X-axis
 * 
 * This is the most common convention in aerospace applications.
 * 
 * COORDINATE FRAMES:
 * ==================
 * - Inertial Frame: Fixed reference frame (typically ECI)
 * - Body Frame: Spacecraft-fixed frame with origin at center of mass
 *   - X-axis: Typically forward/velocity direction
 *   - Y-axis: Typically cross-track/lateral
 *   - Z-axis: Typically nadir/perpendicular to orbit plane
 * 
 * DEPENDENCIES:
 * =============
 * External Libraries:
 * - CasADi: Symbolic computation and optimization framework
 * - TOAC: Time-Optimal Attitude Control library (spacecraft parameters)
 * - CGPOPS: C++ General Pseudospectral Optimal Control Software
 * 
 * Standard Library:
 * - C++17 filesystem support required
 * - POSIX I/O for output redirection (Unix/Linux/macOS)
 * 
 * TYPICAL USAGE WORKFLOW:
 * =======================
 * @code
 * // 1. Parse input states
 * auto [X_0, X_f, angles_0, angles_f] = parseInput(
 *     "0,0,180,0,0,0",  // Initial: 180° yaw rotation, rest
 *     "0,0,0,0,0,0"      // Final: identity, rest
 * );
 * 
 * // 2. Load solver and solve optimization problem
 * auto solver = get_solver();
 * casadi::DMDict inputs = {{"x0", X_0}, {"xf", X_f}};
 * casadi::DMDict result = solver(inputs);
 * 
 * // 3. Process results
 * processResults(result, angles_0, angles_f);
 * 
 * // 4. Export trajectory for visualization
 * exportTrajectory(result.at("X"), result.at("U"), result.at("T"), 
 *                  result.at("dt"), angles_0, angles_f, "trajectory.csv");
 * @endcode
 * 
 * @note All angle inputs and outputs use DEGREES unless explicitly stated otherwise
 * @note Angular velocities always use RADIANS/SECOND
 * @note Quaternions are always normalized to unit length
 * 
 * @see <toac/symmetric_spacecraft.h> for spacecraft physical parameters
 * @see <cgpops/cgpopsAuxExt.hpp> for CGPOPS global variables
 * 
 * @author Leonardo Eitner
 * @version 1.0
 * @date 2025
 */

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

// =============================================================================
// STANDARD LIBRARY HEADERS
// =============================================================================
// Core C++ standard library includes for basic functionality

#include <string>       ///< String manipulation and representation
#include <vector>       ///< Dynamic array container (std::vector variants of functions)
#include <tuple>        ///< Multiple return values (e.g., parseInput)
#include <map>          ///< Associative containers for parameter storage
#include <sstream>      ///< String stream for parsing CSV and formatted data
#include <fstream>      ///< File I/O operations (CSV, MATLAB file parsing)
#include <iostream>     ///< Console I/O for logging and display
#include <stdexcept>    ///< Exception handling (invalid_argument, runtime_error)
#include <filesystem>   ///< C++17 filesystem operations (requires C++17 or later)
#include <regex>        ///< Regular expressions for MATLAB file parsing
#include <cmath>        ///< Mathematical functions (sin, cos, atan2, sqrt, etc.)
#include <cstdio>       ///< C-style I/O (FILE*, fprintf, fflush)
#include <iomanip>      ///< I/O manipulators (setprecision, fixed)
#include <cstring>      ///< C string functions (strerror for errno conversion)

// =============================================================================
// POSIX SYSTEM HEADERS
// =============================================================================
// Unix/Linux/macOS system headers for low-level I/O operations

#include <unistd.h>     ///< POSIX operating system API (dup, dup2, fsync)
#include <fcntl.h>      ///< File control options (open, O_WRONLY, O_CREAT)

// =============================================================================
// EXTERNAL LIBRARY HEADERS
// =============================================================================

/**
 * @brief CasADi symbolic framework
 * 
 * Provides:
 * - Symbolic computation (MX, SX types)
 * - Numerical computation (DM type)
 * - Automatic differentiation
 * - Optimization problem formulation
 * - Dynamic library loading (Function::load)
 * - Solver interfaces
 * 
 * Used throughout for optimization and trajectory computation.
 */
#include <casadi/casadi.hpp>

/**
 * @brief TOAC spacecraft parameter library
 * 
 * Provides spacecraft-specific physical parameters:
 * - Moments of inertia (i_x, i_y, i_z)
 * - Control bounds (tau_min, tau_max)
 * - State bounds (w_min, w_max, q_min, q_max)
 * - Time parameters (T_0, T_min, T_max)
 * - Problem dimensions (n_states, n_controls)
 * - Mathematical constants (PI, DEG, RAD)
 * 
 * These parameters define the spacecraft dynamics and constraints.
 */
#include <toac/symmetric_spacecraft.h>

/**
 * @brief CGPOPS auxiliary variables
 * 
 * Provides access to CGPOPS framework global variables through TOAC library.
 * Required for interfacing with CGPOPS pseudospectral optimization methods.
 */
#include <cgpops/cgpopsAuxExt.hpp>

// =============================================================================
// GLOBAL VARIABLES FOR OUTPUT REDIRECTION
// =============================================================================
// These variables maintain state for output redirection functionality.
// They are defined in helper_functions.cpp and declared here as extern.

/**
 * @brief Original stdout file descriptor (saved before redirection)
 * 
 * Stores the original file descriptor for stdout (typically 1) before
 * redirection occurs. Used by restore_output_to_console() to restore
 * normal console output after redirection.
 * 
 * @note Initialized to -1 to indicate "not saved"
 * @note Modified by redirect_output_to_file() and restore_output_to_console()
 */
extern int original_stdout_fd;

/**
 * @brief Original stderr file descriptor (saved before redirection)
 * 
 * Stores the original file descriptor for stderr (typically 2) before
 * redirection occurs. Used by restore_output_to_console() to restore
 * normal error output after redirection.
 * 
 * @note Initialized to -1 to indicate "not saved"
 * @note Modified by redirect_output_to_file() and restore_output_to_console()
 */
extern int original_stderr_fd;

/**
 * @brief Log file descriptor for redirected output
 * 
 * File descriptor for the log file to which stdout/stderr are redirected.
 * Valid when output is redirected, -1 otherwise.
 * 
 * @note Initialized to -1 to indicate "no redirection active"
 * @note Opened by redirect_output_to_file(), closed by restore_output_to_console()
 */
extern int log_fd;

// =============================================================================
// QUATERNION MATHEMATICS
// =============================================================================
// Functions for quaternion operations and conversions between attitude
// representations. All functions maintain unit quaternion normalization
// and use scalar-first convention: q = [q0, q1, q2, q3] = [w, x, y, z].

/**
 * @brief Convert Euler angles to unit quaternion (CasADi DM version)
 * 
 * Converts ZYX Euler angle sequence to quaternion representation using
 * the aerospace standard convention (yaw-pitch-roll).
 * 
 * MATHEMATICAL FORMULATION:
 *   q = qz(ψ) ⊗ qy(θ) ⊗ qx(φ)
 * 
 * where ⊗ denotes quaternion multiplication and qi(α) represents a
 * rotation of α radians about axis i.
 * 
 * @param phi Roll angle (rotation about X-axis, radians)
 *            Range: [-π, π] for full rotation representation
 * @param theta Pitch angle (rotation about Y-axis, radians)
 *              Range: [-π/2, π/2] to avoid gimbal lock
 * @param psi Yaw angle (rotation about Z-axis, radians)
 *            Range: [-π, π] for full rotation representation
 * 
 * @return Unit quaternion as CasADi DM object [4 x 1]
 *         Format: [q0, q1, q2, q3] = [w, x, y, z] (scalar-first)
 *         Property: ||q|| = 1 (normalized by construction)
 * 
 * @note Input angles must be in RADIANS (convert from degrees using * M_PI/180)
 * @note Output quaternion is normalized to unit length
 * @note Gimbal lock occurs when theta = ±π/2, resulting in loss of one degree of freedom
 * @note Function handles gimbal lock case but may produce non-unique quaternion
 * 
 * @see Veuler2quat() for std::vector<double> version
 * @see quat2euler() for inverse transformation
 * 
 * @par Example:
 * @code
 * // 90° rotation about Z-axis (yaw)
 * casadi::DM q = euler2quat(0.0, 0.0, M_PI/2);
 * // Result: q ≈ [0.7071, 0, 0, 0.7071]
 * @endcode
 */
casadi::DM euler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Convert Euler angles to unit quaternion (std::vector version)
 * 
 * Identical mathematical operation to euler2quat() but uses standard C++
 * vectors instead of CasADi DM for compatibility with non-CasADi code.
 * 
 * @param phi Roll angle (radians)
 * @param theta Pitch angle (radians)
 * @param psi Yaw angle (radians)
 * 
 * @return Unit quaternion as std::vector<double> with 4 elements
 *         Format: [q0, q1, q2, q3] = [w, x, y, z]
 *         Property: ||q|| = 1
 * 
 * @note All angles in RADIANS
 * @see euler2quat() for CasADi DM version and detailed documentation
 * @see VparseStateVector() uses this function
 */
std::vector<double> Veuler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Convert quaternion to Euler angles with unwrapping (CasADi DM version)
 * 
 * Converts unit quaternion to ZYX Euler angles while maintaining continuity
 * with previous Euler angles to avoid discontinuities due to angle wrapping.
 * 
 * ANGLE UNWRAPPING:
 * The unwrapping process ensures smooth angle trajectories by detecting and
 * correcting 360° jumps that occur when angles wrap around ±180°. This is
 * critical for trajectory visualization and analysis.
 * 
 * ALGORITHM:
 * 1. Convert quaternion to Euler angles using atan2
 * 2. For each angle, check if difference from previous > 180° or < -180°
 * 3. Adjust by ±360° to maintain continuity
 * 
 * @param euler_prev Previous Euler angles [roll, pitch, yaw] in DEGREES
 *                   Used as reference for unwrapping to maintain continuity
 *                   Format: CasADi DM [3 x 1]
 * @param q Current quaternion [q0, q1, q2, q3] to convert
 *          Should be unit quaternion: ||q|| ≈ 1
 *          Format: CasADi DM [4 x 1]
 * 
 * @return Euler angles [roll, pitch, yaw] in DEGREES
 *         Format: CasADi DM [3 x 1]
 *         Unwrapped to maintain continuity with euler_prev
 * 
 * @note Input quaternion should be normalized (||q|| = 1)
 * @note Output angles are in DEGREES (not radians)
 * @note Unwrapping prevents discontinuous jumps in angle trajectories
 * @note For first conversion in a sequence, use [0, 0, 0] as euler_prev
 * @note Gimbal lock (pitch = ±90°) may cause numerical issues
 * 
 * @see unwrapAngle() for single angle unwrapping algorithm
 * @see euler2quat() for inverse transformation
 * 
 * @par Example:
 * @code
 * casadi::DM q = casadi::DM::vertcat({0.7071, 0, 0, 0.7071});  // 90° yaw
 * casadi::DM prev = casadi::DM::vertcat({0, 0, 0});            // Start from zero
 * casadi::DM euler = quat2euler(prev, q);
 * // Result: euler ≈ [0, 0, 90] (degrees)
 * @endcode
 */
casadi::DM quat2euler(const casadi::DM& euler_prev, const casadi::DM& q);

/**
 * @brief Unwrap angle to maintain continuity with previous value
 * 
 * Adjusts current angle by adding or subtracting 360° to minimize the
 * difference from the previous angle, ensuring smooth trajectories without
 * discontinuous jumps at the ±180° boundary.
 * 
 * ALGORITHM:
 *   1. Compute difference: diff = current - previous
 *   2. If diff > 180°, subtract 360° from current
 *   3. If diff < -180°, add 360° to current
 *   4. Return adjusted angle
 * 
 * @param current Current angle value in degrees
 * @param previous Previous angle value in degrees (reference for continuity)
 * 
 * @return Unwrapped angle in degrees, adjusted to be continuous with previous
 * 
 * @note Both inputs and output are in DEGREES
 * @note Does not modify the physical angle, only its representation
 * @note Essential for plotting smooth angle trajectories
 * @note Multiple unwrapping calls create continuous multi-revolution trajectories
 * 
 * @par Example:
 * @code
 * double prev = 175.0;  // degrees
 * double curr = -175.0; // degrees (jumped across ±180° boundary)
 * double unwrapped = unwrapAngle(curr, prev);
 * // Result: unwrapped = 185.0 (continuous with prev)
 * @endcode
 */
double unwrapAngle(double current, double previous);

/**
 * @brief Normalize angle to range [-180, 180] degrees
 * 
 * Maps any angle to its equivalent representation in the principal range
 * [-180°, 180°] using modular arithmetic. This ensures a unique canonical
 * representation for each physical angle.
 * 
 * ALGORITHM:
 *   angle_normalized = fmod(angle + 180, 360) - 180
 *   with correction for negative modulo results
 * 
 * @param angle Input angle in degrees (can be any real value)
 * 
 * @return Normalized angle in range [-180, 180] degrees
 * 
 * @note Input and output are both in DEGREES
 * @note Result is mathematically equivalent to input (same physical angle)
 * @note Useful for comparing angles and enforcing canonical representation
 * @note Different from unwrapAngle() which maintains continuity
 * 
 * @par Example:
 * @code
 * double angle1 = normalizeAngle(370.0);   // Result: 10.0
 * double angle2 = normalizeAngle(-190.0);  // Result: 170.0
 * double angle3 = normalizeAngle(180.0);   // Result: -180.0 or 180.0 (boundary case)
 * @endcode
 */
double normalizeAngle(double angle);

/**
 * @brief Multiply two quaternions (CasADi DM version)
 * 
 * Performs Hamilton quaternion multiplication: q_result = q1 ⊗ q2
 * This operation represents composition of rotations.
 * 
 * MATHEMATICAL DEFINITION:
 * Given q1 = [w1, x1, y1, z1] and q2 = [w2, x2, y2, z2]:
 *   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
 *   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
 *   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
 *   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
 * 
 * GEOMETRIC INTERPRETATION:
 * Quaternion multiplication represents composition of rotations.
 * q1 ⊗ q2 applies rotation q2 first, then rotation q1.
 * 
 * @param q1 First quaternion [q0, q1, q2, q3] as CasADi DM [4 x 1]
 * @param q2 Second quaternion [q0, q1, q2, q3] as CasADi DM [4 x 1]
 * 
 * @return Product quaternion q1 ⊗ q2 as CasADi DM [4 x 1]
 * 
 * @note Quaternion multiplication is NOT commutative: q1⊗q2 ≠ q2⊗q1 (generally)
 * @note If both inputs are unit quaternions, output is unit quaternion
 * @note Order matters: represents sequential rotations from right to left
 * @note For non-unit quaternions, result norm is product of input norms
 * 
 * @see Vquat_mul() for std::vector<double> version
 * @see quat_conj() for quaternion conjugation (needed for inverse)
 */
casadi::DM quat_mul(const casadi::DM& q1, const casadi::DM& q2);

/**
 * @brief Multiply two quaternions (std::vector version)
 * 
 * Standard C++ vector implementation of Hamilton quaternion multiplication.
 * Identical mathematics to quat_mul() but for std::vector<double> type.
 * 
 * @param q1 First quaternion as std::vector<double> (4 elements)
 * @param q2 Second quaternion as std::vector<double> (4 elements)
 * 
 * @return Product quaternion as std::vector<double> (4 elements)
 * 
 * @see quat_mul() for CasADi DM version and detailed documentation
 */
std::vector<double> Vquat_mul(const std::vector<double>& q1, const std::vector<double>& q2);

/**
 * @brief Compute quaternion conjugate (CasADi DM version)
 * 
 * Returns the conjugate of a quaternion, which represents the inverse rotation
 * for unit quaternions.
 * 
 * MATHEMATICAL DEFINITION:
 * Given q = [w, x, y, z], conjugate is q* = [w, -x, -y, -z]
 * 
 * GEOMETRIC INTERPRETATION:
 * For unit quaternions (||q|| = 1):
 *   - q* represents the inverse rotation
 *   - q ⊗ q* = [1, 0, 0, 0] (identity quaternion)
 *   - Conjugate reverses the rotation direction
 * 
 * @param q Input quaternion [q0, q1, q2, q3] as CasADi DM [4 x 1]
 * 
 * @return Conjugate quaternion [q0, -q1, -q2, -q3] as CasADi DM [4 x 1]
 * 
 * @note For unit quaternions: q* = q^(-1) (conjugate equals inverse)
 * @note For non-unit quaternions: q^(-1) = q* / ||q||²
 * @note Conjugate preserves quaternion norm: ||q*|| = ||q||
 * @note Used for rotating vectors: v' = q ⊗ [0, v] ⊗ q*
 * 
 * @par Example:
 * @code
 * casadi::DM q = casadi::DM::vertcat({0.7071, 0, 0, 0.7071});  // 90° yaw
 * casadi::DM q_conj = quat_conj(q);
 * // Result: q_conj = [0.7071, 0, 0, -0.7071] (inverse rotation)
 * @endcode
 */
casadi::DM quat_conj(const casadi::DM& q);

/**
 * @brief Normalize quaternion to unit length (std::vector version)
 * 
 * Scales quaternion to unit norm while preserving direction. Essential for
 * maintaining valid rotation representation after numerical operations that
 * may introduce rounding errors.
 * 
 * ALGORITHM:
 *   norm = sqrt(q0² + q1² + q2² + q3²)
 *   q_normalized = [q0/norm, q1/norm, q2/norm, q3/norm]
 * 
 * @param q Input quaternion as std::vector<double> (4 elements)
 *          Can have any non-zero norm
 * 
 * @return Normalized unit quaternion as std::vector<double> (4 elements)
 *         Property: ||q_out|| = 1 (within machine precision)
 * 
 * @pre Input quaternion must be non-zero (||q|| > 0)
 * @throw May produce undefined behavior if ||q|| ≈ 0 (division by near-zero)
 * 
 * @note Always normalize quaternions after arithmetic operations
 * @note Numerical integration can cause quaternion drift from unit norm
 * @note Normalization is computationally inexpensive (one sqrt, four divides)
 * @note For q = [0, 0, 0, 0], behavior is undefined (no meaningful normalization)
 * 
 * @par Example:
 * @code
 * std::vector<double> q = {1.5, 0.0, 0.0, 0.0};  // Non-unit quaternion
 * std::vector<double> q_norm = normalize(q);
 * // Result: q_norm = [1.0, 0.0, 0.0, 0.0] (identity rotation)
 * @endcode
 */
std::vector<double> normalize(const std::vector<double>& q);

/**
 * @brief Compute dot product of two quaternions (std::vector version)
 * 
 * Computes the standard Euclidean inner product of two quaternions treated
 * as 4-dimensional vectors.
 * 
 * MATHEMATICAL DEFINITION:
 *   dot(q1, q2) = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
 * 
 * GEOMETRIC INTERPRETATION:
 * For unit quaternions:
 *   - dot(q1, q2) = cos(θ/2), where θ is rotation angle between orientations
 *   - Values near 1: orientations are similar (small rotation between them)
 *   - Values near 0: orientations differ by ~90° rotation
 *   - Values near -1: orientations are opposite (~180° apart)
 * 
 * @param q1 First quaternion as std::vector<double> (4 elements)
 * @param q2 Second quaternion as std::vector<double> (4 elements)
 * 
 * @return Dot product (scalar value)
 *         Range: [-||q1||·||q2||, ||q1||·||q2||]
 *         For unit quaternions: [-1, 1]
 * 
 * @note Used for quaternion similarity measurement
 * @note Can compute quaternion norm: ||q|| = sqrt(dot(q,q))
 * @note Commutative: dot(q1, q2) = dot(q2, q1)
 * @note Bilinear operation
 * 
 * @par Example:
 * @code
 * std::vector<double> q1 = {1, 0, 0, 0};        // Identity rotation
 * std::vector<double> q2 = {0.7071, 0, 0, 0.7071};  // 90° rotation
 * double similarity = dot(q1, q2);
 * // Result: similarity ≈ 0.7071 = cos(45°)
 * @endcode
 */
double dot(const std::vector<double>& q1, const std::vector<double>& q2);

// =============================================================================
// STATE VECTOR PARSING & CONVERSION
// =============================================================================
// Functions for parsing command-line inputs and converting between different
// attitude representations (Euler angles ↔ quaternions) with angular velocities.

/**
 * @brief Parse Euler angles and angular velocities into complete state vectors
 * 
 * Converts user-friendly Euler angle representation to quaternion-based state
 * vectors required by optimization algorithms. Handles both CasADi DM format
 * and preserves original Euler angles for reference and display.
 * 
 * INPUT FORMAT:
 * Both initial_state and final_state are comma-separated strings:
 *   "roll,pitch,yaw,omega_x,omega_y,omega_z"
 * where:
 *   - roll, pitch, yaw: Euler angles in DEGREES
 *   - omega_x, omega_y, omega_z: Angular velocities in DEGREES/SECOND
 * 
 * OUTPUT STATE VECTOR FORMAT:
 *   X = [q0, q1, q2, q3, ωx, ωy, ωz]ᵀ (7 elements)
 * where:
 *   - q0-q3: Unit quaternion components (dimensionless, normalized)
 *   - ωx-ωz: Angular velocities in RADIANS/SECOND (converted from input)
 * 
 * @param initial_state Comma-separated string with 6 values
 *                      Format: "roll,pitch,yaw,wx,wy,wz" (degrees, deg/s)
 *                      Example: "0,0,180,0,0,0" (180° yaw, rest-to-rest)
 * 
 * @param final_state Comma-separated string with 6 values
 *                    Format: "roll,pitch,yaw,wx,wy,wz" (degrees, deg/s)
 *                    Example: "0,0,0,0,0,0" (identity, rest-to-rest)
 * 
 * @return std::tuple<casadi::DM, casadi::DM, casadi::DM, casadi::DM> containing:
 *         - [0] X_0: Initial state vector [7 x 1] (quaternion + rad/s)
 *         - [1] X_f: Final state vector [7 x 1] (quaternion + rad/s)
 *         - [2] angles_0: Initial Euler angles [3 x 1] (degrees, for display)
 *         - [3] angles_f: Final Euler angles [3 x 1] (degrees, for display)
 * 
 * @throw std::invalid_argument if input strings are malformed
 * @throw std::invalid_argument if wrong number of comma-separated values
 * @throw std::invalid_argument if values cannot be parsed as numbers
 * 
 * @note Angular velocities are converted: degrees/second → radians/second
 * @note Output quaternions are automatically normalized to unit length
 * @note Euler angles are preserved separately for result display and CSV headers
 * @note Whitespace around values is NOT trimmed - avoid spaces in input
 * 
 * @see VparseStateVector() for std::vector<double> version (CGPOPS interface)
 * 
 * @par Example:
 * @code
 * auto [X_0, X_f, angles_0, angles_f] = parseInput(
 *     "0,0,90,0,0,0",   // 90° yaw, at rest
 *     "45,0,0,0,0,0"    // 45° roll, at rest
 * );
 * // X_0: [0.7071, 0, 0, 0.7071, 0, 0, 0]  (quaternion + zero angular velocity)
 * // X_f: [0.9239, 0.3827, 0, 0, 0, 0, 0]  (quaternion + zero angular velocity)
 * // angles_0: [0, 0, 90] (degrees, for display)
 * // angles_f: [45, 0, 0] (degrees, for display)
 * @endcode
 */
std::tuple<casadi::DM, casadi::DM, casadi::DM, casadi::DM>
parseInput(const std::string& initial_state, const std::string& final_state);

/**
 * @brief Parse state vectors into quaternion format (std::vector version)
 * 
 * Standard C++ vector implementation of state parsing. Converts Euler angles
 * and angular velocities to quaternion-based state representation compatible
 * with CGPOPS interface which requires std::vector types.
 * 
 * @param initial_state Comma-separated string: "roll,pitch,yaw,wx,wy,wz" (degrees, deg/s)
 * @param final_state Comma-separated string: "roll,pitch,yaw,wx,wy,wz" (degrees, deg/s)
 * 
 * @return std::vector<std::vector<double>> with 2 elements:
 *         - [0]: Initial state [q0, q1, q2, q3, ωx, ωy, ωz] (7 elements, rad/s)
 *         - [1]: Final state [q0, q1, q2, q3, ωx, ωy, ωz] (7 elements, rad/s)
 * 
 * @note Angular velocities converted from DEGREES/SECOND to RADIANS/SECOND in output
 * @note Quaternions are normalized to unit length
 * @note Used by CGPOPS interface (cgpops_go function)
 * @note Does not preserve Euler angles (use parseInput() if needed)
 * 
 * @see parseInput() for CasADi DM version with detailed documentation
 */
std::vector<std::vector<double>> VparseStateVector(const std::string& initial_state, 
                                                    const std::string& final_state);

// =============================================================================
// TRAJECTORY I/O
// =============================================================================
// Functions for reading and writing trajectory data in various formats (CSV,
// MATLAB .m files). Supports both optimization warm-starting and result storage.

/**
 * @brief Extract initial guess from existing CSV trajectory file
 * 
 * Loads a previously computed trajectory to use as initial guess (warm start)
 * for a new optimization. This technique significantly improves convergence
 * speed and reliability for similar maneuvers.
 * 
 * CSV FILE FORMAT (expected columns, comma-separated):
 *   time, q0, q1, q2, q3, wx, wy, wz, ux, uy, uz
 * 
 * EXPECTED FILE STRUCTURE:
 *   - First row: Header (skipped)
 *   - Subsequent rows: Numerical data
 *   - At least 2 data rows required (initial and final points)
 * 
 * @param csv_data Path to CSV file containing trajectory data
 *                 Must contain complete state and control history with timestamps
 * 
 * @param[out] X_guess State trajectory matrix [7 x (n_points)]
 *                     Rows: [q0, q1, q2, q3, ωx, ωy, ωz]
 *                     Columns: Time samples (from initial to final)
 *                     Allocated/resized by this function
 * 
 * @param[out] U_guess Control sequence matrix [3 x (n_points-1)]
 *                     Rows: [τx, τy, τz]
 *                     Columns: Time intervals (one fewer than states)
 *                     Allocated/resized by this function
 * 
 * @param[out] dt_guess Time step vector [1 x (n_points-1)]
 *                      Duration of each control interval (seconds)
 *                      Computed from time column differences
 *                      Allocated/resized by this function
 * 
 * @throw std::runtime_error if file cannot be opened
 * @throw std::runtime_error if file has insufficient rows (< 2 data rows)
 * @throw std::runtime_error if CSV parsing fails (wrong number of columns)
 * 
 * @note Output matrices are cleared and resized by this function
 * @note CSV file must have header row (first row is skipped)
 * @note Time steps computed as dt[i] = time[i+1] - time[i]
 * @note Control at last time point duplicated from previous (for dimension matching)
 * 
 * @see exportTrajectory() for creating compatible CSV files
 * 
 * @par Example:
 * @code
 * casadi::DM X_guess, U_guess, dt_guess;
 * extractInitialGuess("../output/previous_solution.csv", X_guess, U_guess, dt_guess);
 * // Use X_guess, U_guess, dt_guess as warm start for new optimization
 * casadi::DMDict inputs = {{"x0_guess", X_guess}, {"u0_guess", U_guess}, ...};
 * @endcode
 */
void extractInitialGuess(const std::string& csv_data, 
                         casadi::DM& X_guess, 
                         casadi::DM& U_guess, 
                         casadi::DM& dt_guess);

/**
 * @brief Export trajectory data to CSV file for analysis and visualization
 * 
 * Saves complete optimal trajectory including states, controls, and time
 * information to CSV format. Output is compatible with Python/MATLAB/Excel
 * plotting tools and can be reloaded as initial guess for future optimizations.
 * 
 * OUTPUT CSV FORMAT (11 columns, comma-separated):
 *   time, q0, q1, q2, q3, wx, wy, wz, ux, uy, uz
 * 
 * CSV STRUCTURE:
 *   - Row 1: Descriptive header with maneuver information
 *   - Row 2+: Numerical trajectory data
 * 
 * @param X State trajectory matrix [7 x (n_steps+1)]
 *          Format: Each column is state at one time point
 *          Rows: [q0, q1, q2, q3, ωx, ωy, ωz]
 * 
 * @param U Control sequence matrix [3 x n_steps]
 *          Format: Each column is control at one time interval
 *          Rows: [τx, τy, τz]
 *          Note: One fewer column than X (piecewise constant controls)
 * 
 * @param T Total maneuver time (scalar, seconds)
 *          Used for header information and validation
 * 
 * @param dt Time step vector [1 x n_steps]
 *           Duration of each control interval (seconds)
 *           Should sum to approximately T
 * 
 * @param angles_0 Initial Euler angles [3 x 1] (degrees)
 *                 Format: [roll, pitch, yaw]
 *                 Used only for CSV header annotation
 * 
 * @param angles_f Final Euler angles [3 x 1] (degrees)
 *                 Format: [roll, pitch, yaw]
 *                 Used only for CSV header annotation
 * 
 * @param filename Output file path (relative or absolute)
 *                 Example: "../output/trajectory.csv"
 *                 Parent directories must exist
 * 
 * @throw std::runtime_error if file cannot be created/opened for writing
 * @throw std::runtime_error if parent directory does not exist
 * 
 * @note Creates human-readable CSV with descriptive header row
 * @note Time column computed as cumulative sum: t[i] = sum(dt[0:i])
 * @note Last control extended (duplicated) to match X dimensions for CSV format
 * @note Output file can be used as input to extractInitialGuess()
 * @note File is overwritten if it already exists
 * 
 * @see extractInitialGuess() for loading CSV trajectories as initial guess
 * 
 * @par Example:
 * @code
 * exportTrajectory(X_opt, U_opt, T_opt, dt_opt,
 *                  angles_initial, angles_final,
 *                  "../output/optimal_trajectory.csv");
 * // File created with header and trajectory data ready for plotting
 * @endcode
 */
void exportTrajectory(casadi::DM& X, const casadi::DM& U, const casadi::DM& T, 
                      const casadi::DM& dt, const casadi::DM& angles_0, 
                      const casadi::DM& angles_f, const std::string& filename);

/**
 * @brief Parse CGPOPS MATLAB output file to extract optimal trajectory
 * 
 * Reads MATLAB .m file generated by CGPOPS solver containing optimal trajectory
 * data in MATLAB assignment statement format. Extracts state history, control
 * history, and time information using regex pattern matching.
 * 
 * EXPECTED FILE FORMAT (MATLAB assignments):
 *   systemXX.phase(1).x(i).point(j) = value;  // State i at point j
 *   systemXX.phase(1).u(i).point(j) = value;  // Control i at point j
 *   systemXX.phase(1).t.point(j) = value;     // Time at point j
 *   systemXX.phase(1).tf = value;             // Final time
 * 
 * where XX is the two-character suffix determined by derivativeSupplier.
 * 
 * @param derivativeSupplier Integer identifier for derivative provider (0-3)
 *                           Used to construct filename suffix via fileSufix()
 *                           0→HD, 1→BC, 2→CF, 3→NF
 * 
 * @param[out] states State trajectory [7 x n_points]
 *                    Format: states[i][j] = i-th state component at j-th point
 *                    Order: [q0, q1, q2, q3, ωx, ωy, ωz]
 *                    Vector is cleared and resized by this function
 * 
 * @param[out] controls Control sequence [3 x n_points]
 *                      Format: controls[i][j] = i-th control at j-th point
 *                      Order: [τx, τy, τz]
 *                      Vector is cleared and resized by this function
 * 
 * @param[out] dt Time intervals between collocation points
 *               Computed as dt[i] = t[i+1] - t[i]
 *               Vector is cleared and populated by this function
 * 
 * @param[out] T Total maneuver time (final time from file)
 *              Extracted from systemXX.phase(1).tf assignment
 * 
 * @throw std::runtime_error if file cannot be opened
 * @throw std::runtime_error if file format is invalid (missing expected patterns)
 * @throw std::runtime_error if tf value not found in file
 * 
 * @note Uses regex pattern matching to parse MATLAB variable assignments
 * @note Automatically determines data dimensions from file content
 * @note File location: ../output/cgpopsIPOPTSolution{suffix}.m
 * @note All output vectors are cleared before populating
 * 
 * @see fileSufix() for derivative supplier code mapping
 * @see getCgpopsSolution() for extracting only objective value (faster)
 * 
 * @par Example:
 * @code
 * std::vector<std::vector<double>> states, controls;
 * std::vector<double> dt;
 * double T;
 * parseMatlab(1, states, controls, dt, T);  // Load BC derivative supplier results
 * // states[0-6][0..n]: q0, q1, q2, q3, wx, wy, wz trajectories
 * // controls[0-2][0..n]: ux, uy, uz trajectories
 * @endcode
 */
void parseMatlab(const int& derivativeSupplier,
                 std::vector<std::vector<double>>& states,
                 std::vector<std::vector<double>>& controls,
                 std::vector<double>& dt,
                 double& T);

/**
 * @brief Load state samples from CSV file for Monte Carlo analysis
 * 
 * Reads CSV file containing multiple initial and final state pairs for
 * batch optimization or statistical analysis (e.g., Monte Carlo simulation,
 * Latin Hypercube Sampling).
 * 
 * CSV FILE FORMAT (14 columns, no header):
 *   q0_i, q1_i, q2_i, q3_i, wx_i, wy_i, wz_i, q0_f, q1_f, q2_f, q3_f, wx_f, wy_f, wz_f
 * 
 * Each row represents one initial/final state pair:
 *   - Columns 0-6: Initial state [q0, q1, q2, q3, ωx, ωy, ωz]
 *   - Columns 7-13: Final state [q0, q1, q2, q3, ωx, ωy, ωz]
 * 
 * @param[out] initial_states Vector of initial state vectors (each 7 elements)
 *                            Cleared and populated by this function
 * 
 * @param[out] final_states Vector of final state vectors (each 7 elements)
 *                          Cleared and populated by this function
 * 
 * @param filename Path to CSV file containing state samples
 *                 Default: "../output/lhs_samples.csv"
 * 
 * @return true if file loaded successfully and data parsed
 *         false if file not found, cannot be opened, or parsing fails
 * 
 * @note Output vectors are cleared before loading (previous contents lost)
 * @note Quaternions in file should already be normalized (not verified)
 * @note Angular velocities should be in RADIANS/SECOND
 * @note File should NOT have header row (all rows treated as data)
 * @note Empty lines and malformed rows are skipped silently
 * 
 * @par Example:
 * @code
 * std::vector<std::vector<double>> X0_samples, Xf_samples;
 * if (loadStateSamples(X0_samples, Xf_samples)) {
 *     std::cout << "Loaded " << X0_samples.size() << " state pairs\n";
 *     for (size_t i = 0; i < X0_samples.size(); i++) {
 *         // Solve optimization for each state pair
 *         solveProblem(X0_samples[i], Xf_samples[i]);
 *     }
 * } else {
 *     std::cerr << "Failed to load state samples\n";
 * }
 * @endcode
 */
bool loadStateSamples(std::vector<std::vector<double>>& initial_states,
                      std::vector<std::vector<double>>& final_states,
                      const std::string& filename = "../output/lhs_samples.csv");

/**
 * @brief Load PSO (Particle Swarm Optimization) parameters from CSV file
 * 
 * Reads CSV file containing multiple parameter sets for PSO-based optimization
 * or hyperparameter tuning experiments. Format depends on PSO variant.
 * 
 * @param[out] params_vector Vector of parameter vectors
 *                           Each inner vector contains one PSO parameter set
 *                           Cleared and populated by this function
 * 
 * @param filename Path to CSV file containing PSO parameter samples
 *                 No default value - must be explicitly provided
 * 
 * @return true if file loaded successfully and data parsed
 *         false if file not found, cannot be opened, or parsing fails
 * 
 * @note CSV format is problem-specific (depends on PSO variant used)
 * @note Typical parameters: swarm size, inertia weight, cognitive/social coefficients
 * @note Output vector is cleared before loading (previous contents lost)
 * @note File should NOT have header row (all rows treated as data)
 * 
 * @par Example:
 * @code
 * std::vector<std::vector<double>> pso_params;
 * if (loadPSOSamples(pso_params, "../data/pso_hyperparameters.csv")) {
 *     for (const auto& params : pso_params) {
 *         // Run PSO with this parameter set
 *         runPSO(params);
 *     }
 * }
 * @endcode
 */
bool loadPSOSamples(std::vector<std::vector<double>>& params_vector,
                    const std::string& filename);

// =============================================================================
// OPTIMIZATION UTILITIES
// =============================================================================
// Functions for post-processing optimization results, extracting performance
// metrics, and displaying solution information.

/**
 * @brief Process and display optimization results with formatted output
 * 
 * Extracts key information from optimization result dictionary and displays
 * in human-readable format including maneuver time, boundary conditions,
 * and basic trajectory statistics. Useful for quick result verification.
 * 
 * EXPECTED RESULT DICTIONARY KEYS:
 *   - "X": State trajectory [7 x (n+1)] (required)
 *   - "U": Control sequence [3 x n] (required)
 *   - "T": Total time (scalar) (required)
 *   - "dt": Time steps [1 x n] (optional)
 * 
 * DISPLAYED INFORMATION (printed to std::cout):
 *   - Initial and final Euler angles
 *   - Total maneuver time
 *   - Number of time steps
 *   - Basic trajectory statistics
 * 
 * @param results CasADi dictionary (DMDict) containing optimization results
 *                Must contain at minimum: "X", "U", "T" keys
 * 
 * @param angles_0 Initial Euler angles [3 x 1] in DEGREES
 *                 Format: [roll, pitch, yaw]
 *                 Used for display purposes only (not modified)
 * 
 * @param angles_f Final Euler angles [3 x 1] in DEGREES
 *                 Format: [roll, pitch, yaw]
 *                 Used for display purposes only (not modified)
 * 
 * @note Output is printed to std::cout (not returned)
 * @note Function does not modify input data
 * @note If required keys are missing, may throw or produce incomplete output
 * @note Useful for quick result verification during development/debugging
 * 
 * @par Example:
 * @code
 * casadi::DMDict results = solver(inputs);
 * processResults(results, initial_euler, final_euler);
 * // Console output:
 * // ========================================
 * // Optimization Results
 * // ========================================
 * // Initial: [0.0, 0.0, 180.0] deg
 * // Final:   [0.0, 0.0, 0.0] deg
 * // Time:    2.35 seconds
 * // Steps:   50
 * // ========================================
 * @endcode
 */
void processResults(casadi::DMDict& results, const casadi::DM& angles_0, const casadi::DM& angles_f);

/**
 * @brief Extract optimal objective value from CGPOPS solution file
 * 
 * Reads only the final time (objective function value) from CGPOPS MATLAB
 * output file without loading the entire trajectory. Much faster than
 * parseMatlab() when only the objective value is needed.
 * 
 * FILE FORMAT EXPECTED (single line):
 *   systemXX.phase(1).tf = value;
 * 
 * where XX is determined by derivativeSupplier via fileSufix().
 * 
 * @param derivativeSupplier Integer identifier for derivative provider (0-3)
 *                           Determines which solution file to read
 *                           0→HD, 1→BC, 2→CF, 3→NF
 * 
 * @return Optimal final time in seconds (objective function value)
 * 
 * @throw std::runtime_error if file cannot be opened
 * @throw std::runtime_error if tf value not found in file
 * @throw std::runtime_error if tf line cannot be parsed
 * 
 * @note Much faster than parseMatlab() when only objective needed
 * @note Uses regex matching to extract single floating-point value
 * @note File path automatically constructed: ../output/cgpopsIPOPTSolution{suffix}.m
 * @note Does not validate file format beyond finding tf assignment
 * 
 * @see parseMatlab() for full trajectory extraction
 * @see fileSufix() for derivative supplier to suffix mapping
 * 
 * @par Example:
 * @code
 * double min_time = getCgpopsSolution(1);  // BC derivative supplier
 * std::cout << "Minimum time: " << min_time << " seconds\n";
 * // Much faster than loading entire trajectory if only time is needed
 * @endcode
 */
double getCgpopsSolution(const int& derivativeSupplier);

/**
 * @brief Get file suffix corresponding to derivative provider
 * 
 * Maps integer derivative provider codes to two-character string suffixes
 * used in CGPOPS output filenames. Enables systematic file naming across
 * different automatic differentiation methods.
 * 
 * MAPPING TABLE:
 *   Input → Output
 *   0     → "HD" (HyperDual numbers - first and second derivatives)
 *   1     → "BC" (Bicomplex numbers - complex-step differentiation)
 *   2     → "CF" (Central Finite Differences)
 *   3     → "NF" (Naive/Forward Finite Differences)
 *   other → "XX" (default/unknown)
 * 
 * @param derivative_provider Integer code (0-3, or other)
 * 
 * @return Two-character string suffix
 * 
 * @note Used for constructing CGPOPS output filenames systematically
 * @note Filename pattern: "cgpopsIPOPTSolution" + suffix + ".m"
 * @note For invalid codes, may return default "XX" or handle error
 * 
 * @see parseMatlab() uses this for file location determination
 * @see getCgpopsSolution() uses this for file location determination
 * 
 * @par Example:
 * @code
 * std::string suffix = fileSufix(1);
 * // suffix = "BC"
 * std::string filename = "../output/cgpopsIPOPTSolution" + suffix + ".m";
 * // filename = "../output/cgpopsIPOPTSolutionBC.m"
 * @endcode
 */
std::string fileSufix(const int& derivative_provider);

// =============================================================================
// SOLVER INTERFACE
// =============================================================================
// Functions for loading compiled optimization solvers and managing solver
// execution including logging and status monitoring.

/**
 * @brief Load compiled CasADi solver from shared library
 * 
 * Dynamically loads a pre-compiled optimization solver function from shared
 * object (.so on Linux/macOS, .dll on Windows) file. The solver is typically
 * generated using CasADi's code generation and compilation features for
 * improved performance compared to interpreted symbolic evaluation.
 * 
 * EXPECTED FILE LOCATION:
 *   ./solver.so (current directory, relative to executable)
 *   or platform-specific: solver.dll (Windows), solver.dylib (macOS)
 * 
 * SOLVER FUNCTION INTERFACE (typical):
 *   Inputs:  {"x0": initial_state, "xf": final_state, ...}
 *   Outputs: {"X": trajectory, "U": controls, "T": time, "dt": timesteps, ...}
 * 
 * @return CasADi Function object representing the loaded solver
 *         Can be called like: result = solver(input_dict)
 *         Returns DMDict with solution data
 * 
 * @throw std::runtime_error if .so/.dll file cannot be found
 * @throw std::runtime_error if file is not a valid CasADi function library
 * @throw std::runtime_error if function signature is incompatible with expected format
 * 
 * @note Compiled solver must be regenerated if problem formulation changes
 * @note Loading compiled function is much faster than symbolic evaluation
 * @note Shared library must be compiled for current platform/architecture
 * @note Function name within library is typically "solver" or specified during codegen
 * 
 * @par Example:
 * @code
 * casadi::Function solver = get_solver();
 * casadi::DMDict inputs = {{"x0", X_initial}, {"xf", X_final}};
 * casadi::DMDict result = solver(inputs);
 * casadi::DM X_opt = result.at("X");
 * casadi::DM U_opt = result.at("U");
 * @endcode
 */
casadi::Function get_solver();

/**
 * @brief Redirect console output to log file
 * 
 * Redirects stdout and stderr to specified file for capturing solver output,
 * warnings, and error messages. Essential for batch processing and automated
 * testing where console output needs to be captured or suppressed.
 * 
 * MECHANISM:
 * Uses POSIX dup2() system call to redirect file descriptors:
 * - Saves current stdout (fd=1) and stderr (fd=2) descriptors
 * - Opens log file with write permissions
 * - Redirects stdout and stderr to log file descriptor
 * 
 * @param filename Path to output log file (created if doesn't exist, truncated if exists)
 *                 Example: "../output/solver_logs.log"
 *                 Parent directories must exist
 * 
 * @throw std::runtime_error if file cannot be created/opened
 * @throw std::runtime_error if dup() or dup2() system calls fail
 * @throw std::runtime_error if parent directory does not exist
 * 
 * @note POSIX-specific (Unix/Linux/macOS) - NOT portable to Windows without modification
 * @note Must call restore_output_to_console() to restore normal output
 * @note File is opened in write mode (O_WRONLY | O_CREAT | O_TRUNC)
 * @note Both stdout and stderr are redirected to same file
 * @note Nested redirections not supported (restore before new redirect)
 * 
 * @see restore_output_to_console() for output restoration
 * @see get_solver_status() for parsing redirected output logs
 * 
 * @par Example:
 * @code
 * redirect_output_to_file("../output/solver.log");
 * auto solver = get_solver();
 * auto result = solver(inputs);  // Output goes to file, not console
 * restore_output_to_console();   // Restore normal output
 * int status = get_solver_status("../output/solver.log");
 * @endcode
 */
void redirect_output_to_file(const std::string& filename);

/**
 * @brief Restore console output after redirection
 * 
 * Restores stdout and stderr to their original destinations (typically the
 * terminal) after they were redirected using redirect_output_to_file().
 * Must be called to return to normal console I/O.
 * 
 * MECHANISM:
 * Uses POSIX dup2() to restore saved file descriptors:
 * - Restores original stdout descriptor
 * - Restores original stderr descriptor
 * - Closes log file descriptor
 * - Resets global state variables
 * 
 * @throw std::runtime_error if restoration fails (e.g., invalid saved descriptors)
 * @throw std::runtime_error if close() fails on log file descriptor
 * 
 * @note Must be called after redirect_output_to_file() to restore normal I/O
 * @note Safe to call multiple times (subsequent calls are no-ops)
 * @note Safe to call even if output was not redirected (checks global state)
 * @note POSIX-specific implementation (not portable to Windows)
 * @note After restoration, log file remains on disk with captured output
 * 
 * @see redirect_output_to_file() for output redirection
 * 
 * @par Example:
 * @code
 * redirect_output_to_file("log.txt");
 * std::cout << "This goes to file\n";
 * restore_output_to_console();
 * std::cout << "This goes to console\n";  // Normal output restored
 * @endcode
 */
void restore_output_to_console();

/**
 * @brief Extract solver status code from log file
 * 
 * Parses solver output log to determine convergence status. Searches for
 * standard IPOPT solver return messages and maps them to integer codes
 * for programmatic status checking.
 * 
 * RECOGNIZED IPOPT MESSAGES (case-sensitive):
 *   - "EXIT: Optimal Solution Found." → 0 (success)
 *   - "EXIT: Solved To Acceptable Level." → 1 (acceptable)
 *   - "EXIT: Maximum CPU Time Exceeded." → -1
 *   - "EXIT: Restoration Failed!" → -2
 *   - "EXIT: Maximum Number of Iterations Exceeded." → -3
 *   - "EXIT: Converged to a point of local infeasibility." → -4
 *   - Other/none found → -5 (unknown/failure)
 * 
 * @param filename Path to solver log file
 *                 Default: "../output/solver_logs.log"
 * 
 * @return Integer status code:
 *         -  0: Optimal solution found (success)
 *         -  1: Solved to acceptable level (success with relaxed tolerances)
 *         - -1: Maximum CPU time exceeded (incomplete)
 *         - -2: Restoration failed (infeasible or numerical issues)
 *         - -3: Maximum iterations exceeded (incomplete)
 *         - -4: Local infeasibility (problem may be infeasible)
 *         - -5: Other failure or log file not found/parseable
 * 
 * @note Log file must contain IPOPT solver output text
 * @note String matching is case-sensitive
 * @note Returns -5 if file cannot be opened
 * @note Only checks for specific IPOPT exit messages
 * @note First matching message determines return code
 * 
 * @see redirect_output_to_file() for capturing solver output to log file
 * 
 * @par Example:
 * @code
 * redirect_output_to_file("solver.log");
 * casadi::DMDict result = solver(inputs);
 * restore_output_to_console();
 * 
 * int status = get_solver_status("solver.log");
 * if (status == 0) {
 *     std::cout << "Optimal solution found!\n";
 * } else if (status == 1) {
 *     std::cout << "Acceptable solution found.\n";
 * } else if (status < 0) {
 *     std::cerr << "Solver failed with code " << status << "\n";
 * }
 * @endcode
 */
int get_solver_status(const std::string& filename = "../output/solver_logs.log");

// =============================================================================
// RESULT FILE MANAGEMENT
// =============================================================================
// Functions for managing output result files including crash recovery,
// appending new results, and progress reporting.

/**
 * @brief Count number of existing result rows in output CSV file
 * 
 * Determines how many simulations have already been completed by counting
 * non-empty lines in the results file (excluding header). This enables
 * crash recovery by resuming from the last completed iteration.
 * 
 * @param filepath Path to results CSV file
 * @param file_exists Output parameter: true if file exists
 * 
 * @return Number of data rows already written (0 if file doesn't exist or is empty)
 * 
 * @note Assumes first line is header (skipped in count)
 * @note Empty lines are ignored
 * @note Returns 0 if file cannot be opened
 */
int countExistingResults(const std::string& filepath, bool& file_exists);

/**
 * @brief Initialize results CSV file with header or open for appending
 * 
 * Creates new results file with header if needed, or opens existing file
 * for appending. Handles overwrite mode and crash recovery scenarios.
 * 
 * @param filepath Path to results CSV file
 * @param overwrite If true, create new file (overwrites existing)
 * @param file_exists Input: whether file currently exists
 * 
 * @return FILE* pointer for writing, or nullptr on error
 * 
 * @note Caller is responsible for closing returned FILE*
 * @note Returns nullptr on failure (error message printed)
 * @note Header format: "T, time, status\n"
 */
FILE* initializeResultsFile(const std::string& filepath, bool overwrite, bool file_exists);

/**
 * @brief Display progress update with time estimates
 * 
 * Prints formatted progress information including:
 * - Current iteration and total count
 * - Completion percentage
 * - Elapsed time
 * - Estimated time remaining
 * 
 * @param current_iteration Current iteration number (0-indexed)
 * @param total_iterations Total number of iterations
 * @param start_time Time point when processing started
 * 
 * @note Output formatted for readability with fixed precision
 * @note Time estimates improve in accuracy as more iterations complete
 */
void reportProgress(int current_iteration, int total_iterations,
                   const std::chrono::high_resolution_clock::time_point& start_time);

// =============================================================================
// NUMERICAL INTEGRATION
// =============================================================================
// Functions for numerical integration of differential equations and construction
// of mathematical operators used in dynamics formulation.

/**
 * @brief Construct 4×4 skew-symmetric matrix from 3D angular velocity vector
 * 
 * Creates the skew-symmetric matrix Ω(ω) used in quaternion kinematics
 * differential equation: q̇ = 0.5 * Ω(ω) * q
 * 
 * MATHEMATICAL DEFINITION:
 *        ⎡  0   -ωx  -ωy  -ωz ⎤
 *   Ω = ⎢  ωx    0   ωz  -ωy ⎥
 *        ⎢  ωy  -ωz    0   ωx ⎥
 *        ⎣  ωz   ωy  -ωx    0 ⎦
 * 
 * PROPERTIES:
 * - Skew-symmetric: Ω^T = -Ω (transpose equals negative)
 * - Trace zero: tr(Ω) = 0
 * - Maps 3D angular velocity to 4D quaternion rate space
 * - Used in quaternion propagation: q̇ = 0.5 * Ω(ω) * q
 * 
 * @param w Angular velocity vector [ωx, ωy, ωz]ᵀ as CasADi MX [3 x 1]
 *          Units: radians/second (or any consistent angular rate unit)
 * 
 * @return 4×4 skew-symmetric matrix as CasADi MX [4 x 4]
 *         Satisfies Ω^T = -Ω
 * 
 * @note Input is 3D vector but output is 4×4 matrix (quaternion space dimension)
 * @note Used internally in quaternion dynamics formulation
 * @note Matrix is always skew-symmetric by construction
 * @note No normalization or validation of input performed
 * 
 * @par Example:
 * @code
 * casadi::MX omega = casadi::MX::sym("omega", 3);  // Symbolic angular velocity
 * casadi::MX Omega = skew4(omega);                 // 4x4 skew-symmetric matrix
 * casadi::MX q = casadi::MX::sym("q", 4);          // Quaternion
 * casadi::MX q_dot = 0.5 * casadi::MX::mtimes(Omega, q);  // Quaternion rate
 * @endcode
 */
casadi::MX skew4(const casadi::MX& w);

/**
 * @brief Perform one step of 4th-order Runge-Kutta integration
 * 
 * Integrates ordinary differential equation ẋ = f(x) forward by time step dt
 * using classical RK4 method. Provides 4th-order accuracy in dt, making it
 * suitable for most spacecraft dynamics applications.
 * 
 * ALGORITHM (Classical RK4):
 *   k1 = f(x)
 *   k2 = f(x + dt/2 * k1)
 *   k3 = f(x + dt/2 * k2)
 *   k4 = f(x + dt * k3)
 *   x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 * 
 * PROPERTIES:
 * - 4th order accuracy: local error ~ O(dt^5), global error ~ O(dt^4)
 * - Requires 4 function evaluations per step (higher than Euler, lower than RK8)
 * - More accurate than Euler or RK2 for same step size
 * - Standard choice for moderate accuracy requirements
 * - Explicit method (no implicit equation solving)
 * 
 * @param x_dot Derivative function ẋ = f(x) as CasADi MX
 *              Must be expression in terms of x (state) only
 *              Can depend on controls/parameters as constants
 * 
 * @param x Current state vector as CasADi MX
 *          Can be any dimension (scalar or vector)
 * 
 * @param dt Time step (scalar) as CasADi MX
 *           Smaller dt → higher accuracy but more computation
 *           Larger dt → faster but less accurate, may be unstable
 * 
 * @return Next state x(t + dt) as CasADi MX
 *         Same dimension as input x
 * 
 * @note For quaternion states, normalize after integration to maintain ||q|| = 1
 * @note Time step dt should be chosen based on system dynamics time scale
 * @note Typical dt: 0.001-0.1 seconds for spacecraft attitude dynamics
 * @note Method is explicit - no matrix inversions required
 * @note Stability limited by dt (explicit method stability constraint)
 * 
 * @par Example:
 * @code
 * casadi::MX x = casadi::MX::sym("x", 7);           // State [q, omega]
 * casadi::MX u = casadi::MX::sym("u", 3);           // Control
 * casadi::MX x_dot = computeDynamics(x, u);         // Compute ẋ = f(x,u)
 * casadi::MX dt = 0.01;                             // 10ms time step
 * casadi::MX x_next = rk4(x_dot, x, dt);           // Integrate forward
 * // For quaternions: normalize x_next[0:4] after integration
 * @endcode
 */
casadi::MX rk4(const casadi::MX& x_dot, const casadi::MX& x, const casadi::MX& dt);

// =============================================================================
// SYSTEM UTILITIES
// =============================================================================
// Miscellaneous utility functions for numerical formatting and system operations.

/**
 * @brief Round double to specified decimal precision
 * 
 * Rounds a floating-point value to the specified number of decimal places
 * using standard rounding (round half away from zero). Useful for formatting
 * output and comparing numerical results within tolerance.
 * 
 * ALGORITHM:
 *   1. Multiply value by 10^precision
 *   2. Round to nearest integer using std::round()
 *   3. Divide by 10^precision
 * 
 * ROUNDING RULE:
 *   - Round half away from zero (standard rounding)
 *   - 2.5 → 3.0, -2.5 → -3.0
 * 
 * @param value The double value to round
 * @param precision Number of decimal places to retain (default: 2)
 *                  Range: 0 to ~15 (limited by double precision ~16 digits)
 *                  Negative values round to left of decimal (tens, hundreds, etc.)
 * 
 * @return Rounded value with specified decimal places
 * 
 * @note Uses standard rounding (round half away from zero), not banker's rounding
 * @note Precision > 15 may not be meaningful due to double precision limits (~15-17 digits)
 * @note Result is still double type (not truncated string representation)
 * @note Negative precision rounds to left of decimal: rnd(1234, -2) → 1200
 * 
 * @par Examples:
 * @code
 * double x = rnd(3.14159);        // Result: 3.14 (default precision=2)
 * double y = rnd(3.14159, 3);     // Result: 3.142
 * double z = rnd(2.5, 0);         // Result: 3.0 (round half away from zero)
 * double w = rnd(-2.5, 0);        // Result: -3.0
 * double v = rnd(1234.5, -2);     // Result: 1200.0 (round to nearest hundred)
 * @endcode
 */
double rnd(double value, int precision = 2);

// =============================================================================
// END OF HEADER
// =============================================================================

#endif /* HELPER_FUNCTIONS_H */