/**
 * @file cgpopsAuxDec.hpp
 * @brief CGPOPS auxiliary variable definitions (migrated to TOAC library)
 * 
 * OVERVIEW:
 * =========
 * This header historically contained the initial definitions (with default values)
 * for global variables used throughout the CGPOPS framework for spacecraft attitude
 * optimal control problems. These definitions have been migrated to the TOAC
 * (Time-Optimal Attitude Control) library for centralized management.
 * 
 * CURRENT STATE:
 * ==============
 * All variable definitions are now commented out and preserved as reference
 * documentation. The actual definitions are provided by the TOAC library's
 * implementation files, ensuring:
 * - Single source of truth for default values
 * - Consistent initialization across the framework
 * - Centralized parameter management
 * - Reduced code duplication and maintenance burden
 * 
 * RELATIONSHIP TO cgpopsAuxExt.hpp:
 * ==================================
 * These files work together in the extern/definition pattern:
 * 
 * cgpopsAuxExt.hpp (External Declarations):
 *   - Declares variables as 'extern' (no storage allocation)
 *   - Included by files that USE the variables
 *   - Now includes <toac/symmetric_spacecraft.h> for declarations
 * 
 * cgpopsAuxDec.hpp (Definitions - THIS FILE):
 *   - Originally defined variables with initial values (storage allocation)
 *   - Included by ONE implementation file to create the actual variables
 *   - Now deprecated in favor of TOAC library definitions
 * 
 * USAGE GUIDELINES:
 * ==================
 * For new code, use:
 * @code
 * #include <cgpops/cgpopsAuxExt.hpp>  // Gets extern declarations from TOAC
 * // Variables are defined in TOAC library implementation
 * @endcode
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 * @note This file is DEPRECATED for active use - preserved for documentation only
 * 
 * @see cgpopsAuxExt.hpp External declarations (now from TOAC library)
 * @see <toac/symmetric_spacecraft.h> Current source of definitions
 * 
 * @author Yunus M. Agamawi, Anil Vithala Rao
 */

#ifndef __CGPOPS_AUX_DEC_HPP__
#define __CGPOPS_AUX_DEC_HPP__

// =============================================================================
// CGPOPS CLASS DEFINITIONS
// =============================================================================

/**
 * @brief CGPOPS class definitions and type declarations
 * 
 * This header provides fundamental class definitions used throughout the
 * CGPOPS framework, including:
 * - Base classes for equations (ObjectiveEq, PathConstraintEq, OrdinaryDifferentialEq)
 * - Data structures for NLP problem representation
 * - Type definitions for automatic differentiation (HyperDual, Bicomplex)
 * - Utility classes and helper functions
 * 
 * These definitions are required even though variable definitions have been
 * migrated to TOAC library, as they define the core framework types.
 * 
 * @note This include is still necessary for framework type definitions
 */
#include <cgpopsClassDef.hpp>

// =============================================================================
// REFERENCE DOCUMENTATION: ORIGINAL VARIABLE DEFINITIONS
// =============================================================================
// The following definitions were originally active in this header but are now
// provided by the TOAC library. They are preserved as commented reference
// documentation to show:
// 1. Original default values used in the framework
// 2. Rationale for default value choices
// 3. Physical meaning and units for each parameter
//
// DO NOT UNCOMMENT unless removing TOAC library dependency entirely.
//
// HISTORICAL PATTERN:
// This header was included in exactly ONE .cpp implementation file to create
// and initialize these global variables. This is the definition side of the
// extern/definition pattern (cgpopsAuxExt.hpp contains extern declarations).
// =============================================================================

// -----------------------------------------------------------------------------
// SPACECRAFT PHYSICAL PROPERTIES
// -----------------------------------------------------------------------------
// Principal moments of inertia about body-fixed coordinate axes (kg⋅m²)
//
// DEFAULT VALUES: 1.0 kg⋅m² for all axes
// RATIONALE:
// - Represents a symmetric spacecraft (Ix = Iy = Iz)
// - Unity values simplify initial testing and debugging
// - Provides neutral baseline for normalized dynamics
// - Must be overridden with actual spacecraft parameters for real problems
//
// PHYSICAL CONSTRAINTS:
// - All moments must be positive: Ix, Iy, Iz > 0
// - Triangle inequality: |Ix - Iy| < Iz < Ix + Iy (and cyclic permutations)
//
// TYPICAL RANGES:
// - Small spacecraft (CubeSats): 0.001 - 0.1 kg⋅m²
// - Medium spacecraft: 0.5 - 5.0 kg⋅m²
// - Large spacecraft: 10 - 1000 kg⋅m²
//
//
// double Ix = 1.0;  // X-axis moment of inertia (kg⋅m²) [DEFAULT: symmetric spacecraft]
// double Iy = 1.0;  // Y-axis moment of inertia (kg⋅m²) [DEFAULT: symmetric spacecraft]
// double Iz = 1.0;  // Z-axis moment of inertia (kg⋅m²) [DEFAULT: symmetric spacecraft]

// -----------------------------------------------------------------------------
// CONTROL BOUNDS
// -----------------------------------------------------------------------------
// Maximum and minimum applied torque from actuators (N⋅m)
//
// DEFAULT VALUES: ±1.0 N⋅m (symmetric bounds)
// RATIONALE:
// - Symmetric bounds (|tau_min| = tau_max) simplify analysis
// - Unity magnitude provides normalized baseline
// - Typical for small spacecraft reaction wheel systems
// - Must be adjusted based on actual actuator capabilities
//
// PHYSICAL INTERPRETATION:
// - Limits the control authority available for maneuvers
// - Directly determines minimum achievable maneuver time
// - Smaller torque limits → longer minimum time
// - Should account for actuator saturation and dynamics
//
// TYPICAL RANGES:
// - Micro spacecraft: ±0.001 - ±0.1 N⋅m
// - Small spacecraft: ±0.1 - ±1.0 N⋅m
// - Medium spacecraft: ±1.0 - ±10.0 N⋅m
// - Large spacecraft: ±10.0 - ±100.0 N⋅m
//
// double tau_max{1.0};   // Maximum applied torque (N⋅m) [DEFAULT: 1.0]
// double tau_min{-1.0};  // Minimum applied torque (N⋅m) [DEFAULT: -1.0]

// -----------------------------------------------------------------------------
// STATE BOUNDS
// -----------------------------------------------------------------------------
// Bounds on state variables during optimal trajectory
//
// QUATERNION BOUNDS:
// DEFAULT VALUES: q_min = -1.0, q_max = 1.0
// RATIONALE:
// - Covers full range of possible quaternion component values
// - Unit quaternion constraint: q₀² + q₁² + q₂² + q₃² = 1 ensures |qᵢ| ≤ 1
// - These bounds allow unrestricted attitude space exploration
// - Can be tightened if maneuver is restricted to smaller attitude range
//
// ANGULAR VELOCITY BOUNDS:
// DEFAULT VALUES: w_min = -3.0 rad/s, w_max = 3.0 rad/s (symmetric)
// RATIONALE:
// - Symmetric bounds simplify analysis
// - 3.0 rad/s ≈ 172°/s is reasonable for many spacecraft
// - Based on structural limits, sensor ranges, control authority
// - Should be adjusted for specific spacecraft capabilities
//
// TYPICAL ANGULAR VELOCITY RANGES:
// - Slow maneuvers: ±0.1 - ±1.0 rad/s
// - Medium maneuvers: ±1.0 - ±5.0 rad/s  
// - Fast maneuvers: ±5.0 - ±20.0 rad/s
//
// double q_max{1.0};   // Maximum quaternion component [DEFAULT: 1.0, full range]
// double q_min{-1.0};  // Minimum quaternion component [DEFAULT: -1.0, full range]
// double w_max{3.0};   // Maximum angular velocity (rad/s) [DEFAULT: 3.0 rad/s ≈ 172°/s]
// double w_min{-3.0};  // Minimum angular velocity (rad/s) [DEFAULT: -3.0 rad/s]

// -----------------------------------------------------------------------------
// TIME PARAMETERS
// -----------------------------------------------------------------------------
// Time bounds for the optimal control problem
//
// DEFAULT VALUES: T_min = 0.0 s, T_max = 6.0 s
// RATIONALE:
// - T_min = 0.0: Allows optimizer full freedom (may be changed to small positive value)
// - T_max = 6.0: Reasonable upper bound for moderate spacecraft maneuvers
// - Range accommodates typical rest-to-rest attitude changes
//
// USAGE NOTES:
// - T_min > 0 prevents singular solutions and numerical issues (recommend 0.1 - 1.0 s)
// - T_max should be based on operational requirements
// - Tighter bounds improve convergence but may exclude optimal solution
// - Initial guess T_0 (not defined here) should be within [T_min, T_max]
//
// TYPICAL TIME SCALES:
// - Fast maneuvers: 0.5 - 2.0 seconds
// - Moderate maneuvers: 2.0 - 10.0 seconds
// - Slow/precise maneuvers: 10.0 - 60.0 seconds
//
// double T_max{6.0};  // Maximum final time (seconds) [DEFAULT: 6.0 s]
// double T_min{0.0};  // Minimum final time (seconds) [DEFAULT: 0.0 s, recommend > 0]

// -----------------------------------------------------------------------------
// PROBLEM DIMENSIONS
// -----------------------------------------------------------------------------
// Number of state and control variables in the optimal control problem
//
// DEFAULT VALUES: n_states = 7, n_controls = 3
// RATIONALE:
// - n_states = 7: Standard for quaternion-based attitude control
//   [q₀, q₁, q₂, q₃, ωx, ωy, ωz] = [quaternion (4) + angular velocity (3)]
// - n_controls = 3: Standard for 3-axis attitude control
//   [τx, τy, τz] = torque components about body axes
//
// INVARIANT:
// These values are fixed by the problem formulation and should not be changed
// unless fundamentally altering the state/control representation (e.g., using
// Euler angles instead of quaternions, or adding additional states).
//
// int n_states{7};    // Number of state variables [FIXED: quaternion formulation]
// int n_controls{3};  // Number of control variables [FIXED: 3-axis control]

// -----------------------------------------------------------------------------
// NUMERICAL PARAMETERS
// -----------------------------------------------------------------------------
// Number of time steps for discretization/integration
//
// DEFAULT VALUE: n_stp = 50
// RATIONALE:
// - Provides moderate temporal resolution for auxiliary calculations
// - Balance between accuracy and computational cost
// - May be used in initial mesh generation or post-processing
//
// NOTE: This parameter is separate from CGPOPS collocation mesh, which is
// configured independently via numintervalsG and initcolptsG.
//
// TYPICAL RANGES:
// - Coarse: 10 - 30 steps
// - Medium: 50 - 100 steps
// - Fine: 100 - 500 steps
//
// double n_stp{50};  // Number of time steps [DEFAULT: 50, moderate resolution]

// -----------------------------------------------------------------------------
// MATHEMATICAL CONSTANTS
// -----------------------------------------------------------------------------
// Standard mathematical constants for unit conversions
//
// DEFAULT VALUES:
// - PI = 3.141592653589793 (mathematical constant π)
// - DEG = PI / 180.0 ≈ 0.01745 (radians per degree conversion factor)
// - RAD = 180.0 / PI ≈ 57.296 (degrees per radian conversion factor)
//
// RATIONALE:
// - High-precision PI value for accurate calculations
// - Conversion factors enable easy angle unit conversions
// - Standardized across framework for consistency
//
// USAGE:
// - angle_rad = angle_deg * DEG  (degrees → radians)
// - angle_deg = angle_rad * RAD  (radians → degrees)
//
// NOTE: Modern C++ provides M_PI in <cmath>, but these explicit definitions
// ensure portability and precision consistency.
//
// double PI{3.141592653589793};  // Mathematical constant π [HIGH PRECISION]
// double DEG{PI / 180.0};         // Radians per degree [CONVERSION: deg→rad]
// double RAD{180.0 / PI};         // Degrees per radian [CONVERSION: rad→deg]

// =============================================================================
// END OF HEADER
// =============================================================================

#endif  // __CGPOPS_AUX_DEC_HPP__