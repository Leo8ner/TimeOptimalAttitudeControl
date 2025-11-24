/**
 * @file cgpopsAuxExt.hpp
 * @brief CGPOPS auxiliary variable declarations and TOAC library integration
 * 
 * OVERVIEW:
 * =========
 * This header serves as the primary interface between the CGPOPS (C++ General
 * Pseudospectral Optimal Control Software) framework and the TOAC (Time-Optimal
 * Attitude Control) library. It provides access to global physical parameters,
 * problem bounds, and configuration constants required for spacecraft attitude
 * optimal control problems.
 * 
 * ARCHITECTURE:
 * =============
 * Originally, this file contained explicit external declarations for all global
 * variables. The implementation has been refactored to use the TOAC library's
 * symmetric_spacecraft.h header, which now provides these declarations in a
 * centralized, maintainable location.
 * 
 * The TOAC library symmetric_spacecraft.h header defines:
 * - Physical properties: moments of inertia (i_x, i_y, i_z)
 * - Control bounds: torque limits (tau_min, tau_max)
 * - State bounds: angular velocity limits (w_min, w_max), quaternion bounds (q_min, q_max)
 * - Time parameters: time guess (T_0), time bounds (T_min, T_max)
 * - Problem dimensions: number of states (n_states), number of controls (n_controls)
 * - Mathematical constants: PI, DEG (degrees per radian), RAD (radians per degree)
 * - Numerical parameters: number of time steps (n_stp)
 * 
 * USAGE:
 * ======
 * Simply include this header in any file requiring access to CGPOPS auxiliary
 * variables and TOAC library functionality.
 * 
 * 
 * MIGRATION NOTES:
 * ================
 * Previous versions of this file explicitly declared external variables.
 * These declarations are preserved below as commented reference documentation
 * to maintain clarity about what the TOAC library provides.
 * 
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 * 
 * @see <toac/symmetric_spacecraft.h> TOAC library spacecraft parameter definitions
 * @see cgpops_main.hpp Main interface requiring these parameters
 * @see cgpops_gov.hpp Governing equations using these parameters
 * 
 * @author Yunus M. Agamawi, Anil Vithala Rao
 */

#ifndef __CGPOPS_AUX_EXT_HPP__
#define __CGPOPS_AUX_EXT_HPP__

// =============================================================================
// TOAC LIBRARY INTEGRATION
// =============================================================================
// The TOAC (Time-Optimal Attitude Control) library provides centralized
// definitions for all spacecraft physical properties, problem bounds, and
// configuration parameters required by the CGPOPS framework.

/**
 * @brief TOAC library header providing spacecraft parameters and problem bounds
 * 
 * This header from the TOAC library defines all global variables previously
 * declared explicitly in this file. It provides:
 * 
 * SPACECRAFT PHYSICAL PROPERTIES:
 * - i_x, i_y, i_z : Principal moments of inertia (kg⋅m²)
 * 
 * CONTROL BOUNDS:
 * - tau_min : Minimum applied torque (N⋅m, typically negative)
 * - tau_max : Maximum applied torque (N⋅m, typically positive)
 * 
 * STATE BOUNDS:
 * - q_min, q_max : Quaternion component bounds (typically [-1, 1])
 * - w_min, w_max : Angular velocity bounds (rad/s)
 * 
 * TIME PARAMETERS:
 * - T_0    : Initial guess for final time (seconds)
 * - T_min  : Minimum final time bound (seconds)
 * - T_max  : Maximum final time bound (seconds)
 * 
 * PROBLEM DIMENSIONS:
 * - n_states   : Number of state variables (7 for quaternion + angular velocity)
 * - n_controls : Number of control variables (3 for torque components)
 * 
 * NUMERICAL PARAMETERS:
 * - n_stp : Number of time steps for numerical methods
 * 
 * MATHEMATICAL CONSTANTS:
 * - PI  : π ≈ 3.14159265358979323846
 * - DEG : Degrees per radian (180/π)
 * - RAD : Radians per degree (π/180)
 * 
 * @note All variables are declared as external doubles or integers
 * @note Variables must be defined (assigned values) before use
 * @note Typical initialization occurs in problem setup code
 */
#include <toac/symmetric_spacecraft.h>

// =============================================================================
// REFERENCE DOCUMENTATION: PREVIOUSLY EXPLICIT DECLARATIONS
// =============================================================================
// The following declarations were previously explicit in this header but are
// now provided by the TOAC library. They are preserved here as commented
// reference documentation to clarify the interface contract.
//
// DO NOT UNCOMMENT unless removing the TOAC library dependency.
// =============================================================================

// -----------------------------------------------------------------------------
// SPACECRAFT PHYSICAL PROPERTIES
// -----------------------------------------------------------------------------
// Principal moments of inertia about body-fixed axes (kg⋅m²)
//
// These values characterize the spacecraft's mass distribution and determine
// its rotational dynamics. For a rigid body with principal axes aligned to
// the body frame:
// - i_x : Moment of inertia about x-axis
// - i_y : Moment of inertia about y-axis  
// - i_z : Moment of inertia about z-axis
//
// Typical values: 0.5 - 5.0 kg⋅m² for small spacecraft
// Constraints: i_x, i_y, i_z > 0 (physical requirement)
//
// extern double i_x;  // X-axis moment of inertia (kg⋅m²)
// extern double i_y;  // Y-axis moment of inertia (kg⋅m²)
// extern double i_z;  // Z-axis moment of inertia (kg⋅m²)

// -----------------------------------------------------------------------------
// CONTROL BOUNDS
// -----------------------------------------------------------------------------
// Applied torque limits from actuators (reaction wheels, thrusters, etc.)
//
// These bounds represent the maximum control authority available for the
// attitude maneuver. They directly impact the minimum achievable time.
//
// Typical values: ±0.1 to ±10.0 N⋅m depending on spacecraft size
// Convention: tau_min < 0 < tau_max (symmetric bounds recommended)
//
// extern double tau_max;  // Maximum applied torque (N⋅m)
// extern double tau_min;  // Minimum applied torque (N⋅m)

// -----------------------------------------------------------------------------
// STATE BOUNDS
// -----------------------------------------------------------------------------
// Bounds on state variables during trajectory
//
// Quaternion component bounds:
// - Standard bounds: [-1, 1] (covers all possible attitudes)
// - May be tightened for specific maneuvers if attitude range is limited
//
// Angular velocity bounds:
// - Based on spacecraft dynamics, structural limits, and sensor range
// - Typical values: ±1 to ±10 rad/s for small spacecraft
//
// extern double q_max;  // Maximum quaternion component value
// extern double q_min;  // Minimum quaternion component value
// extern double w_max;  // Maximum angular velocity (rad/s)
// extern double w_min;  // Minimum angular velocity (rad/s)

// -----------------------------------------------------------------------------
// TIME PARAMETERS
// -----------------------------------------------------------------------------
// Time bounds and initial guess for the optimal control problem
//
// T_0: Initial guess for final time
// - Should be a reasonable estimate to aid convergence
// - For 180° rotation: typically 1-5 seconds depending on inertia and torque
// - Can be computed analytically or from previous solutions
//
// T_min: Minimum final time
// - Prevents singular solutions and numerical issues
// - Should be positive and physically meaningful
// - Typical value: 0.1 - 1.0 seconds
//
// T_max: Maximum final time
// - Prevents excessively long maneuvers
// - Based on operational requirements or physical constraints
// - Typical value: 5.0 - 20.0 seconds
//
// extern double T_0;    // Initial time guess (seconds)
// extern double T_max;  // Maximum final time bound (seconds)
// extern double T_min;  // Minimum final time bound (seconds)

// -----------------------------------------------------------------------------
// PROBLEM DIMENSIONS
// -----------------------------------------------------------------------------
// Number of state and control variables
//
// For spacecraft attitude control using quaternions:
// - n_states = 7 : quaternion (4) + angular velocity (3)
// - n_controls = 3 : torque components (τx, τy, τz)
//
// These dimensions define the size of the optimal control problem and
// affect memory allocation in the CGPOPS framework.
//
// extern int n_states;    // Number of state variables (typically 7)
// extern int n_controls;  // Number of control variables (typically 3)

// -----------------------------------------------------------------------------
// NUMERICAL PARAMETERS
// -----------------------------------------------------------------------------
// Number of time steps for numerical integration or discretization
//
// This parameter may be used for:
// - Initial mesh generation
// - Numerical integration in auxiliary calculations
// - Post-processing and visualization
//
// Note: Not directly used in CGPOPS collocation (which uses its own mesh)
// but may be referenced in pre/post-processing code.
//
// extern double n_stp;  // Number of time steps

// -----------------------------------------------------------------------------
// MATHEMATICAL CONSTANTS
// -----------------------------------------------------------------------------
// Standard mathematical constants for unit conversions and calculations
//
// PI: The mathematical constant π
// - Used in trigonometric calculations and rotations
// - Value: 3.14159265358979323846
//
// DEG: Degrees per radian
// - Conversion factor: multiply radians by DEG to get degrees
// - Value: 180/π ≈ 57.2957795131
// - Usage: angle_deg = angle_rad * DEG
//
// RAD: Radians per degree  
// - Conversion factor: multiply degrees by RAD to get radians
// - Value: π/180 ≈ 0.0174532925
// - Usage: angle_rad = angle_deg * RAD
//
// extern double DEG;  // Degrees per radian (180/π)
// extern double RAD;  // Radians per degree (π/180)
// extern double PI;   // Mathematical constant π

// =============================================================================
// USAGE GUIDELINES
// =============================================================================
// 
// INITIALIZATION:
// Before calling cgpops_go() or any CGPOPS functions, ensure all variables
// are properly initialized with physically meaningful values.
//
// VALIDATION:
// Verify that:
// - All moments of inertia are positive
// - Torque bounds are symmetric or appropriately defined
// - Time bounds satisfy: 0 < T_min < T_max
// - State bounds allow the maneuver (e.g., q_min ≤ -1 ≤ q_max)
//
// TYPICAL INITIALIZATION SEQUENCE:
// @code
// // Physical properties
// i_x = 1.5;  i_y = 2.0;  i_z = 1.0;
//
// // Control bounds (symmetric)
// tau_min = -1.0;  tau_max = 1.0;
//
// // State bounds
// q_min = -1.0;  q_max = 1.0;
// w_min = -2.0;  w_max = 2.0;
//
// // Time parameters
// T_min = 0.1;  T_max = 10.0;  T_0 = 2.4;
//
// // Problem dimensions (for quaternion formulation)
// n_states = 7;  n_controls = 3;
//
// // Now ready to call cgpops_go()
// @endcode
//
// =============================================================================

// =============================================================================
// END OF HEADER
// =============================================================================

#endif  // __CGPOPS_AUX_EXT_HPP__