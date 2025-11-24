/**
 * @file cgpops_gov.cpp
 * @brief Governing equations for time-optimal spacecraft attitude maneuver problem
 * 
 * This file implements the continuous-time optimal control problem for minimum-time
 * spacecraft attitude reorientation using quaternion representation. The problem
 * formulates a time-optimal control strategy for rotating a rigid spacecraft from
 * an initial attitude to a desired final attitude while respecting torque constraints.
 * 
 * MATHEMATICAL FORMULATION:
 * ========================
 * 
 * Objective:
 *   Minimize J = tf (final time)
 * 
 * State Variables (7):
 *   x = [q0, q1, q2, q3, ωx, ωy, ωz]ᵀ
 *   where:
 *     q = [q0, q1, q2, q3]ᵀ : Unit quaternion representing attitude
 *     ω = [ωx, ωy, ωz]ᵀ    : Angular velocity in body frame (rad/s)
 * 
 * Control Variables (3):
 *   u = [τx, τy, τz]ᵀ : Applied torques in body frame (N⋅m)
 * 
 * Dynamics:
 *   Quaternion kinematics:
 *     q̇ = 0.5 * Ω(ω) * q
 *     where Ω(ω) is the skew-symmetric matrix of angular velocity
 * 
 *   Euler's rotational equations:
 *     Iẋ ωx = (Iy - Iz)ωy ωz + τx
 *     Iy ωy = (Iz - Ix)ωz ωx + τy
 *     Iz ωz = (Ix - Iy)ωx ωy + τz
 * 
 * Path Constraints:
 *   ||q||² = q₀² + q₁² + q₂² + q₃² = 1  (quaternion normalization)
 * 
 * COORDINATE FRAME:
 * =================
 * Body-fixed coordinate frame is used for angular velocities and torques.
 * Quaternion represents rotation from inertial to body frame.
 * 
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 */

#include <cgpops/cgpops_gov.hpp>

// =============================================================================
// STATE AND CONTROL VARIABLE ACCESS MACROS
// =============================================================================
// These macros provide convenient access to state and control variables within
// the equation definitions. While modern C++ prefers inline functions, these
// macros are used for consistency with the CGPOPS framework conventions.

// Quaternion components (scalar-first convention: q = [q0, q1, q2, q3])
#define qo       x[0]  ///< Quaternion scalar component (q₀)
#define q1       x[1]  ///< Quaternion vector component (q₁)
#define q2       x[2]  ///< Quaternion vector component (q₂)
#define q3       x[3]  ///< Quaternion vector component (q₃)

// Angular velocity components in body frame (rad/s)
#define wx       x[4]  ///< Angular velocity about x-axis (ωx)
#define wy       x[5]  ///< Angular velocity about y-axis (ωy)
#define wz       x[6]  ///< Angular velocity about z-axis (ωz)

// Control torque components in body frame (N⋅m)
#define tx       u[0]  ///< Applied torque about x-axis (τx)
#define ty       u[1]  ///< Applied torque about y-axis (τy)
#define tz       u[2]  ///< Applied torque about z-axis (τz)

// Final time (optimization variable)
#define tf_      tf[0][0]  ///< Final time (to be minimized)

// =============================================================================
// PHYSICAL CONSTANTS
// =============================================================================
// Note: Inertia values (i_x, i_y, i_z) are defined externally and set via
// the CGPOPS problem configuration. They represent principal moments of
// inertia about the body-fixed coordinate axes (kg⋅m²).

// =============================================================================
// OBJECTIVE FUNCTION
// =============================================================================

/**
 * @brief Minimum time objective function
 * 
 * Implements the cost functional J = tf for time-optimal control.
 * The optimizer minimizes the final time tf to achieve the fastest
 * possible attitude maneuver.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex for automatic differentiation)
 * @param[out] lhs Left-hand side of the equation (objective value)
 * @param[in] x0 Initial state values (unused for Mayer cost)
 * @param[in] xf Final state values (unused for Mayer cost)
 * @param[in] q Integral cost values (unused for Mayer cost)
 * @param[in] t0 Initial time (unused for Mayer cost)
 * @param[in] tf Final time array
 * @param[in] s Static parameters (unused)
 * @param[in] e Event parameters (unused)
 */
template <class T>
void MinTf::eq_def(T& lhs, T** x0, T** xf, T** q, T** t0, T** tf,
                   T* s, T* e)
{
    lhs = tf_;  // Objective: minimize final time
}

// =============================================================================
// PATH CONSTRAINTS
// =============================================================================

/**
 * @brief Quaternion normalization constraint
 * 
 * Enforces the unit quaternion constraint ||q||² = 1 along the trajectory.
 * This is a fundamental requirement for valid attitude representation using
 * quaternions. The constraint is enforced as a path constraint rather than
 * being integrated into the dynamics to maintain numerical stability.
 * 
 * Mathematical form: q₀² + q₁² + q₂² + q₃² = 1
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Left-hand side (should equal 1.0)
 * @param[in] x State vector [q0, q1, q2, q3, ωx, ωy, ωz]
 * @param[in] u Control vector [τx, τy, τz] (unused)
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 */
template <class T>
void QuatNormConstraint::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    // Compute quaternion norm squared: ||q||² = q₀² + q₁² + q₂² + q₃²
    lhs = qo*qo + q1*q1 + q2*q2 + q3*q3;
}

// =============================================================================
// QUATERNION KINEMATICS
// =============================================================================
// The following four functions implement the quaternion kinematic differential
// equations that relate the time derivative of the quaternion to the angular
// velocity vector. These equations are derived from:
//
//   q̇ = 0.5 * Ω(ω) * q
//
// where Ω(ω) is the 4×4 skew-symmetric matrix:
//
//        ⎡  0  -ωx  -ωy  -ωz ⎤
//   Ω = ⎢ ωx    0   ωz  -ωy ⎥
//        ⎢ ωy  -ωz    0   ωx ⎥
//        ⎣ ωz   ωy  -ωx    0 ⎦
//
// Expanding the matrix multiplication yields the four scalar equations below.

/**
 * @brief Time derivative of quaternion scalar component
 * 
 * Implements: q̇₀ = 0.5 * (-ωx*q₁ - ωy*q₂ - ωz*q₃)
 * 
 * This is the scalar component of the quaternion kinematic equation,
 * representing how the "amount" of rotation changes with angular velocity.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative q̇₀
 * @param[in] x State vector containing quaternion and angular velocity
 * @param[in] u Control vector (unused in kinematics)
 * @param[in] t Current time (unused - autonomous system)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 */
template <class T>
void Q0Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5 * (-wx*q1 - wy*q2 - wz*q3);
}

/**
 * @brief Time derivative of quaternion vector component q₁
 * 
 * Implements: q̇₁ = 0.5 * (ωx*q₀ + ωz*q₂ - ωy*q₃)
 * 
 * First vector component of quaternion kinematics, representing rotation
 * contribution about the x-axis of the body frame.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative q̇₁
 * @param[in] x State vector
 * @param[in] u Control vector (unused)
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 */
template <class T>
void Q1Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5 * (wx*qo + wz*q2 - wy*q3);
}

/**
 * @brief Time derivative of quaternion vector component q₂
 * 
 * Implements: q̇₂ = 0.5 * (ωy*q₀ - ωz*q₁ + ωx*q₃)
 * 
 * Second vector component of quaternion kinematics, representing rotation
 * contribution about the y-axis of the body frame.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative q̇₂
 * @param[in] x State vector
 * @param[in] u Control vector (unused)
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 */
template <class T>
void Q2Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5 * (wy*qo - wz*q1 + wx*q3);
}

/**
 * @brief Time derivative of quaternion vector component q₃
 * 
 * Implements: q̇₃ = 0.5 * (ωz*q₀ + ωy*q₁ - ωx*q₂)
 * 
 * Third vector component of quaternion kinematics, representing rotation
 * contribution about the z-axis of the body frame.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative q̇₃
 * @param[in] x State vector
 * @param[in] u Control vector (unused)
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 */
template <class T>
void Q3Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5 * (wz*qo + wy*q1 - wx*q2);
}

// =============================================================================
// EULER'S ROTATIONAL EQUATIONS OF MOTION
// =============================================================================
// The following three functions implement Euler's equations for rigid body
// rotational dynamics in the body-fixed frame. These equations describe how
// applied torques and gyroscopic coupling affect angular velocity.
//
// General form: I·ω̇ + ω × (I·ω) = τ
//
// For a principal axis coordinate system (Ixy = Iyz = Ixz = 0):
//   Ix·ω̇x = (Iy - Iz)·ωy·ωz + τx
//   Iy·ω̇y = (Iz - Ix)·ωz·ωx + τy
//   Iz·ω̇z = (Ix - Iy)·ωx·ωy + τz
//
// The gyroscopic terms (Iy - Iz)·ωy·ωz, etc., represent coupling between
// rotation rates due to the spacecraft's moment of inertia distribution.

/**
 * @brief Time derivative of angular velocity about x-axis
 * 
 * Implements Euler's equation for x-axis rotation:
 *   ω̇x = (1/Ix) * [(Iy - Iz)·ωy·ωz + τx]
 * 
 * The gyroscopic term (Iy - Iz)·ωy·ωz represents the coupling effect where
 * rotations about y and z axes induce acceleration about the x-axis due to
 * conservation of angular momentum in the non-inertial body frame.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative ω̇x (rad/s²)
 * @param[in] x State vector containing angular velocities
 * @param[in] u Control vector containing applied torques
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 * 
 * @note i_x is defined externally as the x-axis moment of inertia (kg⋅m²)
 */
template <class T>
void WXDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    // ω̇x = (1/Ix) * [gyroscopic coupling + applied torque]
    lhs = (1.0 / i_x) * ((i_y - i_z) * wy * wz + tx);
}

/**
 * @brief Time derivative of angular velocity about y-axis
 * 
 * Implements Euler's equation for y-axis rotation:
 *   ω̇y = (1/Iy) * [(Iz - Ix)·ωz·ωx + τy]
 * 
 * The gyroscopic term (Iz - Ix)·ωz·ωx represents the coupling effect where
 * rotations about z and x axes induce acceleration about the y-axis.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative ω̇y (rad/s²)
 * @param[in] x State vector containing angular velocities
 * @param[in] u Control vector containing applied torques
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 * 
 * @note i_y is defined externally as the y-axis moment of inertia (kg⋅m²)
 */
template <class T>
void WYDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    // ω̇y = (1/Iy) * [gyroscopic coupling + applied torque]
    lhs = (1.0 / i_y) * ((i_z - i_x) * wx * wz + ty);
}

/**
 * @brief Time derivative of angular velocity about z-axis
 * 
 * Implements Euler's equation for z-axis rotation:
 *   ω̇z = (1/Iz) * [(Ix - Iy)·ωx·ωy + τz]
 * 
 * The gyroscopic term (Ix - Iy)·ωx·ωy represents the coupling effect where
 * rotations about x and y axes induce acceleration about the z-axis.
 * 
 * @tparam T Numeric type (double, HyperDual, or Bicomplex)
 * @param[out] lhs Time derivative ω̇z (rad/s²)
 * @param[in] x State vector containing angular velocities
 * @param[in] u Control vector containing applied torques
 * @param[in] t Current time (unused)
 * @param[in] s Static parameters (unused)
 * @param[in] p Dynamic parameters (unused)
 * 
 * @note i_z is defined externally as the z-axis moment of inertia (kg⋅m²)
 */
template <class T>
void WZDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    // ω̇z = (1/Iz) * [gyroscopic coupling + applied torque]
    lhs = (1.0 / i_z) * ((i_x - i_y) * wx * wy + tz);
}

// =============================================================================
// TEMPLATE INSTANTIATIONS FOR AUTOMATIC DIFFERENTIATION
// =============================================================================
// The CGPOPS framework requires explicit instantiations for three numeric types:
//   - double:     Standard floating-point evaluation
//   - HyperDual:  First and second derivative computation
//   - Bicomplex:  Complex-step derivative computation
//
// These instantiations enable automatic differentiation for computing Jacobians
// and Hessians required by the NLP solver.

// -----------------------------------------------------------------------------
// Objective Function Instantiations
// -----------------------------------------------------------------------------

void MinTf::eval_eq(double& lhs, double** x0, double** xf, double** q, double** t0,
                    double** tf, double* s, double* e)
{
    eq_def(lhs, x0, xf, q, t0, tf, s, e);
}

void MinTf::eval_eq(HyperDual& lhs, HyperDual** x0, HyperDual** xf, HyperDual** q,
                    HyperDual** t0, HyperDual** tf, HyperDual* s, HyperDual* e)
{
    eq_def(lhs, x0, xf, q, t0, tf, s, e);
}

void MinTf::eval_eq(Bicomplex& lhs, Bicomplex** x0, Bicomplex** xf, Bicomplex** q,
                    Bicomplex** t0, Bicomplex** tf, Bicomplex* s, Bicomplex* e)
{
    eq_def(lhs, x0, xf, q, t0, tf, s, e);
}

// -----------------------------------------------------------------------------
// Path Constraint Instantiations
// -----------------------------------------------------------------------------

void QuatNormConstraint::eval_eq(double& lhs, double* x, double* u, double& t,
                                  double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void QuatNormConstraint::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u,
                                  HyperDual& t, HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void QuatNormConstraint::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u,
                                  Bicomplex& t, Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

// -----------------------------------------------------------------------------
// Quaternion Kinematics Instantiations
// -----------------------------------------------------------------------------

void Q0Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q0Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q0Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q1Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q1Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q1Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q2Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q2Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q2Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q3Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q3Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void Q3Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

// -----------------------------------------------------------------------------
// Euler's Equations Instantiations
// -----------------------------------------------------------------------------

void WXDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WXDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WXDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WYDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WYDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WYDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WZDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WZDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                    HyperDual* s, HyperDual* p)
{
    eq_def(lhs, x, u, t, s, p);
}

void WZDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                    Bicomplex* s, Bicomplex* p)
{
    eq_def(lhs, x, u, t, s, p);
}

// =============================================================================
// GLOBAL DATA INITIALIZATION
// =============================================================================

/**
 * @brief Initializes global tabular data for the problem
 * 
 * This function is called during problem setup to load any pre-computed
 * lookup tables or global data structures required by the governing equations.
 * 
 * @note Currently unused for this problem as all equations are analytical.
 *       This function is provided for CGPOPS framework compatibility.
 */
void setGlobalTabularData(void)
{
    // No tabular data required for analytical attitude dynamics
}