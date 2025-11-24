/**
 * @file cgpops_gov.hpp
 * @brief Governing equation class declarations for spacecraft attitude optimal control
 * 
 * OVERVIEW:
 * =========
 * This header declares the governing equation classes for the time-optimal spacecraft
 * attitude maneuver problem. These classes define the objective function, path constraints,
 * and differential equations that constitute the continuous-time optimal control problem.
 * 
 * The CGPOPS framework uses these equation classes to automatically compute derivatives
 * via automatic differentiation (HyperDual and Bicomplex number systems), which are
 * required for the nonlinear programming (NLP) solver (IPOPT) to efficiently find
 * optimal solutions.
 * 
 * MATHEMATICAL PROBLEM STRUCTURE:
 * ================================
 * 
 * Objective Function (Mayer Form):
 *   J = tf  (minimize final time)
 * 
 * State Vector (7 components):
 *   x = [q₀, q₁, q₂, q₃, ωx, ωy, ωz]ᵀ
 * 
 * Control Vector (3 components):
 *   u = [τx, τy, τz]ᵀ
 * 
 * Dynamics (7 differential equations):
 *   q̇₀ = 0.5(-ωx·q₁ - ωy·q₂ - ωz·q₃)           [Q0Dot]
 *   q̇₁ = 0.5(ωx·q₀ + ωz·q₂ - ωy·q₃)            [Q1Dot]
 *   q̇₂ = 0.5(ωy·q₀ - ωz·q₁ + ωx·q₃)            [Q2Dot]
 *   q̇₃ = 0.5(ωz·q₀ + ωy·q₁ - ωx·q₂)            [Q3Dot]
 *   ω̇x = (1/Ix)[(Iy-Iz)ωy·ωz + τx]             [WXDot]
 *   ω̇y = (1/Iy)[(Iz-Ix)ωz·ωx + τy]             [WYDot]
 *   ω̇z = (1/Iz)[(Ix-Iy)ωx·ωy + τz]             [WZDot]
 * 
 * Path Constraints (1 constraint):
 *   ||q||² = q₀² + q₁² + q₂² + q₃² = 1         [QuatNormConstraint]
 * 
 * AUTOMATIC DIFFERENTIATION ARCHITECTURE:
 * ========================================
 * Each equation class provides three implementations via function overloading:
 * 
 * 1. double:     Standard evaluation for function values
 * 2. HyperDual:  Computes first and second derivatives simultaneously
 * 3. Bicomplex:  Computes derivatives using complex-step differentiation
 * 
 * This design enables the CGPOPS framework to automatically compute:
 * - Jacobian matrices (first derivatives) for NLP constraint gradients
 * - Hessian matrices (second derivatives) for NLP objective curvature
 * 
 * without requiring manual derivative implementations, ensuring numerical
 * accuracy and reducing implementation errors.
 * 
 * CLASS HIERARCHY:
 * ================
 * All equation classes inherit from CGPOPS framework base classes:
 * - ObjectiveEq:            Base for endpoint cost functions
 * - PathConstraintEq:       Base for algebraic path constraints
 * - OrdinaryDifferentialEq: Base for differential equation right-hand sides
 * 
 * Each derived class implements the Template Method pattern:
 * - Public eval_eq() methods (virtual overrides for each numeric type)
 * - Private eq_def() template method (actual equation implementation)
 * 
 * This pattern separates the automatic differentiation framework from the
 * mathematical equation logic, improving code maintainability and reusability.
 * 
 * IMPLEMENTATION NOTES:
 * =====================
 * - Equation implementations are in cgpops_gov.cpp
 * - External parameters (i_x, i_y, i_z) must be defined before use
 * - All classes use pass-by-reference for efficiency with AD types
 * - Template methods enable compile-time polymorphism for AD types
 * 
 * @copyright Copyright (c) Yunus M. Agamawi and Anil Vithala Rao. All Rights Reserved.
 * @note This file is part of the CGPOPS Tool Box framework.
 * 
 * @see cgpops_gov.cpp for equation implementations
 * @see cgpops_main.hpp for problem setup and solution interface
 * 
 * @author Yunus M. Agamawi, Anil Vithala Rao
 */

#ifndef __CGPOPS_GOV_HPP__
#define __CGPOPS_GOV_HPP__

// =============================================================================
// REQUIRED FRAMEWORK HEADERS
// =============================================================================

#include "nlpGlobVarExt.hpp"        ///< NLP global variables and type definitions
#include "cgpopsFuncDec.hpp"        ///< CGPOPS function declarations
#include <cgpops/cgpopsAuxExt.hpp>  ///< CGPOPS auxiliary declarations (base classes)

// =============================================================================
// OBJECTIVE FUNCTION CLASS
// =============================================================================
// The objective function defines the cost to be minimized. For time-optimal
// control, this is simply the final time tf (Mayer formulation).

/**
 * @class MinTf
 * @brief Minimum time objective function for time-optimal control
 * 
 * Implements the Mayer-form objective function:
 *   J = tf
 * 
 * This objective drives the optimizer to find the fastest possible maneuver
 * that satisfies all constraints and reaches the target state. The minimization
 * of final time is the defining characteristic of time-optimal control problems.
 * 
 * MATHEMATICAL FORMULATION:
 *   Objective: min J(x,u,tf) = tf
 * 
 * This is an endpoint cost that depends only on the final time, not on the
 * trajectory itself (Mayer form rather than Lagrange or Bolza form).
 * 
 * @note Inherits from ObjectiveEq base class which defines the interface for
 *       objective function evaluation with automatic differentiation support.
 * 
 * @see ObjectiveEq Base class defining the objective function interface
 */
class MinTf : public ObjectiveEq
{
public:
    /**
     * @brief Evaluates objective function using standard double precision
     * 
     * @param[out] lhs Objective function value J = tf
     * @param[in] x0 Initial state values (unused in Mayer cost)
     * @param[in] xf Final state values (unused in Mayer cost)
     * @param[in] q Integral cost values (unused in Mayer cost)
     * @param[in] t0 Initial time (unused in Mayer cost)
     * @param[in] tf Final time array
     * @param[in] s Static parameters (unused)
     * @param[in] e Event parameters (unused)
     */
    virtual void eval_eq(double& lhs, double** x0, double** xf, double** q, double** t0,
                         double** tf, double* s, double* e);
    
    /**
     * @brief Evaluates objective function using HyperDual numbers
     * 
     * HyperDual evaluation enables simultaneous computation of first and second
     * derivatives for Hessian matrix construction in the NLP solver.
     * 
     * @param[out] lhs Objective value with derivative information
     * @param[in] x0 Initial state (unused)
     * @param[in] xf Final state (unused)
     * @param[in] q Integral cost (unused)
     * @param[in] t0 Initial time (unused)
     * @param[in] tf Final time with derivative information
     * @param[in] s Static parameters (unused)
     * @param[in] e Event parameters (unused)
     */
    virtual void eval_eq(HyperDual& lhs, HyperDual** x0, HyperDual** xf, HyperDual** q,
                         HyperDual** t0, HyperDual** tf, HyperDual* s, HyperDual* e);
    
    /**
     * @brief Evaluates objective function using Bicomplex numbers
     * 
     * Bicomplex evaluation uses complex-step differentiation for highly accurate
     * derivative computation without subtraction errors.
     * 
     * @param[out] lhs Objective value with derivative information
     * @param[in] x0 Initial state (unused)
     * @param[in] xf Final state (unused)
     * @param[in] q Integral cost (unused)
     * @param[in] t0 Initial time (unused)
     * @param[in] tf Final time with derivative information
     * @param[in] s Static parameters (unused)
     * @param[in] e Event parameters (unused)
     */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex** x0, Bicomplex** xf, Bicomplex** q,
                         Bicomplex** t0, Bicomplex** tf, Bicomplex* s, Bicomplex* e);

private:
    /**
     * @brief Template method containing actual objective function implementation
     * 
     * This template method is instantiated for double, HyperDual, and Bicomplex
     * types, enabling automatic differentiation through templated arithmetic.
     * 
     * @tparam T Numeric type (double, HyperDual, or Bicomplex)
     */
    template <class T> void eq_def(T& lhs, T** x0, T** xf, T** q, T** t0, T** tf, T* s, T* e);
};

// =============================================================================
// PATH CONSTRAINT CLASSES
// =============================================================================
// Path constraints are algebraic equations that must be satisfied at all points
// along the trajectory (not just at endpoints).

/**
 * @class QuatNormConstraint
 * @brief Quaternion normalization path constraint
 * 
 * Enforces the fundamental quaternion unit norm constraint along the entire
 * trajectory:
 *   ||q||² = q₀² + q₁² + q₂² + q₃² = 1
 * 
 * PHYSICAL INTERPRETATION:
 * Quaternions represent rotations and must maintain unit norm to be valid.
 * This constraint ensures the attitude representation remains physically
 * meaningful throughout the maneuver.
 * 
 * CONSTRAINT FORMULATION:
 *   g(x) = q₀² + q₁² + q₂² + q₃² - 1 = 0
 * 
 * This is enforced as an equality constraint at every collocation point,
 * preventing quaternion drift due to numerical integration errors.
 * 
 * @note While quaternion normalization could theoretically be maintained through
 *       the dynamics alone, explicit enforcement as a path constraint improves
 *       numerical stability and solver robustness.
 * 
 * @see PathConstraintEq Base class for algebraic path constraints
 */
class QuatNormConstraint : public PathConstraintEq
{
public:
    /**
     * @brief Evaluates quaternion norm constraint using double precision
     * 
     * @param[out] lhs Constraint value (should equal 1.0)
     * @param[in] x State vector [q₀, q₁, q₂, q₃, ωx, ωy, ωz]
     * @param[in] u Control vector (unused in this constraint)
     * @param[in] t Current time (unused - autonomous constraint)
     * @param[in] s Static parameters (unused)
     * @param[in] p Dynamic parameters (unused)
     */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /**
     * @brief Evaluates quaternion norm constraint using HyperDual numbers
     * 
     * @param[out] lhs Constraint value with derivative information
     * @param[in] x State vector with derivative information
     * @param[in] u Control vector (unused)
     * @param[in] t Current time (unused)
     * @param[in] s Static parameters (unused)
     * @param[in] p Dynamic parameters (unused)
     */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /**
     * @brief Evaluates quaternion norm constraint using Bicomplex numbers
     * 
     * @param[out] lhs Constraint value with derivative information
     * @param[in] x State vector with derivative information
     * @param[in] u Control vector (unused)
     * @param[in] t Current time (unused)
     * @param[in] s Static parameters (unused)
     * @param[in] p Dynamic parameters (unused)
     */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /**
     * @brief Template method for quaternion norm constraint implementation
     * 
     * @tparam T Numeric type (double, HyperDual, or Bicomplex)
     */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

// =============================================================================
// QUATERNION KINEMATICS CLASSES
// =============================================================================
// These four classes implement the quaternion kinematic differential equations
// that relate quaternion rates to angular velocity. Together they describe how
// the attitude (quaternion) changes as the spacecraft rotates.
//
// Derivation: q̇ = 0.5 * Ω(ω) * q, where Ω is the skew-symmetric matrix of ω

/**
 * @class Q0Dot
 * @brief Time derivative of quaternion scalar component
 * 
 * Implements the kinematic equation for the scalar part of the quaternion:
 *   q̇₀ = 0.5 * (-ωx·q₁ - ωy·q₂ - ωz·q₃)
 * 
 * PHYSICAL INTERPRETATION:
 * The scalar component q₀ represents the "amount" of rotation. This equation
 * shows how it changes based on the angular velocity and the vector part of
 * the quaternion. When angular velocity is zero, q₀ remains constant.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₁ = f₁(x,u) = -0.5(ωx·q₁ + ωy·q₂ + ωz·q₃)
 * 
 * This is the first of seven ordinary differential equations in the state dynamics.
 * 
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class Q0Dot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates q̇₀ using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates q̇₀ using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates q̇₀ using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for q̇₀ equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

/**
 * @class Q1Dot
 * @brief Time derivative of first quaternion vector component
 * 
 * Implements the kinematic equation for the x-component of the quaternion vector:
 *   q̇₁ = 0.5 * (ωx·q₀ + ωz·q₂ - ωy·q₃)
 * 
 * PHYSICAL INTERPRETATION:
 * The component q₁ represents rotation about the x-axis of the body frame.
 * This equation couples rotation rates in all three axes through the
 * quaternion structure.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₂ = f₂(x,u) = 0.5(ωx·q₀ + ωz·q₂ - ωy·q₃)
 * 
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class Q1Dot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates q̇₁ using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates q̇₁ using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates q̇₁ using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for q̇₁ equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

/**
 * @class Q2Dot
 * @brief Time derivative of second quaternion vector component
 * 
 * Implements the kinematic equation for the y-component of the quaternion vector:
 *   q̇₂ = 0.5 * (ωy·q₀ - ωz·q₁ + ωx·q₃)
 * 
 * PHYSICAL INTERPRETATION:
 * The component q₂ represents rotation about the y-axis of the body frame.
 * The coupling terms show how rotation about one axis affects the others
 * in the quaternion representation.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₃ = f₃(x,u) = 0.5(ωy·q₀ - ωz·q₁ + ωx·q₃)
 * 
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class Q2Dot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates q̇₂ using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates q̇₂ using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates q̇₂ using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for q̇₂ equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

/**
 * @class Q3Dot
 * @brief Time derivative of third quaternion vector component
 * 
 * Implements the kinematic equation for the z-component of the quaternion vector:
 *   q̇₃ = 0.5 * (ωz·q₀ + ωy·q₁ - ωx·q₂)
 * 
 * PHYSICAL INTERPRETATION:
 * The component q₃ represents rotation about the z-axis of the body frame.
 * Together with Q0Dot, Q1Dot, and Q2Dot, this completes the quaternion
 * kinematic description of attitude motion.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₄ = f₄(x,u) = 0.5(ωz·q₀ + ωy·q₁ - ωx·q₂)
 * 
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class Q3Dot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates q̇₃ using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates q̇₃ using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates q̇₃ using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for q̇₃ equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

// =============================================================================
// EULER'S ROTATIONAL EQUATIONS CLASSES
// =============================================================================
// These three classes implement Euler's equations for rigid body rotational
// dynamics in the body-fixed frame. They describe how applied torques and
// gyroscopic coupling affect angular velocity.
//
// General form: I·ω̇ + ω × (I·ω) = τ

/**
 * @class WXDot
 * @brief Angular acceleration about x-axis (Euler's equation)
 * 
 * Implements Euler's rotational equation for the x-axis:
 *   ω̇x = (1/Ix) * [(Iy - Iz)·ωy·ωz + τx]
 * 
 * PHYSICAL INTERPRETATION:
 * This equation describes how the angular velocity about the x-axis changes
 * due to two effects:
 * 1. Gyroscopic coupling: (Iy - Iz)·ωy·ωz
 *    When the spacecraft rotates about y and z simultaneously, conservation
 *    of angular momentum in the non-inertial body frame creates a torque
 *    about the x-axis. The magnitude depends on the inertia distribution.
 * 
 * 2. Applied control torque: τx
 *    Direct control input accelerates rotation about x-axis.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₅ = f₅(x,u) = (1/Ix)[(Iy - Iz)·ωy·ωz + τx]
 * 
 * @note Requires external parameter i_x (x-axis moment of inertia, kg⋅m²)
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class WXDot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates ω̇x using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates ω̇x using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates ω̇x using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for ω̇x equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

/**
 * @class WYDot
 * @brief Angular acceleration about y-axis (Euler's equation)
 * 
 * Implements Euler's rotational equation for the y-axis:
 *   ω̇y = (1/Iy) * [(Iz - Ix)·ωz·ωx + τy]
 * 
 * PHYSICAL INTERPRETATION:
 * Describes angular acceleration about y-axis from:
 * 1. Gyroscopic coupling: (Iz - Ix)·ωz·ωx
 *    Simultaneous rotation about z and x axes induces acceleration about y.
 * 
 * 2. Applied control torque: τy
 *    Direct control input about y-axis.
 * 
 * The gyroscopic term explains phenomena like how a spinning top precesses
 * when tilted - rotation about one axis induces motion about perpendicular axes.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₆ = f₆(x,u) = (1/Iy)[(Iz - Ix)·ωz·ωx + τy]
 * 
 * @note Requires external parameter i_y (y-axis moment of inertia, kg⋅m²)
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class WYDot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates ω̇y using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates ω̇y using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates ω̇y using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for ω̇y equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

/**
 * @class WZDot
 * @brief Angular acceleration about z-axis (Euler's equation)
 * 
 * Implements Euler's rotational equation for the z-axis:
 *   ω̇z = (1/Iz) * [(Ix - Iy)·ωx·ωy + τz]
 * 
 * PHYSICAL INTERPRETATION:
 * Describes angular acceleration about z-axis from:
 * 1. Gyroscopic coupling: (Ix - Iy)·ωx·ωy
 *    Simultaneous rotation about x and y axes induces acceleration about z.
 *    The coupling strength depends on the inertia asymmetry (Ix - Iy).
 * 
 * 2. Applied control torque: τz
 *    Direct control input about z-axis.
 * 
 * Together with WXDot and WYDot, this completes Euler's equations describing
 * how the spacecraft's rotational motion evolves under applied torques and
 * gyroscopic effects.
 * 
 * MATHEMATICAL FORM:
 *   ẋ₇ = f₇(x,u) = (1/Iz)[(Ix - Iy)·ωx·ωy + τz]
 * 
 * @note Requires external parameter i_z (z-axis moment of inertia, kg⋅m²)
 * @see OrdinaryDifferentialEq Base class for ODE right-hand sides
 */
class WZDot : public OrdinaryDifferentialEq
{
public:
    /** @brief Evaluates ω̇z using double precision */
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p);
    
    /** @brief Evaluates ω̇z using HyperDual numbers for automatic differentiation */
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    
    /** @brief Evaluates ω̇z using Bicomplex numbers for automatic differentiation */
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);

private:
    /** @brief Template method for ω̇z equation implementation */
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * @brief Initializes global tabular data for the problem
 * 
 * This function is called during problem setup to load any pre-computed lookup
 * tables or global data structures required by the governing equations.
 * 
 * For the spacecraft attitude problem with analytical equations, this function
 * is currently unused but provided for framework compatibility. It could be
 * used in future extensions that require tabulated data (e.g., aerodynamic
 * coefficients, gravity models, or environmental disturbances).
 * 
 * @note Must be called before solving the optimal control problem
 * @note Currently has no implementation for analytical attitude dynamics
 * 
 * @see cgpops_gov.cpp for implementation
 */
void setGlobalTabularData(void);

// =============================================================================
// END OF HEADER
// =============================================================================

#endif  // __CGPOPS_GOV_HPP__