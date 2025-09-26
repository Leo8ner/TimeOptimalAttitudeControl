// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao.  All Rights Reserved
//
// cgpops_main_test.cpp
// Test Main
// Test CGPOPS functions
//

#include "cgpops_main.hpp"


void cgpops_go(doubleMat& cgpopsResults)
{
    // Define Global Variables used to determine problem size
    PG      = 1;    // Number of phases in problem
    nsG     = 0;    // Number of static parameters in problem
    nbG     = 0;    // Number of event constraints in problem
    nepG    = 0;    // Number of endpoint parameters in problem

    // Allocate memory for each phase
    initGlobalVars();

    // Define number of components for each parameter in problem phases
    // Phase 1 parameters
    nxG[0]  = 7;    // Number of state components in phase 1 [q0, q1, q2, q3, wx, wy, wz]
    nuG[0]  = 3;    // Number of control components in phase 1 [u1, u2, u3]
    nqG[0]  = 0;    // Number of integral constraints in phase 1
    ncG[0]  = 1;    // Number of path constraints in phase 1 [||q|| = 1]
    nppG[0] = 0;    // Number of phase parameters in phase 1
    
    setGlobalTabularData(); // Set any global tabular data used in problem
    
    // Define mesh grids for each phase in problem for LGR collocation
    // Phase 1 mesh grid
    int M1 = numintervalsG;         // Number of intervals used in phase 1 mesh
    int initcolpts = initcolptsG;   // Initial number of collocation points in each
                                    // interval
    double fraction1[M1];           // Allocate memory for fraction vector for phase 1
                                    // mesh
    int colpoints1[M1];             // Allocate memory for colpoints vector for phase 1
                                    // mesh
    for (int m=0; m<M1; m++)
    {
        fraction1[m] = 1.0/((double) M1);
        colpoints1[m] = initcolpts;
    }
    setRPMDG(0,M1,fraction1,colpoints1);

    // Set information for transcribed NLP resulting from LGR collocation using defined
    // mesh grid
    setInfoNLPG();

    /*------------------------------Provide Problem Bounds------------------------------*/
    // Phase 1
    double t0min = 0,                   t0max = 0;              // minimum and maximum initial times
    double tfmin = T_min,               tfmax = T_max;          // minimum and maximum final times

    // Initial attitude (quaternion) - identity quaternion for initial rest
    double q0_0 = 1.0, q1_0 = 0.0, q2_0 = 0.0, q3_0 = 0.0;
    // Final attitude (quaternion) - define your target orientation
    double q0_f = 0.0, q1_f = 0.0, q2_f = 0.0, q3_f = 1.0;  // 180 degree rotation about z-axis

    // Initial and final angular velocities (typically zero for rest-to-rest)
    double w1_0 = 0.0, w2_0 = 0.0, w3_0 = 0.0;
    double w1_f = 0.0, w2_f = 0.0, w3_f = 0.0;

    // State bounds during trajectory
    double qmin = q_min,             qmax = q_max;            // quaternion component bounds
    double wmin = w_min,             wmax = w_max;            // angular velocity bounds (rad/s)

    // Control bounds (torque limits)
    double torquemin = tau_min,        torquemax = tau_max;        // torque bounds (N⋅m)

    // Path constraint bounds (quaternion normalization)
    double normmin = 1.0,           normmax = 1.0;          // ||q|| = 1

    // Define Bounds of Optimal Control Problem
    // Phase 1 bounds
    int phase1 = 0;
    double x0l1[nxG[phase1]],   x0u1[nxG[phase1]];
    double xfl1[nxG[phase1]],   xfu1[nxG[phase1]];
    double xl1[nxG[phase1]],    xu1[nxG[phase1]];
    double ul1[nuG[phase1]],    uu1[nuG[phase1]];
    double ql1[nqG[phase1]],    qu1[nqG[phase1]];
    double cl1[ncG[phase1]],    cu1[ncG[phase1]];
    double t0l1,    t0u1;
    double tfl1,    tfu1;

    // Initial state bounds (quaternion + angular velocity)
    x0l1[0] = q0_0;   x0l1[1] = q1_0;   x0l1[2] = q2_0;   x0l1[3] = q3_0;
    x0l1[4] = w1_0;   x0l1[5] = w2_0;   x0l1[6] = w3_0;
    x0u1[0] = q0_0;   x0u1[1] = q1_0;   x0u1[2] = q2_0;   x0u1[3] = q3_0;
    x0u1[4] = w1_0;   x0u1[5] = w2_0;   x0u1[6] = w3_0;

    // Final state bounds
    xfl1[0] = q0_f;   xfl1[1] = q1_f;   xfl1[2] = q2_f;   xfl1[3] = q3_f;
    xfl1[4] = w1_f;   xfl1[5] = w2_f;   xfl1[6] = w3_f;
    xfu1[0] = q0_f;   xfu1[1] = q1_f;   xfu1[2] = q2_f;   xfu1[3] = q3_f;
    xfu1[4] = w1_f;   xfu1[5] = w2_f;   xfu1[6] = w3_f;

    // State bounds during trajectory
    xl1[0] = qmin;    xl1[1] = qmin;    xl1[2] = qmin;    xl1[3] = qmin;
    xl1[4] = wmin;    xl1[5] = wmin;    xl1[6] = wmin;
    xu1[0] = qmax;    xu1[1] = qmax;    xu1[2] = qmax;    xu1[3] = qmax;
    xu1[4] = wmax;    xu1[5] = wmax;    xu1[6] = wmax;

    // Control bounds (torques)
    ul1[0] = torquemin;   ul1[1] = torquemin;   ul1[2] = torquemin;
    uu1[0] = torquemax;   uu1[1] = torquemax;   uu1[2] = torquemax;

    // Path constraint bounds (quaternion normalization)
    cl1[0] = normmin;
    cu1[0] = normmax;

    // Time bounds
    t0l1 = t0min;   t0u1 = t0max;
    tfl1 = tfmin;   tfu1 = tfmax;
    
    // Set parameterized constructor for NLP Phase Bounds Class
    setNLPPBG(phase1,x0l1,x0u1,xfl1,xfu1,xl1,xu1,ul1,uu1,ql1,qu1,cl1,cu1,t0l1,t0u1,tfl1,
              tfu1);
    
    // Whole problem bounds
    double sl[nsG], su[nsG];
    double bl[nbG], bu[nbG];

    // Set parameterized constructor for NLP Whole Bounds Class
    setNLPWBG(sl,su,bl,bu);
    
    // Provide initial guess for NLP Solver
    // Phase 1 guess
    double x0g1[nxG[phase1]],   xfg1[nxG[phase1]];
    double u0g1[nuG[phase1]],   ufg1[nuG[phase1]];
    double qg1[nqG[phase1]];
    double t0g1,    tfg1;

    // Initial state guess (quaternion + angular velocities)
    x0g1[0] = q0_0;   x0g1[1] = q1_0;   x0g1[2] = q2_0;   x0g1[3] = q3_0;
    x0g1[4] = w1_0;   x0g1[5] = w2_0;   x0g1[6] = w3_0;

    // Final state guess
    xfg1[0] = q0_f;   xfg1[1] = q1_f;   xfg1[2] = q2_f;   xfg1[3] = q3_f;
    xfg1[4] = w1_f;   xfg1[5] = w2_f;   xfg1[6] = w3_f;

    // Control guess (torques)
    u0g1[0] = 0.1;    u0g1[1] = 0.1;    u0g1[2] = 0.1;
    ufg1[0] = -0.1;   ufg1[1] = -0.1;   ufg1[2] = -0.1;

    // Time guess
    t0g1 = 0;
    tfg1 = 2.4;  // Guess 2.4 seconds for rest-to-rest 180 deg rotation
    
    // Set parameterized constructor for NLP Phase Guess Class
    setNLPPGG(phase1,x0g1,xfg1,u0g1,ufg1,qg1,t0g1,tfg1);
    
    // Whole problem guess
    double sg[nsG];

    // Set parameterized constructor for NLP Phase Guess Class
    setNLPWGG(sg);
    
    // Set ocp functions
    objEqG = new MinTf;
    pthEqVecG = {{new QuatNormConstraint}};  // For quaternion normalization path constraint
    odeEqVecG = {{new Q0Dot, new Q1Dot, new Q2Dot, new Q3Dot, new WXDot, new WYDot, new WZDot}};
    
    // Make call to CGOPS handler for IPOPT using user provided output settings
    CGPOPS_IPOPT_caller(cgpopsResults);
    
}

