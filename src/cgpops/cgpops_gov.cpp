// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao.  All Rights Reserved
//
// cgpops_gov.cpp
// CGPOPS Tool Box
// Define governing equations of continouos time optimal control problem here
// Time-optimal attitude maneuver problem


#include "cgpops_gov.hpp"

#define qo       x[0]
#define q1       x[1]
#define q2       x[2]
#define q3       x[3]
#define wx       x[4]
#define wy       x[5]
#define wz       x[6]
#define tx       u[0]
#define ty       u[1]
#define tz       u[2]
#define tf_     tf[0][0]

template <class T> void MinTf::eq_def(T& lhs, T** x0, T** xf, T** q, T** t0, T** tf,
                                      T* s, T* e)
{
    lhs = tf_;
}
template <class T> void QuatNormConstraint::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = qo*qo + q1*q1 + q2*q2 + q3*q3;
}
template <class T> void Q0Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5*(-wx*q1 - wy*q2 - wz*q3);
}
template <class T> void Q1Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5*(wx*qo + wz*q2 - wy*q3);
}
template <class T> void Q2Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5*(wy*qo - wz*q1 + wx*q3);
}
template <class T> void Q3Dot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 0.5*(wz*qo + wy*q1 - wx*q2);
}
template <class T> void WXDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 1.0/Ix*((Iy - Iz)*wy*wz + tx);
}
template <class T> void WYDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 1.0/Iy*((Iz - Ix)*wx*wz + ty);
}
template <class T> void WZDot::eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p)
{
    lhs = 1.0/Iz*((Ix - Iy)*wx*wy + tz);
}

void MinTf::eval_eq(double& lhs, double** x0, double** xf, double** q, double** t0,
                    double** tf, double* s, double* e)
{eq_def(lhs,x0,xf,q,t0,tf,s,e);}
void MinTf::eval_eq(HyperDual& lhs, HyperDual** x0, HyperDual** xf, HyperDual** q,
                    HyperDual** t0, HyperDual** tf, HyperDual* s, HyperDual* e)
{eq_def(lhs,x0,xf,q,t0,tf,s,e);}
void MinTf::eval_eq(Bicomplex& lhs, Bicomplex** x0, Bicomplex** xf, Bicomplex** q,
                    Bicomplex** t0, Bicomplex** tf, Bicomplex* s, Bicomplex* e)
{eq_def(lhs,x0,xf,q,t0,tf,s,e);}
void QuatNormConstraint::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{eq_def(lhs,x,u,t,s,p);}
void QuatNormConstraint::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                     HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void QuatNormConstraint::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                     Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void Q0Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                        double* p)
{eq_def(lhs,x,u,t,s,p);}
void Q0Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                        HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void Q0Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                        Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void Q1Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                           double* p)
{eq_def(lhs,x,u,t,s,p);}
void Q1Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                           HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void Q1Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                           Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void Q2Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{eq_def(lhs,x,u,t,s,p);}
void Q2Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                     HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void Q2Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                     Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void Q3Dot::eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                              double* p)
{eq_def(lhs,x,u,t,s,p);}
void Q3Dot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                              HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void Q3Dot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                              Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void WXDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{eq_def(lhs,x,u,t,s,p);}
void WXDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t, HyperDual* s,
                   HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void WXDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t, Bicomplex* s,
                   Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void WYDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{eq_def(lhs,x,u,t,s,p);}
void WYDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t, HyperDual* s,
                   HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void WYDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t, Bicomplex* s,
                   Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}
void WZDot::eval_eq(double& lhs, double* x, double* u, double& t, double* s, double* p)
{eq_def(lhs,x,u,t,s,p);}
void WZDot::eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                     HyperDual* s, HyperDual* p)
{eq_def(lhs,x,u,t,s,p);}
void WZDot::eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                     Bicomplex* s, Bicomplex* p)
{eq_def(lhs,x,u,t,s,p);}

void setGlobalTabularData(void) // Set global tabular data used in problem
{
}



