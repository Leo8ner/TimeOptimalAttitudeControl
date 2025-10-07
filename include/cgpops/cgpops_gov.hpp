// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao.  All Rights Reserved
//
// cgpops_gov.hpp
// CGPOPS Tool Box
// Declare governing equations of continouos time optimal control problem here
// Time-optimal attitude maneuver problem


#ifndef __CGPOPS_GOV_HPP__
#define __CGPOPS_GOV_HPP__


#include "nlpGlobVarExt.hpp"
#include "cgpopsFuncDec.hpp"
#include <cgpops/cgpopsAuxExt.hpp>


// $#$#$

/*-----------------------------------EndPointFunction-----------------------------------*/

class MinTf : public ObjectiveEq
{
public:
    virtual void eval_eq(double& lhs, double** x0, double** xf, double** q, double** t0,
                         double** tf, double* s, double* e);
    virtual void eval_eq(HyperDual& lhs, HyperDual** x0, HyperDual** xf, HyperDual** q,
                         HyperDual** t0, HyperDual** tf, HyperDual* s, HyperDual* e);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex** x0, Bicomplex** xf, Bicomplex** q,
                         Bicomplex** t0, Bicomplex** tf, Bicomplex* s, Bicomplex* e);
private:
    template <class T> void eq_def(T& lhs, T** x0, T** xf, T** q, T** t0, T** tf, T* s,
                                   T* e);
};

/*----------------------------------PhasePointFunction----------------------------------*/
class QuatNormConstraint : public PathConstraintEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};


class Q0Dot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class Q1Dot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class Q2Dot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class Q3Dot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class WXDot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class WYDot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

class WZDot : public OrdinaryDifferentialEq
{
public:
    virtual void eval_eq(double& lhs, double* x, double* u, double& t, double* s,
                         double* p);
    virtual void eval_eq(HyperDual& lhs, HyperDual* x, HyperDual* u, HyperDual& t,
                         HyperDual* s, HyperDual* p);
    virtual void eval_eq(Bicomplex& lhs, Bicomplex* x, Bicomplex* u, Bicomplex& t,
                         Bicomplex* s, Bicomplex* p);
private:
    template <class T> void eq_def(T& lhs, T* x, T* u, T& t, T* s, T* p);
};

void setGlobalTabularData(void); // Set global tabular data used in problem

#endif

