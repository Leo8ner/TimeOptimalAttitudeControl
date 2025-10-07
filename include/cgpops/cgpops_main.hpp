// Copyright (c) Yunus M. Agamawi and Anil Vithala Rao.  All Rights Reserved
//
// cgpops_main.hpp
// CGPOPS Tool Box
// Declare main continouos time optimal control problem here
//


#ifndef __CGPOPS_MAIN_HPP__
#define __CGPOPS_MAIN_HPP__

#include "nlpGlobVarExt.hpp"
#include <cgpops/cgpopsAuxExt.hpp>
#include <cgpops/cgpopsAuxDec.hpp>
#include <cgpops/cgpops_gov.hpp>
#include <vector>


// $#$#$


void cgpops_go(doubleMat& cgpopsResults, const std::vector<double>& initial_state, const std::vector<double>& final_state);


#endif

