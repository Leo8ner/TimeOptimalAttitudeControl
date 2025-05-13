#ifndef PLOTS_H
#define PLOTS_H

#include <iostream>
#include <fstream>
#include <casadi/casadi.hpp>
#include <string>

using namespace casadi;

// Converts a quaternion to Euler angles
DM quat2euler(const DM& q);

void exportTrajectory(const DM& X, const DM& U, const DM& T, const DM& dt, const std::string& filename);

#endif // PLOTS_H