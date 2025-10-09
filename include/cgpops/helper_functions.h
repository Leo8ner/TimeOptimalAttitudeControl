/**
 * @file helper_functions.h
 * @brief Helper functions for spacecraft attitude control optimization
 * 
 * This header provides utility functions for parsing state vectors,
 * extracting initial guesses from CSV data, and processing optimization results.
 */

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <string>
#include <tuple>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <regex>
#include <map>
#include <cmath>

#include <cgpops/cgpopsAuxExt.hpp>

using namespace std;

/**
 * @brief Parse Euler angles and angular velocities into quaternion state vector
 * @param initial_state Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @param final_state Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @return vector containing [q0, q1, q2, q3, omega_x, omega_y, omega_z] for both initial and final states
 */
vector<vector<double>> parseStateVector(const string& initial_state, const string& final_state);

/**
 * @brief Convert Euler angles to a quaternion
 * @param phi Roll angle (rad)
 * @param theta Pitch angle (rad)
 * @param psi Yaw angle (rad)
 * @return Quaternion as vector<double>
 */
vector<double> euler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Parse MATLAB .mat file to extract state trajectory
 * @param filename Path to .mat file
 * @param states Output state trajectory matrix [7 x n_steps]
 * @param controls Output control sequence matrix [3 x n_steps-1]
 * @param times Output time vector [1 x n_steps]
 */
void parseMatlab(
    const string& filename,
    vector<vector<double>>& states,      // Output: 7×201 matrix [w,x,y,z,ωx,ωy,ωz]
    vector<vector<double>>& controls,    // Output: 3×201 matrix [ux,uy,uz]
    vector<double>& dt,           // Output: 1×201 vector [t0, t1, ..., t200]
    double& T             // Output: Total maneuver time (scalar)
);

#endif // HELPER_FUNCTIONS_H