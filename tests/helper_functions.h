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
#include <casadi/casadi.hpp>
#include "symmetric_spacecraft.h"
#include "test2.h"

using namespace casadi;

/**
 * @brief Parse Euler angles and angular velocities into quaternion state vector
 * @param input Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @return Tuple containing (state_vector, euler_angles) as DM objects
 */
std::tuple<DM, DM> parseStateVector(const std::string& input);

/**
 * @brief Extract initial guess from CSV trajectory file
 * @param csv_data Path to CSV file containing trajectory data
 * @param X_guess Output state trajectory matrix [7 x n_steps+1]
 * @param U_guess Output control sequence matrix [3 x n_steps]
 * @param dt_guess Output time step vector [1 x n_steps]
 */
void extractInitialGuess(const std::string& csv_data, DM& X_guess, DM& U_guess, DM& dt_guess);

/**
 * @brief Process and display optimization results
 * @param results Dictionary containing optimization results (X, U, T, dt)
 * @param angles_0 Initial Euler angles [degrees]
 * @param angles_f Final Euler angles [degrees]
 */
void processResults(DMDict& results, const DM& angles_0, const DM& angles_f);

/**
 * @brief Convert Euler angles to a quaternion
 * @param phi Roll angle (rad)
 * @param theta Pitch angle (rad)
 * @param psi Yaw angle (rad)
 * @return Quaternion as DM object
 */
DM euler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Load and return the compiled solver function from shared library
 * @return CasADi Function representing the solver
 */
Function get_solver();

/**
 * @brief Compute the skew-symmetric matrix for a 4D vector
 * @param w 4D vector (angular velocity)
 * @return Skew-symmetric matrix as SX object
 */
SX skew4(const SX& w);

/**
 * @brief Perform one step of RK4 integration
 * @param x_dot Derivative function (as SX)
 * @param x Current state (as SX)
 * @param dt Time step (as SX)
 * @return Next state after time step dt (as SX)
 */
SX rk4(const SX& x_dot, const SX& x, const SX& dt);

/**
 * @brief Export trajectory data to CSV file
 * @param X State trajectory matrix [10 x n_steps+1]
 * @param U Control sequence matrix [3 x n_steps]
 * @param T Total maneuver time (scalar)
 * @param dt Time step vector [1 x n_steps]
 * @param filename Output CSV file path
 */
void exportTrajectory(DM& X, const DM& U, const DM& T, const DM& dt, const std::string& filename);

/**
 * @brief Convert quaternion to Euler angles (ZYX convention)
 * @param quat Quaternion as DM object [4 x 1]
 * @return Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 */
double unwrapAngle(double current_angle, double previous_angle);

/**
 * @brief Convert quaternion to Euler angles (ZYX convention)
 * @param euler_angles Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @param q Quaternion as DM object [4 x 1]
 */
DM quat2euler(const DM& euler_angles, const DM& q);

/**
 * @brief Parse initial and final state vectors from command line arguments
 * @param initial_state Comma-separated string with 6 values: roll_i,pitch_i,yaw_i,wx_i,wy_i,wz_i (degrees)
 * @param final_state Comma-separated string with 6 values: roll_f,pitch_f,yaw_f,wx_f,wy_f,wz_f (degrees)
 * @return Tuple containing (X_0, X_f) as DM objects
 */
std::tuple<DM, DM> parseInput(const std::string& initial_state, const std::string& final_state);

int mainish(int argc, char* argv[]);

#endif /* HELPER_FUNCTIONS_H */