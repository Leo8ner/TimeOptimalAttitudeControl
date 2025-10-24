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
#include <unistd.h>
#include <fcntl.h>
#include <casadi/casadi.hpp>
#include <regex>
#include <map>
#include <cmath>

#include <toac/symmetric_spacecraft.h>
#include <cgpops/cgpopsAuxExt.hpp>

/**
 * @brief Parse Euler angles and angular velocities into quaternion state vector
 * @param input Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @return Tuple containing (state_vector, euler_angles) as DM objects
 */
std::tuple<casadi::DM, casadi::DM> parseStateVector(const std::string& input);

/**
 * @brief Extract initial guess from CSV trajectory file
 * @param csv_data Path to CSV file containing trajectory data
 * @param X_guess Output state trajectory matrix [7 x n_steps+1]
 * @param U_guess Output control sequence matrix [3 x n_steps]
 * @param dt_guess Output time step vector [1 x n_steps]
 */
void extractInitialGuess(const std::string& csv_data, casadi::DM& X_guess, casadi::DM& U_guess, casadi::DM& dt_guess);

/**
 * @brief Process and display optimization results
 * @param results Dictionary containing optimization results (X, U, T, dt)
 * @param angles_0 Initial Euler angles [degrees]
 * @param angles_f Final Euler angles [degrees]
 */
void processResults(casadi::DMDict& results, const casadi::DM& angles_0, const casadi::DM& angles_f);

/**
 * @brief Convert Euler angles to a quaternion
 * @param phi Roll angle (rad)
 * @param theta Pitch angle (rad)
 * @param psi Yaw angle (rad)
 * @return Quaternion as DM object
 */
casadi::DM euler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Load and return the compiled solver function from shared library
 * @return CasADi Function representing the solver
 */
casadi::Function get_solver();

/**
 * @brief Compute the skew-symmetric matrix for a 4D vector
 * @param w 4D vector (angular velocity)
 * @return Skew-symmetric matrix as MX object
 */
casadi::MX skew4(const casadi::MX& w);

/**
 * @brief Perform one step of RK4 integration
 * @param x_dot Derivative function (as MX)
 * @param x Current state (as MX)
 * @param dt Time step (as MX)
 * @return Next state after time step dt (as MX)
 */
casadi::MX rk4(const casadi::MX& x_dot, const casadi::MX& x, const casadi::MX& dt);

/**
 * @brief Export trajectory data to CSV file
 * @param X State trajectory matrix [10 x n_steps+1]
 * @param U Control sequence matrix [3 x n_steps]
 * @param T Total maneuver time (scalar)
 * @param dt Time step vector [1 x n_steps]
 * @param angles_0 Initial Euler angles [degrees]
 * @param angles_f Final Euler angles [degrees]
 * @param filename Output CSV file path
 */
void exportTrajectory(casadi::DM& X, const casadi::DM& U, const casadi::DM& T, const casadi::DM& dt,
                      const casadi::DM& angles_0, const casadi::DM& angles_f, const std::string& filename);

/**
 * @brief Convert quaternion to Euler angles (ZYX convention)
 * @param current Current Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @param previous Previous Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @param target Target Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @return Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 */
double unwrapAngle(double current, double previous, double target);

/**
 * @brief Convert quaternion to Euler angles (ZYX convention)
 * @param euler_prev Previous Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @param q Quaternion as DM object [4 x 1]
 * @param euler_target Target Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 * @return Euler angles [roll, pitch, yaw] in degrees as DM object [3 x 1]
 */
casadi::DM quat2euler(const casadi::DM& euler_prev, const casadi::DM& q, const casadi::DM& euler_target);

/**
 * @brief Parse initial and final state vectors from command line arguments
 * @param initial_state Comma-separated string with 6 values: roll_i,pitch_i,yaw_i,wx_i,wy_i,wz_i (degrees)
 * @param final_state Comma-separated string with 6 values: roll_f,pitch_f,yaw_f,wx_f,wy_f,wz_f (degrees)
 * @return Tuple containing (X_0, X_f, angles_0, angles_f) as DM objects
 */
std::tuple<casadi::DM, casadi::DM, casadi::DM, casadi::DM>
parseInput(const std::string& initial_state, const std::string& final_state);

/**
 * @brief Normalize angle to range [-180, 180] degrees
 * @param angle Angle in degrees
 * @return Normalized angle in degrees
 */
double normalizeAngle(double angle);

/**
 * @brief Redirect console output to a file
 * @param filename Path to the output file
 */
void redirect_output_to_file(const std::string& filename);

/**
 * @brief Restore output to console
 */
void restore_output_to_console();

/**
 * @brief Get solver status code from output file
 * @param filename Path to the solver log file
 * @return Integer status code
 */
int get_solver_status(const std::string& filename = "../output/solver_logs.log");

/**
 * @brief Parse Euler angles and angular velocities into quaternion state vector
 * @param initial_state Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @param final_state Comma-separated string with 6 values: roll,pitch,yaw,wx,wy,wz (degrees)
 * @return vector containing [q0, q1, q2, q3, omega_x, omega_y, omega_z] for both initial and final states
 */
std::vector<std::vector<double>> VparseStateVector(const std::string& initial_state, const std::string& final_state);

/**
 * @brief Convert Euler angles to a quaternion
 * @param phi Roll angle (rad)
 * @param theta Pitch angle (rad)
 * @param psi Yaw angle (rad)
 * @return Quaternion as vector<double>
 */
std::vector<double> Veuler2quat(const double& phi, const double& theta, const double& psi);

/**
 * @brief Parse MATLAB .mat file to extract state trajectory
 * @param derivativeSupplier Derivative supplier identifier
 * @param states Output state trajectory matrix [7 x n_steps]
 * @param controls Output control sequence matrix [3 x n_steps-1]
 * @param times Output time vector [1 x n_steps]
 */
void parseMatlab(const int& derivativeSupplier,
                 std::vector<std::vector<double>>& states,
                 std::vector<std::vector<double>>& controls,
                 std::vector<double>& dt,
                 double& T);

/**
 * @brief Get the solution value from the optimization result
 * @param derivativeSupplier Derivative supplier identifier
 * @return Solution value (scalar)
 */
double getCgpopsSolution(const int& derivativeSupplier);

/**
 * @brief Get file suffix based on derivative provider
 * @param derivative_provider Integer code for derivative provider
 * @return String suffix for files
 */
std::string fileSufix(const int& derivative_provider);

/**
 * @brief Rounds a double to specified precision
 * @param value The double value to round
 * @param precision Number of decimal places to round to (default is 2)
 */
double rnd(double value, int precision=2);

/**
 * @brief Load initial and final state samples from CSV file
 * @param initial_states Output vector of initial state vectors [q0,q1,q2,q3,wx,wy,wz]
 * @param final_states Output vector of final state vectors [q0,q1,q2,q3,wx,wy,wz]
 * @param filename Path to the CSV file containing state samples
 * @return true if successful, false otherwise
 */
bool loadStateSamples(std::vector<std::vector<double>>& initial_states,
                      std::vector<std::vector<double>>& final_states,
                      const std::string& filename="../output/lhs_samples.csv");

#endif /* HELPER_FUNCTIONS_H */