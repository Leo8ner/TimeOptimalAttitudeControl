//==============================================================================
// RK4 DYNAMICS INTEGRATOR HEADER
//==============================================================================

#include <cuda_runtime.h>
#include "symmetric_spacecraft.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <ctime>
#include <iomanip>

//------------------------------------------------------------------------------
// Error checking macros
//------------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            cerr << "CUDA kernel error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << endl; \
            return -1; \
        } \
    } while(0)

__global__ void rk4Step(double* y_in, double* y_out,
                       double* k1, double* k2, double* k3, double* k4,
                       double* torque_params, double* dt_params,
                       int n_states_total);

__device__ void computeDerivatives(int sys, double* y_local, double* ydot_local,
                                 double* torque_params, double dt);

__global__ void computeSparseJacobian(int stage, double* y_stage_data, 
                                     double* jacobian_sparse_out,
                                     double* torque_params, double* dt_params);

__global__ void updateStageArrays(double* y_base, double* k_y, 
                                 double* sens_base, double* k_sens,
                                 double* y_stage_out, double* sens_stage_out,
                                 double* dt_params, double stage_factor);

__global__ void combineRK4Sensitivities(double* sens_final,
                                        double* sens_k1, double* sens_k2, 
                                        double* sens_k3, double* sens_k4,
                                        double* k1, double* k2, double* k3, double* k4,
                                        double* dt_params);
class DynamicsIntegrator {
private:

    //  CUDA arrays:
    double* d_y;           // Device state vector
    double* d_k1, *d_k2, *d_k3, *d_k4;  // RK4 stage vectors
    
    // Keep existing parameter arrays:
    double* d_torque_params_ptr;
    double* d_dt_params_ptr;
    double* h_y_pinned;
    double* h_tau_pinned;
    double* h_dt_pinned;

    cudaStream_t compute_stream;

    // Add finite difference workspace:
    double* h_y_temp;          // Temporary host buffer
    double* h_tau_temp;        // Temporary host torque buffer
    double* h_dt_temp;         // Temporary host dt buffer

    double* h_sens;           // Host sensitivity storage

    // Sensitivity storage (CCS format):
    long long int* y0_row_idx, *u_row_idx, *dt_row_idx;
    long long int* y0_col_ptr, *u_col_ptr, *dt_col_ptr;
    int y0_nnz, u_nnz, dt_nnz;
    int y0_n_cols, u_n_cols, dt_n_cols;
    
    bool spar_computed;
    bool anal_sens_was_setup;
    bool fd_sens_was_setup;

    // Sensitivity value storage:
    double* y0_sens_values;
    double* u_sens_values;
    double* dt_sens_values;

    double* d_sens_matrices;  // 11 params × 50 systems × 7 states
    double* d_jacobian_matrices;     // 4 stages × 50 systems × 30 nnz
    double *d_sens_k1, *d_sens_k2, *d_sens_k3, *d_sens_k4;
    double *d_y_stage, *d_sens_stage;

    // Timing and configuration:
    const int n_params = n_states + n_controls + 1;
    const int sens_total = n_stp * n_params * n_states;
    float setup_time, solve_time;
    cudaEvent_t start, stop;
    static constexpr double FD_EPSILON = 1e-6;

public:
    // Constructor/destructor:
    DynamicsIntegrator(); 
    ~DynamicsIntegrator();

    // Core integration :
    int solve(const double* initial_states, 
              const double* torque_params,
              const double* delta_t);

    // Solution access:
    int getSolution(double* next_state) const;

    // Finite difference sensitivity method:
    int computeFiniteDifferenceSensitivities(const double* initial_states,
                                           const double* torque_params, 
                                           const double* delta_t,
                                           const double* baseline_solution, 
                                           double epsilon = FD_EPSILON);

    // CCS sparsity pattern access:
    int getSensitivitiesY0(double* values) const;
    int getSensitivitiesU(double* values) const;
    int getSensitivitiesDt(double* values) const;
    
    int getSparsityY0(long long int* row_indices, long long int* col_pointers, int* nnz_out);
    int getSparsityU(long long int* row_indices, long long int* col_pointers, int* nnz_out);
    int getSparsityDt(long long int* row_indices, long long int* col_pointers, int* nnz_out);

        // Timing accessors
    float getSolveTime() const { return solve_time; }
    float getSetupTime() const { return setup_time; }
    
    // Physics validation methods
    void generateRandomQuaternion(double* quaternion);
    void generateRandomAngularVelocity(double* omega);
    void generateRandomTorque(double* torque);
    void generateBatchInputs(int batch_size, double* initial_states,
                           double* torque_params, double* dt_params);
    
    void getQuaternionNorms(const double* states, double* norms) const;
    void computeAngularMomentum(const double* states, double* angular_momentum) const;
    void computeRotationalEnergy(const double* states, double* energy) const;
    void computePowerInput(const double* states, const double* torques, double* power) const;
    
    void validatePhysics(const double* initial_states, const double* final_states,
                        const double* torque_params, const double* integration_time) const;
    void verifyPhysics(const double* initial_states, const double* final_states,
                      const double* torque_params, const double* integration_time) const;
    
    // Performance analysis methods
    void profileIntegration(int batch_size, int num_iterations);
    void measureFiniteDifferenceCost(int n_iterations);
    void runComprehensiveTest(int iterations);
    void runAnalyticalSensTest();
private:
    void cleanup();
    void setInitialConditions(const double* initial_states,
                             const double* torque_params,
                             const double* delta_t);
    
    // New private methods:
    void setupAnalyticalSensitivities();
    void setupFDSensitivities();

    int performRK4Integration(double dt_max);
    void computeSparsityPatterns();
    void allocateFiniteDifferenceWorkspace();
    void extractY0Sensitivities(int state_idx, double epsilon, 
                                const double* baseline_solution);
    void extractUSensitivities(int torque_idx, double epsilon, 
                               const double* baseline_solution);
    void extractDtSensitivities(double epsilon, 
                                const double* baseline_solution);
    int computeAnalyticalSensitivities(const double* initial_states,
                                  const double* torque_params,
                                  const double* delta_t);
    void extractAnalyticalToCCS();
    void initializeSensMatrices();
};