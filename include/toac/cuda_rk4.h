//==============================================================================
// RK4 DYNAMICS INTEGRATOR HEADER
//==============================================================================

#include <cuda_runtime.h>
#include <toac/symmetric_spacecraft.h>
#include <iostream>

//------------------------------------------------------------------------------
// Error checking macros
//------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl;       \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define CUDA_CHECK_KERNEL()                                          \
    do                                                               \
    {                                                                \
        cudaError_t err = cudaGetLastError();                        \
        if (err != cudaSuccess)                                      \
        {                                                            \
            cerr << "CUDA kernel error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;   \
            return -1;                                               \
        }                                                            \
    } while (0)

__global__ void rk4Step(double *y_in, double *y_out,
                        double *k1, double *k2, double *k3, double *k4,
                        double *torque_params, double *dt_params,
                        int n_states_total);

__device__ void computeDerivatives(int sys, double *y_local, double *ydot_local,
                                   double *torque_params, double dt);

class CudaRk4
{
private:
    //  CUDA arrays:
    double *d_y;                       // Device state vector
    double *d_k1, *d_k2, *d_k3, *d_k4; // RK4 stage vectors

    // Keep existing parameter arrays:
    double *d_torque_params_ptr;
    double *d_dt_params_ptr;
    double *h_y_pinned;
    double *h_tau_pinned;
    double *h_dt_pinned;

    cudaStream_t compute_stream;

    // Add finite difference workspace:
    double *h_y_temp;   // Temporary host buffer
    double *h_tau_temp; // Temporary host torque buffer
    double *h_dt_temp;  // Temporary host dt buffer

    // Sensitivity storage (CCS format):
    long long int *y0_row_idx, *u_row_idx, *dt_row_idx;
    long long int *y0_col_ptr, *u_col_ptr, *dt_col_ptr;
    int y0_nnz, u_nnz, dt_nnz;

    bool spar_computed;
    bool fd_sens_was_setup;

    // Sensitivity value storage:
    double *y0_sens_values;
    double *u_sens_values;
    double *dt_sens_values;

    // Timing and configuration:
    float setup_time, solve_time;
    cudaEvent_t start, stop;
    static constexpr double FD_EPSILON = 1e-6;

public:
    // Constructor/destructor:
    CudaRk4();
    ~CudaRk4();

    // Core integration :
    int solve(const double *initial_states,
              const double *torque_params,
              const double *delta_t);

    // Solution access:
    int getSolution(double *next_state) const;

    // Finite difference sensitivity method:
    int computeFiniteDifferenceSensitivities(const double *initial_states,
                                             const double *torque_params,
                                             const double *delta_t,
                                             const double *baseline_solution,
                                             double epsilon = FD_EPSILON);

    // CCS sparsity pattern access:
    int getSensitivitiesY0(double *values);
    int getSensitivitiesU(double *values);
    int getSensitivitiesDt(double *values);

    int getSparsityY0(long long int *row_indices, long long int *col_pointers);
    int getSparsityU(long long int *row_indices, long long int *col_pointers);
    int getSparsityDt(long long int *row_indices, long long int *col_pointers);

    int getSparsitySizeY0();
    int getSparsitySizeU();
    int getSparsitySizeDt();

    // Timing accessors
    float getSolveTime() const { return solve_time; }
    float getSetupTime() const { return setup_time; }

private:
    void cleanup();
    void setInitialConditions(const double *initial_states,
                              const double *torque_params,
                              const double *delta_t);

    // New private methods:
    void setupFDSensitivities();

    void computeSparsityPatterns();
    void extractY0Sensitivities(int state_idx, double epsilon,
                                const double *baseline_solution);
    void extractUSensitivities(int torque_idx, double epsilon,
                               const double *baseline_solution);
    void extractDtSensitivities(double epsilon,
                                const double *baseline_solution);
};