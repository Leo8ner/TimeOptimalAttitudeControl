#include <toac/cuda_rk4.h>
#include <toac/symmetric_spacecraft.h>
#include <toac/external_dyn.h>

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) dyn_ ## ID
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
// #define casadi_s7 CASADI_PREFIX(s7)
// #define casadi_s8 CASADI_PREFIX(s8)
// #define casadi_s9 CASADI_PREFIX(s9)
// #define casadi_s10 CASADI_PREFIX(s10)
// #define casadi_s11 CASADI_PREFIX(s11)
// #define casadi_s12 CASADI_PREFIX(s12)
// #define casadi_s13 CASADI_PREFIX(s13)
// #define casadi_s14 CASADI_PREFIX(s14)
// #define casadi_s15 CASADI_PREFIX(s15)
// #define casadi_s16 CASADI_PREFIX(s16)
// #define casadi_s17 CASADI_PREFIX(s17)
// #define casadi_s18 CASADI_PREFIX(s18)
// #define casadi_s19 CASADI_PREFIX(s19)
// #define casadi_s20 CASADI_PREFIX(s20)
// #define casadi_s21 CASADI_PREFIX(s21)
#define casadi_nz4 CASADI_PREFIX(nz4)
#define casadi_nz5 CASADI_PREFIX(nz5)
#define casadi_nz6 CASADI_PREFIX(nz6)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

// Global CudaRk4 instance (eager initialization)

static int forward_cnt = 0;
static int jacobian_cnt = 0;
static CudaRk4* g_integrator = nullptr;
static bool g_initialized = false;

static const casadi_int casadi_nz4 = nnz * n_stp;                   // Total non-zeros in dX_next/dX (block sparsity pattern - each block is has 37 non-zeros - quaternions dont influence angular velocities)
static const casadi_int casadi_nz5 = n_controls * n_states * n_stp; // Total non-zeros in dX_next/du (block sparsity pattern - each block is dense)
static const casadi_int casadi_nz6 = n_states * n_stp;              // Total non-zeros in dX_next/ddt (block sparsity pattern - each block is dense)

static const casadi_int casadi_s0[3] = {n_states, n_stp, 1};   // Initial states sparsity, 1 is for dense
static const casadi_int casadi_s1[3] = {n_controls, n_stp, 1}; // Torque parameters sparsity, 1 is for dense
static const casadi_int casadi_s2[3] = {n_stp, 1, 1};          // Time steps sparsity, 1 is for dense
static casadi_int casadi_s3[3 + n_stp];                        // Output states sparsity, dummy for casadi structure
static casadi_int casadi_s4[3 + casadi_nz4 + n_states * n_stp]; // dX_next/dX sparsity memory allocation
static casadi_int casadi_s5[3 + casadi_nz5 + n_controls * n_stp]; // dX_next/du sparsity memory allocation
static casadi_int casadi_s6[3 + casadi_nz6 + n_stp]; // dX_next/ddt sparsity memory allocation

// static const casadi_int casadi_s7[3 + n_states * n_stp];  
// static const casadi_int casadi_s8[3 + n_controls * n_stp]; 
// static const casadi_int casadi_s9[3 + n_stp];
// static const casadi_int casadi_s10[6953] = {122500, 350, 0, ...};
// static const casadi_int casadi_s11[4353] = {122500, 150, 0, ...};
// static const casadi_int casadi_s12[1453] = {122500, 50, 0, ...};
// static const casadi_int casadi_s13[353]  = {122500, 350, 0, 0, ...};
// static const casadi_int casadi_s14[4553] = {52500, 350, 0, ...};
// static const casadi_int casadi_s15[1953] = {52500, 150, 0, ...};
// static const casadi_int casadi_s16[803]  = {52500, 50, 0, ...};
// static const casadi_int casadi_s17[353]  = {52500, 350, 0, 0, ...};
// static const casadi_int casadi_s18[1753] = {17500, 350, 0, ...};
// static const casadi_int casadi_s19[903]  = {17500, 150, 0, ...};
// static const casadi_int casadi_s20[253]  = {17500, 50, 0, ...};
// static const casadi_int casadi_s21[353]  = {17500, 350, 0, 0, ...};

// Build sparse patterns from CudaRk4
static void build_sparsity_patterns() {
    if (!g_integrator) return;

    // Build casadi_s3 (dummy X_next sparsity)
    int idx = 0;
    casadi_s3[idx++] = n_states;          // nrows
    casadi_s3[idx++] = n_stp;          // ncols
    for (int i = 0; i <= n_stp; i++) {
        casadi_s3[idx++] = 0;    // column pointers
    }

    // Get sparsity sizes
    int y0_nnz = g_integrator->getSparsitySizeY0();
    int u_nnz = g_integrator->getSparsitySizeU();  
    int dt_nnz = g_integrator->getSparsitySizeDt();
    
    if (y0_nnz < 0 || u_nnz < 0 || dt_nnz < 0 || 
        y0_nnz != casadi_nz4 || u_nnz != casadi_nz5 || dt_nnz != casadi_nz6) {
        return; // Size check failed
    }
    
    // Allocate arrays
    static casadi_int y0_row_indices[casadi_nz4];
    static casadi_int y0_col_pointers[n_states * n_stp + 1];
    static casadi_int u_row_indices[casadi_nz5];
    static casadi_int u_col_pointers[n_controls * n_stp + 1];
    static casadi_int dt_row_indices[casadi_nz6];
    static casadi_int dt_col_pointers[n_stp + 1];
    
    // Get sparsity patterns
    int ret1 = g_integrator->getSparsityY0(y0_row_indices, y0_col_pointers);
    int ret2 = g_integrator->getSparsityU(u_row_indices, u_col_pointers);
    int ret3 = g_integrator->getSparsityDt(dt_row_indices, dt_col_pointers);
    
    if (ret1 != 0 || ret2 != 0 || ret3 != 0) return;
    
    // Build casadi_s4 (dy/dy0)
    idx = 0;
    casadi_s4[idx++] = n_states * n_stp;          // nrows
    casadi_s4[idx++] = n_states * n_stp;          // ncols
    for (int i = 0; i <= n_states * n_stp; i++) {
        casadi_s4[idx++] = y0_col_pointers[i];    // column pointers
    }
    for (int i = 0; i < casadi_nz4; i++) {
        casadi_s4[idx++] = y0_row_indices[i];     // row indices
    }
    
    // Build casadi_s5 (dy/du)  
    idx = 0;
    casadi_s5[idx++] = n_states * n_stp;          // nrows
    casadi_s5[idx++] = n_controls * n_stp;        // ncols
    for (int i = 0; i <= n_controls * n_stp; i++) {
        casadi_s5[idx++] = u_col_pointers[i];     // column pointers
    }
    for (int i = 0; i < casadi_nz5; i++) {
        casadi_s5[idx++] = u_row_indices[i];      // row indices
    }
    
    // Build casadi_s6 (dy/dt)
    idx = 0;
    casadi_s6[idx++] = n_states * n_stp;          // nrows
    casadi_s6[idx++] = n_stp;                     // ncols
    for (int i = 0; i <= n_stp; i++) {
        casadi_s6[idx++] = dt_col_pointers[i];    // column pointers
    }
    for (int i = 0; i < casadi_nz6; i++) {
        casadi_s6[idx++] = dt_row_indices[i];     // row indices
    }
}

/* F:(i0[7x50],i1[3x50],i2[50])->(o0[7x50]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {

  F_incref();
  if (!g_integrator) {
    printf("Error initializing integrator!");
    return -1;
  }
  
  // Call CUDA integrator
  int ret = g_integrator->solve(arg[0], arg[1], arg[2]);  // No sensitivity for forward
  if (ret != 0) return ret;

  forward_cnt++; // Increment forward call count

  // Get solution
  return g_integrator->getSolution(res[0]);
}

CASADI_SYMBOL_EXPORT int F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int F_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int F_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void F_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int F_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void F_release(int mem) {
}

CASADI_SYMBOL_EXPORT void F_incref(void) {
  if (!g_initialized) {
    try {
      g_integrator = new CudaRk4();  // Enable sensitivity
      
      // Build sparsity patterns from integrator
      build_sparsity_patterns();
      
      g_initialized = true;
    } catch (const std::exception& e) {
      // Handle initialization error
      if (g_integrator) {
          delete g_integrator;
          g_integrator = nullptr;
      }
      g_initialized = false;
    }
  }
}

CASADI_SYMBOL_EXPORT void F_decref(void) {
  if (g_integrator) {
    delete g_integrator;
    g_integrator = nullptr;
  }
  g_initialized = false;
}


CASADI_SYMBOL_EXPORT casadi_int F_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int F_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real F_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* F_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* F_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* F_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* F_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}

/* jac_F:(i0[7x50],i1[3x50],i2[50],out_o0[7x50,0nz])->(jac_o0_i0[350x350,1550nz],jac_o0_i1[350x150,750nz],jac_o0_i2[350x50,350nz]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {

  F_incref();
  if (!g_integrator) {
    printf("Error initializing integrator!");
    return -1;
  }
  
  const casadi_real epsilon = 1e-6;  // Finite difference step size
  
  g_integrator->computeFiniteDifferenceSensitivities(arg[0], arg[1], arg[2], arg[3], epsilon);
  g_integrator->getSensitivitiesY0(res[0]);
  g_integrator->getSensitivitiesU(res[1]);
  g_integrator->getSensitivitiesDt(res[2]);

  jacobian_cnt++; // Increment Jacobian call count
  // printf("Finite difference Jacobian call count: %d\n", jacobian_cnt);
  return 0; // Success
}

CASADI_SYMBOL_EXPORT int jac_F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int jac_F_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int jac_F_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void jac_F_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int jac_F_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void jac_F_release(int mem) {
}

CASADI_SYMBOL_EXPORT void jac_F_incref(void) {
}

CASADI_SYMBOL_EXPORT void jac_F_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int jac_F_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int jac_F_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real jac_F_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* jac_F_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "out_o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* jac_F_name_out(casadi_int i) {
  switch (i) {
    case 0: return "jac_o0_i0";
    case 1: return "jac_o0_i1";
    case 2: return "jac_o0_i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* jac_F_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* jac_F_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int jac_F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int jac_F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}

// /* jac_jac_F:(i0[7x50],i1[3x50],i2[50],out_o0[7x50,0nz],out_jac_o0_i0[350x350,0nz],out_jac_o0_i1[350x150,0nz],out_jac_o0_i2[350x50,0nz])->(jac_jac_o0_i0_i0[122500x350,6600nz],jac_jac_o0_i0_i1[122500x150,4200nz],jac_jac_o0_i0_i2[122500x50,1400nz],jac_jac_o0_i0_out_o0[122500x350,0nz],jac_jac_o0_i1_i0[52500x350,4200nz],jac_jac_o0_i1_i1[52500x150,1800nz],jac_jac_o0_i1_i2[52500x50,750nz],jac_jac_o0_i1_out_o0[52500x350,0nz],jac_jac_o0_i2_i0[17500x350,1400nz],jac_jac_o0_i2_i1[17500x150,750nz],jac_jac_o0_i2_i2[17500x50,200nz],jac_jac_o0_i2_out_o0[17500x350,0nz]) */
// static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {

//   return 0;
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
//   return casadi_f2(arg, res, iw, w, mem);
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F_alloc_mem(void) {
//   return 0;
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F_init_mem(int mem) {
//   return 0;
// }

// CASADI_SYMBOL_EXPORT void jac_jac_F_free_mem(int mem) {
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F_checkout(void) {
//   return 0;
// }

// CASADI_SYMBOL_EXPORT void jac_jac_F_release(int mem) {
// }

// CASADI_SYMBOL_EXPORT void jac_jac_F_incref(void) {
// }

// CASADI_SYMBOL_EXPORT void jac_jac_F_decref(void) {
// }

// CASADI_SYMBOL_EXPORT casadi_int jac_jac_F_n_in(void) { return 7;}

// CASADI_SYMBOL_EXPORT casadi_int jac_jac_F_n_out(void) { return 12;}

// CASADI_SYMBOL_EXPORT casadi_real jac_jac_F_default_in(casadi_int i) {
//   switch (i) {
//     default: return 0;
//   }
// }

// CASADI_SYMBOL_EXPORT const char* jac_jac_F_name_in(casadi_int i) {
//   switch (i) {
//     case 0: return "i0";
//     case 1: return "i1";
//     case 2: return "i2";
//     case 3: return "out_o0";
//     case 4: return "out_jac_o0_i0";
//     case 5: return "out_jac_o0_i1";
//     case 6: return "out_jac_o0_i2";
//     default: return 0;
//   }
// }

// CASADI_SYMBOL_EXPORT const char* jac_jac_F_name_out(casadi_int i) {
//   switch (i) {
//     case 0: return "jac_jac_o0_i0_i0";
//     case 1: return "jac_jac_o0_i0_i1";
//     case 2: return "jac_jac_o0_i0_i2";
//     case 3: return "jac_jac_o0_i0_out_o0";
//     case 4: return "jac_jac_o0_i1_i0";
//     case 5: return "jac_jac_o0_i1_i1";
//     case 6: return "jac_jac_o0_i1_i2";
//     case 7: return "jac_jac_o0_i1_out_o0";
//     case 8: return "jac_jac_o0_i2_i0";
//     case 9: return "jac_jac_o0_i2_i1";
//     case 10: return "jac_jac_o0_i2_i2";
//     case 11: return "jac_jac_o0_i2_out_o0";
//     default: return 0;
//   }
// }

// CASADI_SYMBOL_EXPORT const casadi_int* jac_jac_F_sparsity_in(casadi_int i) {
//   switch (i) {
//     case 0: return casadi_s0;
//     case 1: return casadi_s1;
//     case 2: return casadi_s2;
//     case 3: return casadi_s3;
//     case 4: return casadi_s7;
//     case 5: return casadi_s8;
//     case 6: return casadi_s9;
//     default: return 0;
//   }
// }

// CASADI_SYMBOL_EXPORT const casadi_int* jac_jac_F_sparsity_out(casadi_int i) {
//   switch (i) {
//     case 0: return casadi_s10;
//     case 1: return casadi_s11;
//     case 2: return casadi_s12;
//     case 3: return casadi_s13;
//     case 4: return casadi_s14;
//     case 5: return casadi_s15;
//     case 6: return casadi_s16;
//     case 7: return casadi_s17;
//     case 8: return casadi_s18;
//     case 9: return casadi_s19;
//     case 10: return casadi_s20;
//     case 11: return casadi_s21;
//     default: return 0;
//   }
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
//   if (sz_arg) *sz_arg = 7;
//   if (sz_res) *sz_res = 12;
//   if (sz_iw) *sz_iw = 0;
//   if (sz_w) *sz_w = 0;
//   return 0;
// }

// CASADI_SYMBOL_EXPORT int jac_jac_F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
//   if (sz_arg) *sz_arg = 7*sizeof(const casadi_real*);
//   if (sz_res) *sz_res = 12*sizeof(casadi_real*);
//   if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
//   if (sz_w) *sz_w = 0*sizeof(casadi_real);
//   return 0;
// }