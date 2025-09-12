// external_dynamics.cpp
#include <toac/external_dynamics.h>

#define n_total_states (n_states * n_stp)
#define n_total_controls (n_controls * n_stp)

//==============================================================================
// GLOBAL STATE
//==============================================================================

static DynamicsIntegrator* g_integrator = nullptr;
static bool g_initialized = false;
static bool sparsity_initialized = false;
static int g_next_handle = 1;
static std::atomic<int> g_ref_count{0};
static std::mutex g_init_mutex;
static std::mutex g_memory_mutex;
static std::unordered_map<int, bool> g_memory_handles;

//==============================================================================
// SPARSITY PATTERN STORAGE (C-STYLE ARRAYS)
//==============================================================================

static casadi_int forward_sparsity_in_0[3];
static casadi_int forward_sparsity_in_1[3];
static casadi_int forward_sparsity_in_2[3];
static casadi_int forward_sparsity_out_0[3];

// C-style arrays for sparse patterns (allocated after integrator initialization)
static casadi_int* jac_sparsity_out_0_data = nullptr;
static casadi_int* jac_sparsity_out_1_data = nullptr;
static casadi_int* jac_sparsity_out_2_data = nullptr;

// Track sizes of allocated arrays
static int jac_sparsity_out_0_size = 0;
static int jac_sparsity_out_1_size = 0;
static int jac_sparsity_out_2_size = 0;

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

static void cleanup_sparsity_patterns(void) {
    if (jac_sparsity_out_0_data) {
        free(jac_sparsity_out_0_data);
        jac_sparsity_out_0_data = nullptr;
        jac_sparsity_out_0_size = 0;
    }
    if (jac_sparsity_out_1_data) {
        free(jac_sparsity_out_1_data);
        jac_sparsity_out_1_data = nullptr;
        jac_sparsity_out_1_size = 0;
    }
    if (jac_sparsity_out_2_data) {
        free(jac_sparsity_out_2_data);
        jac_sparsity_out_2_data = nullptr;
        jac_sparsity_out_2_size = 0;
    }
}

static void build_sparsity_patterns(void) {
    if (!g_integrator) return;
    
    // Cleanup any existing patterns first
    cleanup_sparsity_patterns();
    
    // Get sizes first
    int y0_nnz = g_integrator->getSparsitySizeY0();
    int u_nnz = g_integrator->getSparsitySizeU();
    int dt_nnz = g_integrator->getSparsitySizeDt();
    
    if (y0_nnz < 0 || u_nnz < 0 || dt_nnz < 0) {
        return; // Error getting sizes
    }
    
    // Allocate temporary arrays for sparsity patterns
    casadi_int* y0_row_indices = (casadi_int*)malloc(y0_nnz * sizeof(casadi_int));
    casadi_int* y0_col_pointers = (casadi_int*)malloc((n_total_states + 1) * sizeof(casadi_int));
    
    casadi_int* u_row_indices = (casadi_int*)malloc(u_nnz * sizeof(casadi_int));
    casadi_int* u_col_pointers = (casadi_int*)malloc((n_total_controls + 1) * sizeof(casadi_int));
    
    casadi_int* dt_row_indices = (casadi_int*)malloc(dt_nnz * sizeof(casadi_int));
    casadi_int* dt_col_pointers = (casadi_int*)malloc((n_stp + 1) * sizeof(casadi_int));
    
    // Check for allocation failures and cleanup properly
    if (!y0_row_indices || !y0_col_pointers || !u_row_indices || 
        !u_col_pointers || !dt_row_indices || !dt_col_pointers) {
        // Cleanup allocated arrays
        if (y0_row_indices) free(y0_row_indices);
        if (y0_col_pointers) free(y0_col_pointers);
        if (u_row_indices) free(u_row_indices);
        if (u_col_pointers) free(u_col_pointers);
        if (dt_row_indices) free(dt_row_indices);
        if (dt_col_pointers) free(dt_col_pointers);
        return;
    }
    
    // Get sparsity patterns from integrator
    int actual_y0_nnz, actual_u_nnz, actual_dt_nnz;
    int ret_y0 = g_integrator->getSparsityY0(y0_row_indices, y0_col_pointers, &actual_y0_nnz);
    int ret_u = g_integrator->getSparsityU(u_row_indices, u_col_pointers, &actual_u_nnz);
    int ret_dt = g_integrator->getSparsityDt(dt_row_indices, dt_col_pointers, &actual_dt_nnz);
    
    if (ret_y0 != 0 || ret_u != 0 || ret_dt != 0) {
        // Error getting sparsity - cleanup and return
        free(y0_row_indices);
        free(y0_col_pointers);
        free(u_row_indices);
        free(u_col_pointers);
        free(dt_row_indices);
        free(dt_col_pointers);
        return;
    }
    
    // === Build dy/dX sparsity ===
    jac_sparsity_out_0_size = 2 + (n_total_states + 1) + actual_y0_nnz;
    jac_sparsity_out_0_data = (casadi_int*)malloc(jac_sparsity_out_0_size * sizeof(casadi_int));
    
    if (jac_sparsity_out_0_data) {
        int idx = 0;
        // Header: {nrow, ncol}
        jac_sparsity_out_0_data[idx++] = n_total_states;
        jac_sparsity_out_0_data[idx++] = n_total_states;
        
        // Column pointers
        for (int i = 0; i <= n_total_states; i++) {
            jac_sparsity_out_0_data[idx++] = y0_col_pointers[i];
        }
        
        // Row indices
        for (int i = 0; i < actual_y0_nnz; i++) {
            jac_sparsity_out_0_data[idx++] = y0_row_indices[i];
        }
    }
    
    // === Build dy/dU sparsity ===
    jac_sparsity_out_1_size = 2 + (n_total_controls + 1) + actual_u_nnz;
    jac_sparsity_out_1_data = (casadi_int*)malloc(jac_sparsity_out_1_size * sizeof(casadi_int));
    
    if (jac_sparsity_out_1_data) {
        int idx = 0;
        // Header: {nrow, ncol}
        jac_sparsity_out_1_data[idx++] = n_total_states;
        jac_sparsity_out_1_data[idx++] = n_total_controls;
        
        // Column pointers
        for (int i = 0; i <= n_total_controls; i++) {
            jac_sparsity_out_1_data[idx++] = u_col_pointers[i];
        }
        
        // Row indices
        for (int i = 0; i < actual_u_nnz; i++) {
            jac_sparsity_out_1_data[idx++] = u_row_indices[i];
        }
    }
    
    // === Build dy/ddt sparsity ===
    jac_sparsity_out_2_size = 2 + (n_stp + 1) + actual_dt_nnz;
    jac_sparsity_out_2_data = (casadi_int*)malloc(jac_sparsity_out_2_size * sizeof(casadi_int));
    
    if (jac_sparsity_out_2_data) {
        int idx = 0;
        // Header: {nrow, ncol}
        jac_sparsity_out_2_data[idx++] = n_total_states;
        jac_sparsity_out_2_data[idx++] = n_stp;
        
        // Column pointers
        for (int i = 0; i <= n_stp; i++) {
            jac_sparsity_out_2_data[idx++] = dt_col_pointers[i];
        }
        
        // Row indices
        for (int i = 0; i < actual_dt_nnz; i++) {
            jac_sparsity_out_2_data[idx++] = dt_row_indices[i];
        }
    }

    // Cleanup temporary arrays
    free(y0_row_indices);
    free(y0_col_pointers);
    free(u_row_indices);
    free(u_col_pointers);
    free(dt_row_indices);
    free(dt_col_pointers);
}

static void init_sparsity_patterns(void) {
    if (sparsity_initialized) return;
    
    // Forward dynamics (dense)
    forward_sparsity_in_0[0] = n_states;
    forward_sparsity_in_0[1] = n_stp;
    forward_sparsity_in_0[2] = 1;
    
    forward_sparsity_in_1[0] = n_controls;
    forward_sparsity_in_1[1] = n_stp;
    forward_sparsity_in_1[2] = 1;
    
    forward_sparsity_in_2[0] = n_stp;
    forward_sparsity_in_2[1] = 1;
    forward_sparsity_in_2[2] = 1;
    
    forward_sparsity_out_0[0] = n_states;
    forward_sparsity_out_0[1] = n_stp;
    forward_sparsity_out_0[2] = 1;
    
    // Build sparse Jacobian patterns from integrator
    build_sparsity_patterns();

    sparsity_initialized = true;
}

//==============================================================================
// MEMORY MANAGEMENT (Thread-safe for non-thread-safe integrator)
//==============================================================================

int allocate_memory_handle() {
    std::lock_guard<std::mutex> lock(g_memory_mutex);
    int handle = g_next_handle++;
    g_memory_handles[handle] = false;
    return handle;
}

void free_memory_handle(int handle) {
    std::lock_guard<std::mutex> lock(g_memory_mutex);
    g_memory_handles.erase(handle);
}

bool checkout_memory_handle(int handle) {
    std::lock_guard<std::mutex> lock(g_memory_mutex);
    auto it = g_memory_handles.find(handle);
    if (it != g_memory_handles.end() && !it->second) {
        it->second = true;
        return true;
    }
    return false;
}

void release_memory_handle(int handle) {
    std::lock_guard<std::mutex> lock(g_memory_mutex);
    auto it = g_memory_handles.find(handle);
    if (it != g_memory_handles.end()) {
        it->second = false;
    }
}

//==============================================================================
// FORWARD DYNAMICS API IMPLEMENTATION
//==============================================================================

extern "C" CASADI_SYMBOL_EXPORT void dynamics_incref(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_ref_count.fetch_add(1) == 0) {
        g_integrator = new DynamicsIntegrator(true);  // Enable sensitivity
        init_sparsity_patterns();
        g_initialized = true;
    }
}

extern "C" CASADI_SYMBOL_EXPORT void dynamics_decref(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_ref_count.fetch_sub(1) == 1) {
        if (g_integrator) {
            delete g_integrator;
            g_integrator = nullptr;
        }
        cleanup_sparsity_patterns();  // Cleanup allocated sparsity arrays
        g_initialized = false;
        sparsity_initialized = false;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics_alloc_mem(void) {
    return allocate_memory_handle();
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics_init_mem(int mem) {
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT void dynamics_free_mem(int mem) {
    free_memory_handle(mem);
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics_checkout(void) {
    int handle = allocate_memory_handle();
    return handle;
}

extern "C" CASADI_SYMBOL_EXPORT void dynamics_release(int mem) {
    release_memory_handle(mem);
}

extern "C" CASADI_SYMBOL_EXPORT casadi_int dynamics_n_in(void) { return 3; }
extern "C" CASADI_SYMBOL_EXPORT casadi_int dynamics_n_out(void) { return 1; }

extern "C" CASADI_SYMBOL_EXPORT casadi_real dynamics_default_in(casadi_int i) {
    switch (i) {
        default: return 0.0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* dynamics_name_in(casadi_int ind) {
    switch(ind) {
        case 0: return "X";
        case 1: return "U";
        case 2: return "dt";
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* dynamics_name_out(casadi_int ind) {
    switch(ind) {
        case 0: return "X_next";
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* dynamics_sparsity_in(casadi_int ind) {
    init_sparsity_patterns();
    switch((int)ind) {
        case 0: return forward_sparsity_in_0;
        case 1: return forward_sparsity_in_1;
        case 2: return forward_sparsity_in_2;
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* dynamics_sparsity_out(casadi_int ind) {
    init_sparsity_patterns();
    switch((int)ind) {
        case 0: return forward_sparsity_out_0;
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
    if (sz_arg) {
        sz_arg[0] = n_states * n_stp;      // X
        sz_arg[1] = n_controls * n_stp;    // U  
        sz_arg[2] = n_stp;                 // dt
    }
    if (sz_res) {
        sz_res[0] = n_states * n_stp;      // X_next
    }
    if (sz_iw) *sz_iw = 0;                 // Integer work
    if (sz_w) *sz_w = 0;                   // Real work
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
    if (sz_arg) *sz_arg = 3 * sizeof(const casadi_real*);
    if (sz_res) *sz_res = 1 * sizeof(casadi_real*);
    if (sz_iw) *sz_iw = 0 * sizeof(casadi_int);
    if (sz_w) *sz_w = 0 * sizeof(casadi_real);
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int dynamics(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
    if (!g_initialized) {
        dynamics_incref();
        if (!g_initialized) {
            return -1;
        }
    }
    
    // Solve dynamics (no sensitivity for forward evaluation)
    int ret_code = g_integrator->solve(arg[0], arg[1], arg[2], false);
    if (ret_code != 0) {
        return ret_code;
    }
    
    // Get solution directly to output buffer (no conversion needed)
    ret_code = g_integrator->getSolution(res[0]);
    if (ret_code != 0) {
        return ret_code;
    }

    return 0;
}

//==============================================================================
// JACOBIAN DYNAMICS API IMPLEMENTATION
//==============================================================================

extern "C" CASADI_SYMBOL_EXPORT void jac_dynamics_incref(void) {
    dynamics_incref(); // Reuse same initialization
}

extern "C" CASADI_SYMBOL_EXPORT void jac_dynamics_decref(void) {
    dynamics_decref(); // Reuse same cleanup
}

extern "C" CASADI_SYMBOL_EXPORT int jac_dynamics_alloc_mem(void) {
    return allocate_memory_handle();
}

extern "C" CASADI_SYMBOL_EXPORT int jac_dynamics_init_mem(int mem) {
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT void jac_dynamics_free_mem(int mem) {
    free_memory_handle(mem);
}

extern "C" CASADI_SYMBOL_EXPORT int jac_dynamics_checkout(void) {
    int handle = allocate_memory_handle();
    return handle;
}

extern "C" CASADI_SYMBOL_EXPORT void jac_dynamics_release(int mem) {
    release_memory_handle(mem);
}

extern "C" CASADI_SYMBOL_EXPORT casadi_int jac_dynamics_n_in(void) { return 4; }
extern "C" CASADI_SYMBOL_EXPORT casadi_int jac_dynamics_n_out(void) { return 3; }

extern "C" CASADI_SYMBOL_EXPORT casadi_real jac_dynamics_default_in(casadi_int i) {
    switch (i) {
        default: return 0.0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* jac_dynamics_name_in(casadi_int ind) {
    switch((int)ind) {
        case 0: return "X";
        case 1: return "U";
        case 2: return "dt";
        case 3: return "X_next";
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* jac_dynamics_name_out(casadi_int ind) {
    switch(ind) {
        case 0: return "jac_X_next_X";
        case 1: return "jac_X_next_U";
        case 2: return "jac_X_next_dt";
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* jac_dynamics_sparsity_in(casadi_int ind) {
    init_sparsity_patterns();
    switch((int)ind) {
        case 0: return forward_sparsity_in_0;
        case 1: return forward_sparsity_in_1;
        case 2: return forward_sparsity_in_2;
        case 3: return forward_sparsity_out_0;
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* jac_dynamics_sparsity_out(casadi_int ind) {
    init_sparsity_patterns();
    switch((int)ind) {
        case 0: return jac_sparsity_out_0_data;
        case 1: return jac_sparsity_out_1_data;
        case 2: return jac_sparsity_out_2_data;
        default: return NULL;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int jac_dynamics_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
    init_sparsity_patterns();
    
    if (sz_arg) {
        sz_arg[0] = n_states * n_stp;      // X
        sz_arg[1] = n_controls * n_stp;    // U  
        sz_arg[2] = n_stp;                 // dt
        sz_arg[3] = n_states * n_stp;      // X_next
    }
    if (sz_res) {
        // Return number of non-zeros for each sparse Jacobian
        if (jac_sparsity_out_0_data && jac_sparsity_out_1_data && jac_sparsity_out_2_data) {
            // Calculate nnz from sparsity arrays: last_colptr - first_colptr
            casadi_int n_cols_0 = jac_sparsity_out_0_data[1];  // ncol
            casadi_int n_cols_1 = jac_sparsity_out_1_data[1];  // ncol  
            casadi_int n_cols_2 = jac_sparsity_out_2_data[1];  // ncol
            
            sz_res[0] = jac_sparsity_out_0_data[2 + n_cols_0] - jac_sparsity_out_0_data[2]; // nnz for dy/dX
            sz_res[1] = jac_sparsity_out_1_data[2 + n_cols_1] - jac_sparsity_out_1_data[2]; // nnz for dy/dU
            sz_res[2] = jac_sparsity_out_2_data[2 + n_cols_2] - jac_sparsity_out_2_data[2]; // nnz for dy/ddt
        } else {
            // Fallback to dense sizes if sparsity not yet computed
            sz_res[0] = n_total_states * n_total_states;
            sz_res[1] = n_total_states * n_total_controls;
            sz_res[2] = n_total_states * n_stp;
        }
    }
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int jac_dynamics_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
    init_sparsity_patterns();
    
    if (sz_arg) *sz_arg = 4 * sizeof(const casadi_real*);
    if (sz_res) *sz_res = 3 * sizeof(casadi_real*);
    if (sz_iw) *sz_iw = 0 * sizeof(casadi_int);
    if (sz_w) *sz_w = 0 * sizeof(casadi_real);
    return 0;
}

extern "C" int jac_dynamics(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
    if (!g_initialized) {
        jac_dynamics_incref();
    }
    
    // Solve dynamics WITH sensitivity
    int ret_code = g_integrator->solve(arg[0], arg[1], arg[2], true);
    if (ret_code != 0) {
        return ret_code;
    }
    
    // Extract sensitivities directly to output buffers (no conversion needed)
    ret_code = g_integrator->getSensitivitiesY0(res[0]);
    if (ret_code != 0) {
        return ret_code;
    }
    
    ret_code = g_integrator->getSensitivitiesU(res[1]);
    if (ret_code != 0) {
        return ret_code;
    }
    
    ret_code = g_integrator->getSensitivitiesDt(res[2]);
    if (ret_code != 0) {
        return ret_code;
    }
    
    return 0;
}

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

extern "C" int initialize_dynamics_integrator() {
    dynamics_incref();
    return g_initialized ? 0 : -1;
}

extern "C" void cleanup_dynamics_integrator() {
    dynamics_decref();
}