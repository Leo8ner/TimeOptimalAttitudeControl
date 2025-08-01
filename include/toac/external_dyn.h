#ifndef EXTERNAL_DYN_H
#define EXTERNAL_DYN_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

// Function declarations for F
int F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int F_alloc_mem(void);
int F_init_mem(int mem);
void F_free_mem(int mem);
int F_checkout(void);
void F_release(int mem);
void F_incref(void);
void F_decref(void);
casadi_int F_n_in(void);
casadi_int F_n_out(void);
casadi_real F_default_in(casadi_int i);
const char* F_name_in(casadi_int i);
const char* F_name_out(casadi_int i);
const casadi_int* F_sparsity_in(casadi_int i);
const casadi_int* F_sparsity_out(casadi_int i);
int F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);

#define F_SZ_ARG 3
#define F_SZ_RES 1
#define F_SZ_IW 0
#define F_SZ_W 0

// Function declarations for jac_F
int jac_F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int jac_F_alloc_mem(void);
int jac_F_init_mem(int mem);
void jac_F_free_mem(int mem);
int jac_F_checkout(void);
void jac_F_release(int mem);
void jac_F_incref(void);
void jac_F_decref(void);
casadi_int jac_F_n_in(void);
casadi_int jac_F_n_out(void);
casadi_real jac_F_default_in(casadi_int i);
const char* jac_F_name_in(casadi_int i);
const char* jac_F_name_out(casadi_int i);
const casadi_int* jac_F_sparsity_in(casadi_int i);
const casadi_int* jac_F_sparsity_out(casadi_int i);
int jac_F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int jac_F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);

#define jac_F_SZ_ARG 4
#define jac_F_SZ_RES 3
#define jac_F_SZ_IW 0
#define jac_F_SZ_W 0

// Commented function declarations (keeping as requested)
// int jac_jac_F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
// int jac_jac_F_alloc_mem(void);
// int jac_jac_F_init_mem(int mem);
// void jac_jac_F_free_mem(int mem);
// int jac_jac_F_checkout(void);
// void jac_jac_F_release(int mem);
// void jac_jac_F_incref(void);
// void jac_jac_F_decref(void);
// casadi_int jac_jac_F_n_in(void);
// casadi_int jac_jac_F_n_out(void);
// casadi_real jac_jac_F_default_in(casadi_int i);
// const char* jac_jac_F_name_in(casadi_int i);
// const char* jac_jac_F_name_out(casadi_int i);
// const casadi_int* jac_jac_F_sparsity_in(casadi_int i);
// const casadi_int* jac_jac_F_sparsity_out(casadi_int i);
// int jac_jac_F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
// int jac_jac_F_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
// #define jac_jac_F_SZ_ARG 7
// #define jac_jac_F_SZ_RES 12
// #define jac_jac_F_SZ_IW 0
// #define jac_jac_F_SZ_W 0

#ifdef __cplusplus
}
#endif

#endif // EXTERNAL_DYN_H