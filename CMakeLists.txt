cmake_minimum_required(VERSION 3.20)
project(TimeOptimalAttitudeControl)

# ============================================================================
# Build Configuration Options
# ============================================================================

# Enable compile commands export for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional: Configure external library paths
option(ENABLE_CGPOPS "Enable CGPOPS-based optimization targets" ON)
option(ENABLE_CUDA "Enable CUDA-accelerated computations (required for parallel/PSO)" ON)
option(ENABLE_SUNDIALS "Enable SUNDIALS integration with CUDA (requires ENABLE_CUDA)" OFF)

# Enforce dependency constraints
if(ENABLE_SUNDIALS AND NOT ENABLE_CUDA)
    message(FATAL_ERROR 
        "ENABLE_SUNDIALS requires ENABLE_CUDA to be ON. "
        "Either set ENABLE_CUDA=ON or set ENABLE_SUNDIALS=OFF.")
endif()

# Display feature status
message(STATUS "=== Feature Configuration ===")
message(STATUS "CUDA support: ${ENABLE_CUDA}")
message(STATUS "SUNDIALS integration: ${ENABLE_SUNDIALS}")
message(STATUS "CGPOPS support: ${ENABLE_CGPOPS}")
message(STATUS "=============================")

set(CGPOPS_ROOT "/home/leo8ner/cgpops" CACHE PATH 
    "Root directory of CGPOPS installation")
set(SUNDIALS_ROOT "/usr/local/sundials-double" CACHE PATH 
    "Root directory of SUNDIALS installation")
set(IPOPT_INCLUDE_PATH "/usr/include/coin" CACHE PATH 
    "IPOPT include directory")
set(CASADI_INCLUDE_PATH "/usr/local/include/casadi" CACHE PATH 
    "CasADi include directory")
set(CASADI_LIBRARY_PATH "" CACHE PATH 
    "CasADi library directory (leave empty for auto-detection)")
set(IPOPT_LIBRARY_PATH "" CACHE PATH 
    "IPOPT library directory (leave empty for auto-detection)")

# Derived paths
set(CGPOPS_SRC_DIR "${CGPOPS_ROOT}/src")
set(SUNDIALS_INCLUDE_DIR "${SUNDIALS_ROOT}/include")
set(SUNDIALS_LIBRARIES_DIR "${SUNDIALS_ROOT}/lib")

# ============================================================================
# Compiler Configuration
# ============================================================================

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Linker flags for runtime library path
set(CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# Display compiler information
if(CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# ============================================================================
# External Dependencies
# ============================================================================

# --- CasADi ---
find_package(casadi REQUIRED)
if(NOT casadi_FOUND)
    message(FATAL_ERROR "CasADi not found. Please install CasADi or set CASADI_DIR.")
else()
    message(STATUS "CasADi version: ${casadi_VERSION}")
endif()

# Find CasADi library with configurable path support
find_library(CASADI_LIBRARY
    NAMES casadi
    HINTS 
        ${CASADI_LIBRARY_PATH}              # User-specified path (highest priority)
        ${CASADI_INCLUDE_PATH}/../lib       # Auto-detected from include path
        $ENV{CASADI_PREFIX}/lib             # Environment variable fallback
    PATHS /usr/local/lib /usr/lib           # System-wide fallback
)
if(CASADI_LIBRARY)
    set(CASADI_LIBRARIES ${CASADI_LIBRARY})
else()
    message(FATAL_ERROR "CasADi library not found.")
endif()

# --- SUNDIALS ---
if(ENABLE_SUNDIALS)
    find_package(SUNDIALS REQUIRED)
    if(NOT SUNDIALS_FOUND)
        message(FATAL_ERROR "SUNDIALS not found. Please install SUNDIALS or set SUNDIALS_DIR.")
    else()
        message(STATUS "SUNDIALS version: ${SUNDIALS_VERSION}")
    endif()

    # Find all required SUNDIALS libraries
    find_library(SUNDIALS_CVODES_LIBRARY NAMES sundials_cvodes 
        HINTS ${SUNDIALS_LIBRARIES_DIR})
    find_library(SUNDIALS_NVECCUDA_LIBRARY NAMES sundials_nveccuda 
        HINTS ${SUNDIALS_LIBRARIES_DIR})
    find_library(SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY NAMES sundials_sunmatrixcusparse 
        HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib)
    find_library(SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY NAMES sundials_sunlinsolcusolversp 
        HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib)
    find_library(SUNDIALS_GENERIC_LIBRARY NAMES sundials_generic 
        HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib)

    if(SUNDIALS_CVODES_LIBRARY AND SUNDIALS_NVECCUDA_LIBRARY AND 
       SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY AND SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY AND
       SUNDIALS_GENERIC_LIBRARY)
        set(SUNDIALS_LIBRARIES 
            ${SUNDIALS_CVODES_LIBRARY}
            ${SUNDIALS_NVECCUDA_LIBRARY}
            ${SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY}
            ${SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY}
            ${SUNDIALS_GENERIC_LIBRARY})
        message(STATUS "All SUNDIALS libraries found:")
        message(STATUS "  CVODES: ${SUNDIALS_CVODES_LIBRARY}")
        message(STATUS "  NVECCUDA: ${SUNDIALS_NVECCUDA_LIBRARY}")
        message(STATUS "  SUNMATRIXCUSPARSE: ${SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY}")
        message(STATUS "  SUNLINSOLCUSOLVERSP: ${SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY}")
        message(STATUS "  GENERIC: ${SUNDIALS_GENERIC_LIBRARY}")
    else()
        message(FATAL_ERROR "Some SUNDIALS libraries not found. Check installation.")
    endif()
else()
    # Set empty SUNDIALS_LIBRARIES to avoid undefined variable errors
    set(SUNDIALS_LIBRARIES "")
    message(STATUS "SUNDIALS integration disabled")
endif()

# --- CUDA ---
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CUDA support disabled - skipping parallel and PSO targets")
endif()

# --- Threading ---
find_package(Threads REQUIRED)

# --- IPOPT (required for CGPOPS) ---
if(ENABLE_CGPOPS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(IPOPT_PKG REQUIRED ipopt)
    
    # Find IPOPT library with configurable path support
    find_library(IPOPT_LIBRARY 
        NAMES ipopt
        HINTS 
            ${IPOPT_LIBRARY_PATH}               # User-specified path (highest priority)
            ${IPOPT_PKG_LIBRARY_DIRS}           # pkg-config detected paths
        PATHS 
            /usr/local/lib 
            /usr/lib 
            /opt/lib                            # Standard system paths
    )

    if(IPOPT_LIBRARY)
        set(IPOPT_LIBRARIES ${IPOPT_LIBRARY})
    else()
        message(FATAL_ERROR 
            "IPOPT library not found. "
            "Set IPOPT_LIBRARY_PATH, configure PKG_CONFIG_PATH, or disable CGPOPS with -DENABLE_CGPOPS=OFF")
    endif()
    
    message(STATUS "CGPOPS support enabled")
endif()

# ============================================================================
# Library Targets
# ============================================================================

# --- TOAC Static Library ---
# Select source files based on SUNDIALS configuration
if(ENABLE_SUNDIALS)
    set(TOAC_INTEGRATION_SOURCES
        src/lib/toac/CUDAoptimizer.cpp
        src/lib/toac/CUDAdynamics.cu
        src/lib/toac/external_cvodes.cpp
        src/lib/toac/CustomCallback.cpp
    )
elseif(ENABLE_CUDA)
    set(TOAC_INTEGRATION_SOURCES
        src/lib/toac/CUDAoptimizer.cpp
        src/lib/toac/cudaRK4.cu
        src/lib/toac/external_rk4.cpp
    )
else()
    set(TOAC_INTEGRATION_SOURCES)
endif()

# Base sources (always included)
set(TOAC_BASE_SOURCES
    src/lib/toac/optimizer.cpp 
    src/lib/toac/dynamics.cpp 
    src/lib/toac/lhs.cpp
    src/lib/helper_functions.cpp
)

# Add CUDA-dependent sources conditionally
if(ENABLE_CUDA)
    list(APPEND TOAC_BASE_SOURCES
        src/lib/toac/pso.cu
    )
endif()

add_library(toac STATIC 
    ${TOAC_BASE_SOURCES}
    ${TOAC_INTEGRATION_SOURCES}
)

if(ENABLE_CUDA)
    set_property(TARGET toac PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(toac PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
)

# Add SUNDIALS include directories conditionally
if(ENABLE_SUNDIALS)
    target_include_directories(toac PUBLIC ${SUNDIALS_INCLUDE_DIR})
endif()

# Add CUDA include directories conditionally
if(ENABLE_CUDA)
    target_include_directories(toac PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(toac PUBLIC
    ${CASADI_LIBRARIES}
)

# Add SUNDIALS libraries conditionally
if(ENABLE_SUNDIALS)
    target_link_libraries(toac PUBLIC ${SUNDIALS_LIBRARIES})
endif()

# Add CUDA libraries conditionally
if(ENABLE_CUDA)
    target_link_libraries(toac PUBLIC
        CUDA::cudart
        CUDA::cusparse
        CUDA::cusolver
    )
endif()

target_compile_options(toac PRIVATE
    -Wno-unused-parameter
    -Wno-unused-result
    -Wno-reorder
)

if(ENABLE_CUDA)
    target_compile_options(toac PRIVATE -Wno-deprecated-gpu-targets)

    # --- TOAC Shared Library ---
    add_library(toac_shared SHARED 
        ${TOAC_INTEGRATION_SOURCES}
        src/lib/helper_functions.cpp
    ) 

    set_property(TARGET toac_shared PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(toac_shared PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(toac_shared PUBLIC
        ${CASADI_LIBRARIES}
        CUDA::cudart
        CUDA::cusparse
        CUDA::cusolver
    )

    if(ENABLE_SUNDIALS)
        target_include_directories(toac_shared PUBLIC ${SUNDIALS_INCLUDE_DIR})
        target_link_libraries(toac_shared PUBLIC ${SUNDIALS_LIBRARIES})
    endif()

    target_compile_options(toac_shared PRIVATE
        -Wno-unused-parameter
        -Wno-unused-result
        -Wno-deprecated-gpu-targets
        -Wno-reorder

    )

    set_target_properties(toac_shared PROPERTIES
        VERSION 1.0
        SOVERSION 1
        PREFIX "lib"
    )
endif()

# --- CGPOPS Library (Conditional) ---
if(ENABLE_CGPOPS)
    # Collect external CGPOPS source files
    file(GLOB CGPOPS_EXT_SOURCES 
        "${CGPOPS_SRC_DIR}/cgpopsClassFuncDef.cpp"
        "${CGPOPS_SRC_DIR}/MatClasses.cpp"
        "${CGPOPS_SRC_DIR}/ASCIItable.cpp"
        "${CGPOPS_SRC_DIR}/Bicomplex.cpp"
        "${CGPOPS_SRC_DIR}/HyperDual.cpp"
        "${CGPOPS_SRC_DIR}/LGRClass.cpp"
        "${CGPOPS_SRC_DIR}/cgpopsFuncDef.cpp"
        "${CGPOPS_SRC_DIR}/NLPDerivativesHD.cpp"
        "${CGPOPS_SRC_DIR}/NLPDerivativesBC.cpp"
        "${CGPOPS_SRC_DIR}/NLPDerivativesCD.cpp"
        "${CGPOPS_SRC_DIR}/cgpops_nlp.cpp"
        "${CGPOPS_SRC_DIR}/cgpopsMeshRef.cpp"
        "${CGPOPS_SRC_DIR}/cgpopsHamiltonianDef.cpp"
        "${CGPOPS_SRC_DIR}/find_polynomial_roots_jenkins_traub.cc"
        "${CGPOPS_SRC_DIR}/polynomial.cc"
        "${CGPOPS_SRC_DIR}/OCPFunctions.cpp"
    )

    add_library(cgpops STATIC 
        src/lib/cgpops/cgpops_main.cpp
        src/lib/cgpops/cgpops_gov.cpp
        ${CGPOPS_EXT_SOURCES}
    )

    target_include_directories(cgpops PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/cgpops
        ${CMAKE_SOURCE_DIR}/include/toac
        ${CGPOPS_ROOT}
        ${CGPOPS_SRC_DIR}
        ${IPOPT_PKG_INCLUDE_DIRS}
    )

    target_compile_definitions(cgpops PRIVATE
        IPOPT_BUILD
        SRC_PATH="${CGPOPS_SRC_DIR}"
    )

    target_link_libraries(cgpops PUBLIC
        ${IPOPT_PKG_LIBRARIES}
        toac
    )

    target_link_directories(cgpops PUBLIC
        ${IPOPT_PKG_LIBRARY_DIRS}
    )

    target_compile_options(cgpops PRIVATE
        -O3 -pipe
        -Wno-unknown-pragmas
        -Wno-unused-result
        -Wno-long-long
        -Wno-unused-parameter
        -Wno-maybe-uninitialized
        -Wno-unused-but-set-variable
        -Wno-deprecated-declarations
        -Wno-format-truncation
        -Wno-ignored-qualifiers
        -Wno-unused-variable
        -Wno-deprecated-enum-enum-conversion
        -Wno-int-in-bool-context
        -Wno-deprecated-copy
        ${IPOPT_PKG_CFLAGS_OTHER}
    )
endif()

# ============================================================================
# Helper Functions
# ============================================================================

# Function to create Monte Carlo executables with consistent configuration
function(add_montecarlo_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TARGET_NAME} PRIVATE 
        ${CASADI_LIBRARIES} 
        toac
    )
    
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )
endfunction()

# ============================================================================
# Serial Executables
# ============================================================================

add_executable(SerialGenerateCode src/serial/c_code_gen.cpp)
target_link_libraries(SerialGenerateCode PRIVATE 
    ${CASADI_LIBRARIES} 
    toac
)
target_include_directories(SerialGenerateCode PRIVATE 
    ${CASADI_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
)

add_executable(SerialMain src/serial/main.cpp)
target_link_libraries(SerialMain PRIVATE 
    ${CASADI_LIBRARIES} 
    toac
)
target_include_directories(SerialMain PRIVATE 
    ${CASADI_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
)

add_executable(SerialCodeGen src/serial/main_code_gen.cpp)
target_link_libraries(SerialCodeGen PRIVATE 
    ${CASADI_LIBRARIES} 
    toac
)
target_include_directories(SerialCodeGen PRIVATE 
    ${CASADI_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
)

# PSO executables (require CUDA)
if(ENABLE_CUDA)
    add_executable(PSOwithSerial src/serial/main_pso.cpp)
    target_link_libraries(PSOwithSerial PRIVATE 
        ${CASADI_LIBRARIES} 
        toac 
        CUDA::cudart 
        CUDA::curand
    ) 
    target_include_directories(PSOwithSerial PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )

    add_executable(PSO src/serial/pso_only.cpp)
    target_link_libraries(PSO PRIVATE 
        ${CASADI_LIBRARIES} 
        toac 
        CUDA::cudart 
        CUDA::curand
    ) 
    target_include_directories(PSO PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )
else()
    message(STATUS "Skipping PSO executables (CUDA disabled)")
endif()

# ============================================================================
# Parallel Executables (Require CUDA)
# ============================================================================

if(ENABLE_CUDA)
    add_executable(ParGenerateCode src/parallel/c_code_gen.cpp)
    target_link_libraries(ParGenerateCode PRIVATE
        ${CASADI_LIBRARIES}
        CUDA::cudart
        CUDA::cusparse
        CUDA::cusolver
        toac
    )
    target_include_directories(ParGenerateCode PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )

    add_executable(ParallelMain src/parallel/main.cpp)
    target_link_libraries(ParallelMain PRIVATE
        ${CASADI_LIBRARIES}
        CUDA::cudart
        CUDA::cusparse
        CUDA::cusolver
        toac
    )
    target_include_directories(ParallelMain PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )

    add_executable(ParallelCodeGen src/parallel/main_code_gen.cpp)
    target_link_libraries(ParallelCodeGen PRIVATE 
        ${CASADI_LIBRARIES} 
        ${SUNDIALS_LIBRARIES} 
        toac
    )
    target_include_directories(ParallelCodeGen PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
    )

    if (ENABLE_SUNDIALS)
        target_link_libraries(ParGenerateCode PRIVATE ${SUNDIALS_LIBRARIES})
        target_include_directories(ParGenerateCode PRIVATE ${SUNDIALS_INCLUDE_DIR})

        target_link_libraries(ParallelMain PRIVATE ${SUNDIALS_LIBRARIES})
        target_include_directories(ParallelMain PRIVATE ${SUNDIALS_INCLUDE_DIR})

        target_link_libraries(ParallelCodeGen PRIVATE ${SUNDIALS_LIBRARIES})
        target_include_directories(ParallelCodeGen PRIVATE ${SUNDIALS_INCLUDE_DIR})
    endif()
else()
    message(STATUS "Skipping parallel executables (CUDA disabled)")
endif()

# ============================================================================
# Monte Carlo Simulation Executables
# ============================================================================

# CUDA-independent Monte Carlo executables (always compiled)
add_montecarlo_executable(GenerateSamples src/montecarlo/sample_generator.cpp)
add_montecarlo_executable(MCS_NO_PSO src/montecarlo/mcs_no_pso.cpp)

add_executable(plot_mcs src/montecarlo/mcs_plotter.cpp)
add_executable(GetPSOparams src/montecarlo/pso_params.cpp)

add_executable(GeneratePSOParamsSamples src/montecarlo/pso_sample_gen.cpp)
target_link_libraries(GeneratePSOParamsSamples PRIVATE toac)
target_include_directories(GeneratePSOParamsSamples PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
)

# PSO-related Monte Carlo executables (require CUDA)
if(ENABLE_CUDA)
    add_montecarlo_executable(MCS_PSO_FULL src/montecarlo/mcs_pso_full.cpp)
    add_montecarlo_executable(MCS_PSO_STO src/montecarlo/mcs_pso_sto.cpp)
    add_montecarlo_executable(PSO_TUNER src/montecarlo/pso_tuner.cpp)
    add_montecarlo_executable(PSO_TIME src/montecarlo/pso_time.cpp)
else()
    message(STATUS "Skipping PSO-related Monte Carlo executables (CUDA disabled)")
endif()

# ============================================================================
# CGPOPS-Dependent Executables (Conditional)
# ============================================================================

if(ENABLE_CGPOPS)
    add_executable(MCS_CGPOPS src/montecarlo/mcs_cgpops.cpp)
    target_link_libraries(MCS_CGPOPS PRIVATE 
        ${CASADI_LIBRARIES} 
        cgpops 
        toac
    )
    target_include_directories(MCS_CGPOPS PRIVATE 
        ${CASADI_INCLUDE_PATH}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/toac
        ${CMAKE_SOURCE_DIR}/include/cgpops
    )
    target_compile_options(MCS_CGPOPS PRIVATE
        -Wno-ignored-qualifiers
        -Wno-deprecated-declarations
    )

    add_executable(run_cgpops src/cgpops/cgpops_wrapper.cpp)
    target_include_directories(run_cgpops PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/cgpops
        ${CGPOPS_SRC_DIR}
        ${CMAKE_SOURCE_DIR}/include/toac
    )
    target_link_libraries(run_cgpops PRIVATE
        cgpops
        ${CASADI_LIBRARIES}
    )
    target_compile_options(run_cgpops PRIVATE
        -Wno-deprecated-declarations
        -Wno-ignored-qualifiers
        -Wno-unused-result
    )
else()
    message(STATUS "CGPOPS support disabled - skipping MCS_CGPOPS and run_cgpops targets")
endif()

# ============================================================================
# Custom Build Targets
# ============================================================================

# Serial code generation and execution (always available)
add_custom_target(run_codegen_serial
    COMMAND SerialGenerateCode
    DEPENDS toac SerialGenerateCode
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running SerialGenerateCode for serial implementation"
)

add_custom_target(serial 
    COMMAND SerialCodeGen
    DEPENDS SerialCodeGen run_codegen_serial
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running SerialCodeGen"
)

# Parallel code generation and execution (require CUDA)
if(ENABLE_CUDA)
    add_custom_target(run_codegen_parallel
        COMMAND ParGenerateCode
        DEPENDS toac ParGenerateCode
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running ParGenerateCode for parallel implementation"
    )

    add_custom_target(parallel 
        COMMAND ParallelCodeGen
        DEPENDS run_codegen_parallel ParallelCodeGen
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running ParallelCodeGen"
    )
else()
    message(STATUS "Skipping parallel and PSO custom targets (CUDA disabled)")
endif()

# ============================================================================
# Master Build & Code Generation Target
# ============================================================================

# Compile everything and generate code (serial always, parallel if CUDA enabled)
if(ENABLE_CUDA)
    add_custom_target(everything
        COMMAND ${CMAKE_COMMAND} --build . --target all
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running serial code generation ==="
        COMMAND ${CMAKE_COMMAND} --build . --target run_codegen_serial
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running parallel code generation ==="
        COMMAND ${CMAKE_COMMAND} --build . --target run_codegen_parallel
        COMMAND ${CMAKE_COMMAND} -E echo "=== Build and code generation complete ==="
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building all targets and running code generation for serial and parallel"
    )
else()
    add_custom_target(everything
        COMMAND ${CMAKE_COMMAND} --build . --target all
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running serial code generation ==="
        COMMAND ${CMAKE_COMMAND} --build . --target run_codegen_serial
        COMMAND ${CMAKE_COMMAND} -E echo "=== Build and code generation complete ==="
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building all targets and running code generation for serial only"
    )
endif()