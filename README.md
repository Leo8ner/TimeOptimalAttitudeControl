# Time-Optimal Attitude Control (TOAC)

A high-performance C++/CUDA framework for real-time, time-optimal spacecraft attitude control trajectory generation using advanced numerical optimization techniques.

## Dependencies

### Required
- **CMake** ≥ 3.20
- **GCC/G++** with C++20 support
- **CasADi** ≥ 3.5 (symbolic optimization framework)
- **FATROP** (fast nonlinear optimizer)

### Optional
- **CUDA Toolkit** ≥ 11.0 (for GPU acceleration and parallel execution)
- **SUNDIALS** ≥ 6.0 with CUDA support (for CVODES integration)
- **IPOPT** ≥ 3.14 (alternative NLP solver)
- **CGPOPS** (pseudospectral optimal control solver)

## Installation

### 1. Install Dependencies

#### Ubuntu/Debian
```bash
# Core dependencies
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config

# CasADi (install from source or package manager)
# Follow: https://github.com/casadi/casadi/wiki/InstallationLinux
# Use the 

cmake .. \
  -DWITH_IPOPT=ON -DWITH_BUILD_IPOPT=ON \
  -DWITH_BUILD_MUMPS=ON \
  -DWITH_FATROP=ON -DWITH_BUILD_FATROP=ON \
  -DWITH_PYTHON=ON -DWITH_PYTHON3=ON \
  -DPYTHON_PREFIX=$(python -c 'import site; print(site.getsitepackages()[0])') \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_OPENMP=ON \
  -DWITH_BLASFEO=ON -DWITH_BUILD_BLASFEO=ON \
  -DWITH_BUILD_EIGEN3=ON \
  -DWITH_MUMPS=ON \
  -DWITH_THREAD=ON \
  -DWITH_OPENCL=ON \
  -DWITH_BUILD_REQUIRED=ON \

# On this repository, run cmake without specifying the path. It should find it automatically.
# If it doenst work run cmake with the -DCASADI_ROOT=[PATH_TO_YOUR_CASADI_LIBRARIES] or change the CMakeLists.txt for a more permanent alternative

## Install CUDA Toolkit (optional, for GPU support)
# Run
ubuntu-drivers devices

### **Example Output**
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00002484sv00001043sd0000888Abc03sc00i00
vendor   : NVIDIA Corporation
model    : GA104 [GeForce RTX 3070]
driver   : nvidia-driver-535 - distro non-free
driver   : nvidia-driver-545 - distro non-free
driver   : nvidia-driver-550 - distro non-free
driver   : nvidia-driver-570 - distro non-free recommended # Find the recommended driver (560)
driver   : xserver-xorg-video-nouveau - distro free builtin

# Follow: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html to identify the most recent compatible CUDA Toolkit with your driver version
# Follow: https://developer.nvidia.com/cuda-toolkit-archive to find the appropriate CUDA Toolkit 
# See if the release is compatible with your ubuntu version
# If not, go for an older version or update your ubuntu
# Follow: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages and follow the installation instructions for deb(local)
# Once you install the CUDA Toolkit install the correspondent drivers. Make sure it corresponds to the driver number in your installation instructions:

# Make sure it corresponds to the driver number in your installation instructions:
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pinsudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.debsudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.debsudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/sudo apt-get updatesudo apt-get -y install cuda-toolkit-12-8

sudo apt-get -y install cuda-drivers-570  # Installs driver 570.x 

# Reboot
sudo reboot now
 
# CGPOPS (optional)
# Download the provided software.
# After cloning the repository of this thesis, replace the file cgpopsFuncDef.cpp in the cgpops folder with the one available in the TimeOptimalAttitudeControl/src/cgpops
# On this repository, run cmake with the -DCGPOPS_ROOT=[PATH_TO_YOUR_CGPOPS_FOLDER] or change the CMakeLists.txt for a more permanent alternative


# IPOPT (mandatory for CGPOPS)
# Install IPOPT from source with the MUMPS linear solver
# Follow: https://coin-or.github.io/Ipopt/INSTALL.html
# On this repository, run cmake without specifying the path. It should find it automatically.
# If it doenst work run cmake with the -DIPOPT_ROOT=[PATH_TO_YOUR_IPOPT_LIBRARIES] or change the CMakeLists.txt for a more permanent alternative

# SUNDIALS with CUDA (optional, build from source)
# Follow: https://sundials.readthedocs.io/en/latest/sundials/Install_link.html
# On this repository, run cmake without specifying the path. It should find it automatically.
# If it doenst work run cmake with the -DSUNDIALS_ROOT=[PATH_TO_YOUR_SUNDIALS_LIBRARIES] or change the CMakeLists.txt for a more permanent alternative

```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/TimeOptimalAttitudeControl.git
cd TimeOptimalAttitudeControl
```

### 3. Build Project

#### Standard Build (CGPOPS enabled, CUDA enabled, SUNDIALS disabled)
```bash
mkdir code_gen && mkdir build && cd build
cmake ..
make -j$(nproc)
```

#### Build Options with Default Values
```bash
  -DENABLE_CUDA=ON
  -DENABLE_CGPOPS=ON
  -DENABLE_SUNDIALS=OFF
  -DCASADI_INCLUDE_PATH=/opt/casadi/include/casadi \
  -DCASADI_LIBRARY_PATH= \
  -DCGPOPS_ROOT=/home/leo8ner/cgpops
  -DSUNDIALS_ROOT=/usr/local/sundials-double
  -DIPOPT_INCLUDE_PATH=/usr/include/coin
  -DIPOPT_LIBRARY_PATH=

```

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | `ON` | Enable CUDA acceleration for parallel/PSO algorithms |
| `ENABLE_SUNDIALS` | `OFF` | Enable SUNDIALS CVODES integration (requires CUDA) |
| `ENABLE_CGPOPS` | `ON` | Enable CGPOPS pseudospectral solver |
| `CASADI_INCLUDE_PATH` | `/usr/local/include/casadi` | CasADi header directory |
| `CASADI_LIBRARY_PATH` | (auto-detect) | CasADi library directory |
| `SUNDIALS_ROOT` | `/usr/local/sundials-double` | SUNDIALS installation root |
| `IPOPT_INCLUDE_PATH` | `/usr/include/coin` | IPOPT header directory |
| `IPOPT_LIBRARY_PATH` | (auto-detect) | IPOPT library directory |
| `CGPOPS_ROOT` | `/home/leo8ner/cgpops` | CGPOPS installation root |


#### Build without CUDA (Serial only)
```bash
cmake -DENABLE_CUDA=OFF ..
make -j$(nproc)
```

#### Build with SUNDIALS Integration
```bash
cmake \
  -DENABLE_SUNDIALS=ON \
  -DSUNDIALS_ROOT=/usr/local/sundials-double \
  ..
make -j$(nproc)
```

#### Build with CGPOPS Support
```bash
cmake \
  -DENABLE_CGPOPS=ON \
  -DCGPOPS_ROOT=/path/to/cgpops \
  -DIPOPT_INCLUDE_PATH=/usr/include/coin \
  -DIPOPT_LIBRARY_PATH=/usr/lib \
  ..
make -j$(nproc)
```

### Configuration Examples
```bash
# Minimal build (serial only, no GPU)
cmake -DENABLE_CUDA=OFF -DENABLE_CGPOPS=OFF ..

# All features
cmake -DENABLE_CUDA=ON -DENABLE_SUNDIALS=ON -DENABLE_CGPOPS=ON ..

## Usage

### Quick Start

#### Generate Trajectory
```bash
cd build
make
```

This will:
1. Compile all the code without C-code generation


#### Generate Trajectory
```bash
cd build
make everything
```

This will:
1. Compile all the code
2. Generate optimized C from serial implementation
3. Generate optimized C from parallel implementation (if CUDA is enabled)

#### Run Serial C code generation
```bash
cd build
make serial 
```

This will:
1. Compile all the serial code
2. Generate optimized C from serial implementation
3. Try to run ./SerialCodeGen executable

#### Run Parallel GPU-Accelerated Solver (Only if CUDA is enabled)
```bash
make parallel 
```

This will:
1. Compile all the parallel code
2. Generate optimized C from parallel implementation
3. Try to run ./ParallelCodeGen executable

#### Run Monte Carlo Analysis
```bash
# Generate random parameters using LHS
./GenerateSamples
./GeneratePSO

# Run Monte Carlo with different methods
./MCS_NO_PSO          # Direct transcription only
./MCS_PSO_FULL        # PSO with full control parameterization + direct transcription
./MCS_PSO_STO         # PSO with STO control parameterization + direct transcription
./MCS_CGPOPS          # CGPOPS pseudospectral method

# Visualize results
./plot_mcs
```
### INPUTS ###

Every executable accepts an initial state and final state as inputs.
They should correspond to:
 "phi_i,theta_i,psi_i,wx_i,wy_i,wz_i" "phi_f,theta_f,psi_f,wx_f,wy_f,wz_f" all in degrees(/s)

```bash
#### Run Single CGPOPS
./run_cgpops "0,0,0,0,0,0" "180,0,0,0,0,0"

#### Run Optimized c-code Serial
./SerialCodeGen "0,0,0,0,0,0" "180,0,0,0,0,0"

#### Run Optimized c-code Parallel
./ParallelCodeGen "0,0,0,0,0,0" "180,0,0,0,0,0"

#### Run standalone PSO
./PSO "0,0,0,0,0,0" "180,0,0,0,0,0"

#### Run PSO with optimized serial solver
./PSOwithSerial "0,0,0,0,0,0" "180,0,0,0,0,0"

#### Run PSO Tuning
./PSO_TIME
./PSO_TUNER
./GetPSOparams

### Generate 1000 LHS Samples
./GenerateSamples 1000
./GeneratePSOParamsSamples 1000
```
### Custom Targets

| Target | Description |
|--------|-------------|
| `make serial` | Generate code + run serial solver |
| `make parallel` | Generate code + run parallel solver |
| `make everything` | Build all + run code generation |
| `make run_codegen_serial` | Generate serial solver code only |
| `make run_codegen_parallel` | Generate parallel solver code only |

### Executables

#### Serial Execution
- `SerialGenerateCode` - Generate optimized C solver code 
- `SerialMain` - Run single trajectory optimization without optimized C code
- `SerialCodeGen` - Run optimization with generated C code

#### Parallel Execution (requires CUDA)
- `ParGenerateCode` - Generate parallel solver C code
- `ParallelMain` - Run parallel trajectory optimization without C code
- `ParallelCodeGen` - Run parallel optimization with generated code

#### PSO Algorithms (requires CUDA)
- `PSO` - Standalone PSO optimization
- `PSOwithSerial` - PSO warm-start + NLP refinement

#### Monte Carlo Analysis
- `GenerateSamples` - Generate initial conditions using LHS
- `GeneratePSOParamsSamples` - Generates pso parameter sets using LHS
- `MCS_NO_PSO` - Baseline direct transcription
- `MCS_PSO_FULL` - Full PSO initialization
- `MCS_PSO_STO` - Stochastic PSO variant
- `PSO_TUNER` - PSO hyperparameter optimization
- `PSO_TIME` - PSO timing benchmarks
- `GetPSOparams` - Gives the best PSO parameters out of the PSO_TUNER
- `plot_mcs` - Visualization tool

#### CGPOPS Integration (requires IPOPT and CGPOPS)
- `run_cgpops` - CGPOPS pseudospectral solver
- `MCS_CGPOPS` - Monte Carlo with CGPOPS

## Project Structure
```
TimeOptimalAttitudeControl/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/                    # Header files
│   ├── cgpops/                 # CGPOPS interface headers
│   │   ├── cgpops_gov.hpp
│   │   ├── cgpops_main.hpp
│   │   └── cgpopsAuxDec.hpp
│   └── toac/                   # TOAC library headers
│       ├── casadi_callback.h   
│       ├── cuda_dynamics.h
│       ├── cuda_optimizer.h
│       ├── dynamics.h
│       ├── external_dyn.h
│       ├── optimizer.h
│       └── pso.h
├── src/                        # Source files
│   ├── lib/                    # Library implementations
│   │   ├── cgpops/             # CGPOPS integration
│   │   │   ├── cgpops_gov.cpp
│   │   │   └── cgpops_main.cpp
│   │   └── toac/               # Core TOAC library
│   │       ├── CUDAoptimizer.cpp
│   │       ├── CUDAdynamics.cu     # SUNDIALS integration
│   │       ├── cudaRK4.cu          # GPU RK4 integrator
│   │       ├── CustomCallback.cpp
│   │       ├── dynamics.cpp
│   │       ├── external_cvodes.cpp # SUNDIALS wrapper
│   │       ├── external_rk4.cpp    # RK4 wrapper
│   │       ├── lhs.cpp             
│   │       ├── optimizer.cpp
│   │       └── pso.cu              # PSO implementation
│   ├── serial/                 # Serial executables
│   │   ├── c_code_gen.cpp
│   │   ├── main.cpp
│   │   ├── main_code_gen.cpp
│   │   ├── main_pso.cpp
│   │   └── pso_only.cpp
│   ├── parallel/               # Parallel executables
│   │   ├── c_code_gen.cpp
│   │   ├── main.cpp
│   │   └── main_code_gen.cpp
│   ├── montecarlo/             # Monte Carlo analysis
│   │   ├── sample_generator.cpp
│   │   ├── mcs_cgpops.cpp
│   │   ├── mcs_no_pso.cpp
│   │   ├── mcs_pso_full.cpp
│   │   ├── mcs_pso_sto.cpp
│   │   ├── mcs_plotter.cpp
│   │   ├── pso_params.cpp
│   │   ├── pso_sample_gen.cpp
│   │   ├── pso_time.cpp
│   │   └── pso_tuner.cpp
│   └── cgpops/                 # CGPOPS wrapper
│       |── cgpops_wrapper.cpp
|       └── cgpopsFuncDef.cpp  
├── build/                      # Build artifacts (generated)
├── code_gen/                   # Generated solver code
├── input/                      # Input data files
├── output/                     # Output results
├── Results/                    # Analysis results
```

## Troubleshooting

### Common Issues

#### 1. CUDA not found
```bash
# Set CUDA path explicitly
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0 ..
```

#### 2. CasADi library not found
```bash
# Verify CasADi installation
pkg-config --cflags --libs casadi

# Set paths manually
cmake -DCASADI_INCLUDE_PATH=/path/to/include \
      -DCASADI_LIBRARY_PATH=/path/to/lib ..
```

#### 3. SUNDIALS requires CUDA error
```bash
# Either enable CUDA or disable SUNDIALS
cmake -DENABLE_SUNDIALS=OFF ..
```

#### 4. IPOPT not found (CGPOPS disabled)
```bash
# Install IPOPT
sudo apt-get install coinor-libipopt-dev

# Or set custom path
cmake -DIPOPT_LIBRARY_PATH=/usr/local/lib ..

# Or disable CGPOPS
cmake -DENABLE_CGPOPS=OFF ..
```

## Documentation

Detailed documentation for the optimization algorithms and implementation can be found in:
- `papers/` - Related research papers and references
- `Deliverables/` - Technical report and presentations