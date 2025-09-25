#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/casadi_callback.h>
#include <toac/cuda_optimizer.h>

using namespace casadi;

int main(){

    // Dynamics
    //Dynamics dyn; // Create an instance of the Dynamics class
    // create_dynamics(true); // Create an instance of the dynamics class
    // Function dyn = get_dynamics(); // Get the dynamics function
    // DynamicsCallback callback("F", true);
    // Function dyn = callback;

    Function dyn = external("F", "libtoac_shared.so");

    // Solver
    std::string plugin = "ipopt"; // Specify the solver plugin to use
    Optimizer opti(dyn);     // Create an instance of the
    // options for c-code auto generation
    Dict opts = Dict();
    opts["cpp"] = false;
    opts["with_header"] = true;
    // prefix for c code
    std::string prefix_code = std::filesystem::current_path().parent_path().string() + "/code_gen/";

    // generate dynamics in c code
    casadi::CodeGenerator myCodeGen = casadi::CodeGenerator("parsolver.c", opts);
    myCodeGen.add(opti.solver);
    //myCodeGen.add(dyn.get_jacobian());
    //myCodeGen.add(dyn.get_hessian());
    myCodeGen.generate(prefix_code);
    // compile c code to a shared library
    std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
    std::string compile_command = "gcc -fPIC -shared -O3 " + 
        prefix_code + "parsolver.c -o " +
        prefix_lib + "lib_parsolver.so " +
        "-lipopt -lfatrop -lcasadi -L" + prefix_lib + " -ltoac_shared";
    std::cout << compile_command << std::endl;

    int compile_flag = std::system(compile_command.c_str());
    casadi_assert(compile_flag==0, "Compilation failed!");
    std::cout << "Compilation succeeded!" << std::endl;

    return 0;
}