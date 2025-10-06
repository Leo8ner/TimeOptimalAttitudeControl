#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/cuda_optimizer.h>

using namespace casadi;

int main(){

    // Dynamics
    //Dynamics dyn; // Create an instance of the Dynamics class
    // create_dynamics(true); // Create an instance of the dynamics class
    // Function dyn = get_dynamics(); // Get the dynamics function
    // DynamicsCallback callback("F", true);
    // Function dyn = callback;

    BatchDynamics dyn;

    Function F = dyn.F; // Get the dynamics function from BatchDynamics
    Function jac_F = F.jacobian(); // Get the Jacobian function
    Function jac_jac_F = jac_F.jacobian(); // Get the Hessian function

    // options for c-code auto generation
    Dict opts = Dict();
    opts["cpp"] = false;
    opts["with_header"] = true;
    // prefix for c code
    std::string prefix_code = std::filesystem::current_path().parent_path().string() + "/code_gen/";

    // generate dynamics in c code
    casadi::CodeGenerator myCodeGen = casadi::CodeGenerator("tesdyn.c", opts);
    myCodeGen.add(F);
    myCodeGen.add(jac_F);
    myCodeGen.add(jac_jac_F);
    myCodeGen.generate(prefix_code);
    // compile c code to a shared library
    std::string prefix_lib = std::filesystem::current_path().parent_path().string() + "/build/";
    std::string compile_command = "gcc -fPIC -shared -O3 " + 
        prefix_code + "tesdyn.c -o " +
        prefix_lib + "lib_tesdyn.so " +
        "-lipopt -lfatrop";
    std::cout << compile_command << std::endl;

    int compile_flag = std::system(compile_command.c_str());
    casadi_assert(compile_flag==0, "Compilation failed!");
    std::cout << "Compilation succeeded!" << std::endl;

    return 0;
}