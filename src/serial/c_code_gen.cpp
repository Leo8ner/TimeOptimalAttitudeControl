#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/dynamics.h>
#include <toac/optimizer.h>


namespace fs = std::filesystem;


int main(){

    std::string plugin = "fatrop"; // Specify the solver plugin to use
    std::string method = "shooting"; // Specify the integration method to use
    bool fixed_step = true; // Use fixed step size for the integrator

    // Dynamics
    Dynamics dyn(plugin, method); // Create an instance of the Dynamics class

    // Solver
    Optimizer opti(dyn, fixed_step);     // Create an instance of the Optimizer class

    // options for c-code auto generation
    Dict opts = Dict();
    opts["cpp"] = false;
    opts["with_header"] = true;
    // prefix for c code
    std::string prefix_code = fs::current_path().parent_path().string() + "/code_gen/";

    // generate dynamics in c code
    casadi::CodeGenerator myCodeGen = casadi::CodeGenerator("solver.c", opts);
    myCodeGen.add(opti.solver);
    myCodeGen.generate(prefix_code);

    // compile c code to a shared library
    std::string prefix_lib = fs::current_path().parent_path().string() + "/build/";
    std::string compile_command = "gcc -fPIC -shared -O3 " + 
        prefix_code + "solver.c -o " +
        prefix_lib + "lib_solver.so -lfatrop -lipopt";
    std::cout << compile_command << std::endl;

    int compile_flag = std::system(compile_command.c_str());
    casadi_assert(compile_flag==0, "Compilation failed!");
    std::cout << "Compilation succeeded!" << std::endl;

    return 0;
}