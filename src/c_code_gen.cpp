#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/dynamics.h>
#include <toac/constraints.h>
#include <toac/optimizer.h>


namespace fs = std::filesystem;


int main(){

    // Dynamics
    //Dynamics dyn; // Create an instance of the Dynamics class
    ImplicitDynamics dyn; // Create an instance of the DynCvodes class
    // Constraints
    Constraints cons; // Create an instance of the Constraints class

    // Solver
    Optimizer opti(dyn.F, cons);     // Create an instance of the Optimizer class

    // options for c-code auto generation
    casadi::Dict opts = casadi::Dict();
    opts["cpp"] = false;
    opts["with_header"] = true;
    // prefix for c code
    std::string prefix_code = fs::current_path().parent_path().string() + "/code_gen/";

    // generate dynamics in c code
    casadi::CodeGenerator myCodeGen = casadi::CodeGenerator("solver.c", opts);
    myCodeGen.add(opti.solver);
    //myCodeGen.add(dyn.jac_solver);
    //myCodeGen.add(dyn.jac_jac_solver);
    myCodeGen.generate(prefix_code);

    // compile c code to a shared library
    std::string prefix_lib = fs::current_path().parent_path().string() + "/build/";
    std::string compile_command = "gcc -fPIC -shared -O3 " + 
        prefix_code + "solver.c -o " +
        prefix_lib + "lib_solver.so " +
        "-lipopt";
    std::cout << compile_command << std::endl;

    int compile_flag = std::system(compile_command.c_str());
    casadi_assert(compile_flag==0, "Compilation failed!");
    std::cout << "Compilation succeeded!" << std::endl;

    return 0;
}