#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/dynamics.h>


namespace fs = std::filesystem;


int main(){

    // Dynamics
    Dynamics dyn; // Create an instance of the Dynamics class

    // options for c-code auto generation
    casadi::Dict opts = casadi::Dict();
    opts["cpp"] = false;
    opts["with_header"] = true;
    // prefix for c code
    std::string prefix_code = fs::current_path().parent_path().string() + "/code_gen/";

    // generate functions in c code
    casadi::CodeGenerator myCodeGen = casadi::CodeGenerator("dynamics.c", opts);
    myCodeGen.add(dyn.F);
    myCodeGen.add(dyn.jac_F);
    myCodeGen.add(dyn.jac_jac_F);

    myCodeGen.generate(prefix_code);

    // compile c code to a shared library
    std::string prefix_lib = fs::current_path().parent_path().string() + "/build/";
    std::string compile_command = "gcc -fPIC -shared -O3 " + 
        prefix_code + "dynamics.c -o " +
        prefix_lib + "lib_dynamics.so";

    std::cout << compile_command << std::endl;

    int compile_flag = std::system(compile_command.c_str());
    casadi_assert(compile_flag==0, "Compilation failed!");
    std::cout << "Compilation succeeded!" << std::endl;

    return 0;
}