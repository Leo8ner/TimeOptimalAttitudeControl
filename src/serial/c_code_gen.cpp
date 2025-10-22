#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <toac/dynamics.h>
#include <toac/optimizer.h>

namespace fs = std::filesystem;

void add_stats_prints(const std::string& filepath) {
    // Read the solver.c file
    std::ifstream file_in(filepath);
    if (!file_in.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << std::endl;
        return;
    }
    
    std::stringstream buffer;
    buffer << file_in.rdbuf();
    std::string content = buffer.str();
    file_in.close();
    
    // Find the existing printf line to locate where to insert
    std::string marker = "  d->stats.return_flag = stats->return_flag;";
    
    if (content.find(marker) == std::string::npos) {
        std::cerr << "Error: Could not find the marker line in solver.c" << std::endl;
        return;
    }
    
    // Define all the printf statements for stats structure fields
    std::string print_statements = 
        "  d->stats.return_flag = stats->return_flag;\n"
        "  printf(\"return_flag %d\\n\", stats->return_flag);\n"
        "  printf(\"unified_return_status %d\\n\", d->unified_return_status);\n"
        "  printf(\"success %d\\n\", d->success);\n";

    // Replace the single line with all print statements
    size_t pos = content.find(marker);
    content.replace(pos, marker.length(), print_statements);
    
    // Write back to solver.c
    std::ofstream file_out(filepath);
    if (!file_out.is_open()) {
        std::cerr << "Error: Could not write to file: " << filepath << std::endl;
        return;
    }
    
    file_out << content;
    file_out.close();
    
    std::cout << "Successfully added print statements for all stats fields!" << std::endl;
}

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

    // Add stats print statements to the generated code
    if (plugin == "fatrop") {
        std::string solver_path = prefix_code + "solver.c";
        add_stats_prints(solver_path);
    }

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