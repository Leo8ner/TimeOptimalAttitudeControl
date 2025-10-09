#include <iostream>
#include <fstream>
#include <iomanip>
#include <toac/lhs.h>
#include <toac/helper_functions.h>
#include <casadi/casadi.hpp>
using namespace casadi;

int main(int argc, char** argv) {

    int iterations = 1000;
    // Parse command-line argument for iteration count
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <iterations>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1000" << std::endl;
        std::cerr << "Defaulting to 1000 iterations." << std::endl;
    } else {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0) {
            std::cerr << "Error: iterations must be a positive integer" << std::endl;
            std::cerr << "Defaulting to 1000 iterations." << std::endl;
            iterations = 1000;
        }
    }

    std::cout << "Generating " << iterations << " LHS samples..." << std::endl;
    
    try {
        // Sample points using Latin Hypercube Sampling
        LHS lhs(iterations, 6);
        double max_angle = 180.0 * DEG; // rad
        double max_vel = 0.0 * DEG;   // rad/s
        
        std::vector<double> mins = {-max_angle, -max_angle, -max_angle, -max_vel, -max_vel, -max_vel};
        std::vector<double> maxs = { max_angle,  max_angle,  max_angle,  max_vel,  max_vel,  max_vel};
        
        auto initial_states = lhs.sampleBounded(mins, maxs);
        auto final_states = lhs.sampleBounded(mins, maxs);
        
        // Open CSV file
        std::ofstream csv_file("../output/lhs_samples.csv");
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing" << std::endl;
            return 1;
        }
        
        // Write header
        csv_file << "q0_i,q1_i,q2_i,q3_i,wx_i,wy_i,wz_i,"
                << "q0_f,q1_f,q2_f,q3_f,wx_f,wy_f,wz_f\n";
        
        csv_file << std::fixed << std::setprecision(12);
        
        // Write data
        for (int i = 0; i < iterations; ++i) {
            // Convert Euler angles to quaternions
            DM q_initial = euler2quat(initial_states[i][0], initial_states[i][1], initial_states[i][2]);
            DM q_final = euler2quat(final_states[i][0], final_states[i][1], final_states[i][2]);
            
            csv_file << q_initial(0).scalar() << "," << q_initial(1).scalar() << "," 
                    << q_initial(2).scalar() << "," << q_initial(3).scalar() << ","
                    << initial_states[i][3] << "," << initial_states[i][4] << "," << initial_states[i][5] << ","
                    << q_final(0).scalar() << "," << q_final(1).scalar() << "," 
                    << q_final(2).scalar() << "," << q_final(3).scalar() << ","
                    << final_states[i][3] << "," << final_states[i][4] << "," << final_states[i][5] << "\n";
        }
        
        csv_file.close();
        std::cout << "Successfully generated " << iterations << " LHS samples in ../output/lhs_samples.csv" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}