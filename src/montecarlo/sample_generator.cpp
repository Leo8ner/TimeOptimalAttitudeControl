#include <iostream>
#include <fstream>
#include <iomanip>
#include <toac/lhs.h>
#include <helper_functions.h>
#include <casadi/casadi.hpp>
using namespace casadi;

int main(int argc, char** argv) {

    int iterations = 1000;
    // Parse command-line argument for iteration count
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <output_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1000 lhs_samples.csv" << std::endl;
        std::cerr << "Defaulting to 1000 iterations and saving to ../output/lhs_samples.csv." << std::endl;
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
        LHS lhs(iterations, 12);
        double max_angle = 180.0 * DEG; // rad
        double max_vel = 0.0 * DEG;   // rad/s
        
        std::vector<double> mins = {
            -max_angle, -max_angle, -max_angle, -max_vel, -max_vel, -max_vel,  // initial
            -max_angle, -max_angle, -max_angle, -max_vel, -max_vel, -max_vel   // final
        };
        std::vector<double> maxs = {
            max_angle, max_angle, max_angle, max_vel, max_vel, max_vel,  // initial
            max_angle, max_angle, max_angle, max_vel, max_vel, max_vel   // final
        };

        auto samples = lhs.sampleBounded(mins, maxs);
        
        // Open CSV file
        std::string output_file = "../output/" + std::string(argv[2] ? argv[2] : "lhs_samples.csv");
        std::ofstream csv_file(output_file);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing" << std::endl;
            return 1;
        }
        
        // Write header
        csv_file << "q0_i,q1_i,q2_i,q3_i,wx_i,wy_i,wz_i,"
                << "q0_f,q1_f,q2_f,q3_f,wx_f,wy_f,wz_f," 
                << "phi_i,theta_i,psi_i,phi_f,theta_f,psi_f\n";
        
        csv_file << std::fixed << std::setprecision(12);
        
        // Write data
        for (int i = 0; i < iterations; ++i) {
            std::vector<double> initial_state = {samples[i][0], samples[i][1], samples[i][2], 
                                         samples[i][3], samples[i][4], samples[i][5]};
            std::vector<double> final_state = {samples[i][6], samples[i][7], samples[i][8], 
                                       samples[i][9], samples[i][10], samples[i][11]};
            // Convert Euler angles to quaternions
            DM q_initial = euler2quat(initial_state[0], initial_state[1], initial_state[2]);
            DM q_final = euler2quat(final_state[0], final_state[1], final_state[2]);

            csv_file << q_initial(0).scalar() << "," << q_initial(1).scalar() << ","
                    << q_initial(2).scalar() << "," << q_initial(3).scalar() << ","
                    << initial_state[3] << "," << initial_state[4] << "," << initial_state[5] << ","
                    << q_final(0).scalar() << "," << q_final(1).scalar() << "," 
                    << q_final(2).scalar() << "," << q_final(3).scalar() << ","
                    << final_state[3] << "," << final_state[4] << "," << final_state[5] << ","
                    << initial_state[0]*RAD << "," << initial_state[1]*RAD << "," << initial_state[2]*RAD << ","
                    << final_state[0]*RAD << "," << final_state[1]*RAD << "," << final_state[2]*RAD << "\n";
        }
        
        csv_file.close();
        std::cout << "Successfully generated " << iterations << " LHS samples in " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}