#include <iostream>
#include <fstream>
#include <iomanip>
#include <toac/lhs.h>
#include <helper_functions.h>
#include <casadi/casadi.hpp>
using namespace casadi;

int main(int argc, char** argv) {

    int iterations = 1000;
    bool include_w = false; // whether angular velocities are included
    std::string output_file = "lhs_samples.csv";

    // Usage:
    // ./prog <iterations> <include_w> <output_file>    (include_w: 1/0,true/false)
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <iterations> <include_w> <output_file>\n";
        std::cerr << "  include_w: 1/0 or true/false (default=0)\n";
        std::cerr << "Example: " << argv[0] << " 1000 0 lhs_samples.csv\n";
        return 1;
    } else {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0) {
            std::cerr << "Error: iterations must be a positive integer.\n";
            return 1;
        }
    }

    if (argc >= 3) {
        std::string incl = argv[2];
        std::transform(incl.begin(), incl.end(), incl.begin(), ::tolower);
        if (incl == "0" || incl == "false" || incl == "no") include_w = false;
        else if (incl == "1" || incl == "true" || incl == "yes") include_w = true;
        else {
            std::cerr << "Error: unknown include_w value '" << argv[2] << "'. include_w: 1/0 or true/false (default=0)\n";
            return 1;
        }
    } 
    
    if (argc == 4) {
        output_file = std::string(argv[3]);
    } 

    // prepend output directory if relative filename (keep behavior from before)
    if (output_file.find('/') == std::string::npos && output_file.find('\\') == std::string::npos) {
        output_file = std::string("../output/") + output_file;
    }

    std::cout << "Generating " << iterations << " LHS samples..." << std::endl;
    std::cout << "Include angular velocities: " << (include_w ? "yes" : "no") << std::endl;
    std::cout << "Saving to: " << output_file << std::endl;

    try {
        int dims = include_w ? 12 : 6;
        LHS lhs(iterations, dims);

        double max_ang = max_angle; // rad
        double max_vel = max_vel;     // rad/s

        std::vector<double> mins = {
                -max_ang, -max_ang, -max_ang,
                -max_ang, -max_ang, -max_ang
            };
        std::vector<double> maxs = {
                max_ang,  max_ang,  max_ang,
                max_ang,  max_ang,  max_ang
            };

        if (include_w) {
            for (int i = 0; i < 6; ++i) {
                mins.push_back(-max_vel);
                maxs.push_back(max_vel);
            }
        }

        auto samples = lhs.sampleBounded(mins, maxs);

        // Open CSV file
        std::ofstream csv_file(output_file);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing: " << output_file << std::endl;
            return 1;
        }

        // Write header
        csv_file << "q0_i,q1_i,q2_i,q3_i,wx_i,wy_i,wz_i,"
                    "q0_f,q1_f,q2_f,q3_f,wx_f,wy_f,wz_f,"
                    "phi_i,theta_i,psi_i,phi_f,theta_f,psi_f\n";


        csv_file << std::fixed << std::setprecision(6);

        // Write data
        for (int i = 0; i < iterations; ++i) {
            std::vector<double> initial_state = {samples[i][0], samples[i][1], samples[i][2],
                                 0, 0, 0};
            std::vector<double> final_state   = {samples[i][3], samples[i][4], samples[i][5],
                                 0, 0, 0};
            if (include_w) {
                for (int j = 0; j < 3; ++j) {
                    initial_state[j+3] = samples[i][j+6];
                    final_state[j+3] = samples[i][j+9];
                }
            }
            // Convert Euler angles to quaternions
            DM q_initial = euler2quat(initial_state[0], initial_state[1], initial_state[2]);
            DM q_final   = euler2quat(final_state[0], final_state[1], final_state[2]);
            if (dot(q_initial, q_final).scalar() < 0) {
                q_final = -q_final;
            }

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