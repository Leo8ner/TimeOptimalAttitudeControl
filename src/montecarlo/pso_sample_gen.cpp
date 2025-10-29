#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <toac/lhs.h>
#include <helper_functions.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    int iterations = 1000;
    int n_files = 1;
    std::string output_file; // base name (may include path)

    // Parse arguments:
    // usage: prog [iterations] [n_files] [output_file]
    if (argc > 1) iterations = std::atoi(argv[1]);
    if (iterations <= 0) iterations = 1000;

    if (argc > 2) {
        n_files = std::atoi(argv[2]);
        if (n_files <= 0) n_files = 1;
    }

    if (argc == 4) {
        output_file = std::string(argv[3]);
    } else {
        // default file name based on method
        output_file = "lhs_pso_params_samples.csv";
        output_file = std::string("../output/pso_params/") + output_file;
        std::cout << "Correct usage: " << argv[0] << " [iterations] [n_files] [output_file]\n";
        std::cout << "Using default values for missing arguments and ignoring extra ones.\n";
    }


    std::cout << "Total samples: " << iterations << "\n";
    std::cout << "Number of files: " << n_files << "\n";
    std::cout << "Base output: " << output_file << "\n";

    // ensure output directory exists if not provided by user (i.e., relative path)
    fs::path outpath(output_file);
    fs::path outdir = outpath.parent_path();
    if (outdir.empty()) outdir = ".";
    std::error_code ec;
    fs::create_directories(outdir, ec);

    // common bounds
    double min_particles = 1.0, max_particles = 10.0;
    double min_iterations = 50.0, max_iterations = 500.0;
    double min_inertia_weight = 0.1, max_inertia_weight = 10.0;
    double min_cognitive_coeff = 0.1, max_cognitive_coeff = 10.0;
    double min_social_coeff = 0.1, max_social_coeff = 10.0;
    double min_min_inertia = 0.0, max_min_inertia = 1.0;
    double min_min_cognitive = 0.0, max_min_cognitive = 1.0;
    double min_min_social = 0.0, max_min_social = 1.0;

    double min_sigmoid_alpha = 0.1, max_sigmoid_alpha = 10.0;
    double min_sigmoid_saturation = 0.5, max_sigmoid_saturation = 1.0;

    // divide iterations across files
    int total = iterations;
    int files = n_files;
    int base = total / files;
    int rem = total % files;

    // split output_file into base and ext
    std::string fname = outpath.filename().string();
    std::string prefix = outpath.parent_path().string();
    std::string base_name, ext;
    auto pos = fname.find_last_of('.');
    if (pos == std::string::npos) { base_name = fname; ext = ".csv"; }
    else { base_name = fname.substr(0, pos); ext = fname.substr(pos); }

    int dimensions = 10;
    LHS lhs(iterations, dimensions);

    std::vector<double> mins = {
        min_particles, min_iterations, min_inertia_weight, min_cognitive_coeff, min_social_coeff,
        min_min_inertia, min_min_cognitive, min_min_social, min_sigmoid_alpha, min_sigmoid_saturation
    };
    std::vector<double> maxs = {
        max_particles, max_iterations, max_inertia_weight, max_cognitive_coeff, max_social_coeff,
        max_min_inertia, max_min_cognitive, max_min_social, max_sigmoid_alpha, max_sigmoid_saturation
    };

    auto samples_mat = lhs.sampleBounded(mins, maxs);
    int base_idx = 0;
    for (int idx = 1; idx <= files; ++idx) {
        int samples = base + (idx <= rem ? 1 : 0);
        if (samples <= 0) {
            std::cout << "Skipping file " << idx << " (0 samples)\n";
            continue;
        }

        // build output filename: <base_name>_<idx><ext>
        fs::path outfile = outdir / (base_name + "_" + std::to_string(idx) + ext);
        std::ofstream csv(outfile.string());
        if (!csv.is_open()) {
            std::cerr << "Failed to open " << outfile << " for writing\n";
            continue;
        }

        // header
        csv << "particles,iterations,w,c1,c2,min_w,min_c1,min_c2,alpha,saturation\n";
        csv << std::fixed << std::setprecision(2);

        for (int i = base_idx; i < base_idx + samples; ++i) {
            int n_particles = static_cast<int>(std::round(samples_mat[i][0]) * 640);
            int n_iterations = static_cast<int>(std::round(samples_mat[i][1] / 50.0) * 50);
            double inertia_weight = std::round(samples_mat[i][2] * 10.0) / 10.0;
            double cognitive_coeff = std::round(samples_mat[i][3] * 10.0) / 10.0;
            double social_coeff = std::round(samples_mat[i][4] * 10.0) / 10.0;
            double min_inertia = std::round(samples_mat[i][5] * samples_mat[i][2] * 10.0) / 10.0;
            double min_cognitive = std::round(samples_mat[i][6] * samples_mat[i][3] * 10.0) / 10.0;
            double min_social = std::round(samples_mat[i][7] * samples_mat[i][4] * 10.0) / 10.0;
            double sigmoid_alpha = std::round(samples_mat[i][8] * 10.0) / 10.0;
            double sigmoid_saturation = std::round(samples_mat[i][9] * 100.0) / 100.0;
            csv << n_particles << ","
                << n_iterations << ","
                << inertia_weight << ","
                << cognitive_coeff << ","
                << social_coeff << ","
                << min_inertia << ","
                << min_cognitive << ","
                << min_social << ","
                << sigmoid_alpha << ","
                << sigmoid_saturation << "\n";
        }
        csv.close();
        std::cout << "Wrote " << samples << " samples to " << outfile.string() << "\n";
        base_idx += samples;
    }

    std::cout << "Done.\n";
    return 0;
}