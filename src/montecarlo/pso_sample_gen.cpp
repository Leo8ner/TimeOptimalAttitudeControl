#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <sstream>
#include <toac/lhs.h>
#include <helper_functions.h>

namespace fs = std::filesystem;

static std::string fmt_double(double v, int decimals) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(decimals) << v;
    return ss.str();
}

int main(int argc, char** argv) {
    // Usage: prog <total_samples> <method: sto|full> [n_files=1] [output_file]
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <total_samples> <method: sto|full> [n_files=1] [output_file]\n";
        return 1;
    }

    int total_samples = std::atoi(argv[1]);
    if (total_samples <= 0) {
        std::cerr << "Error: total_samples must be > 0\n";
        return 1;
    }

    std::string method = argv[2];
    std::transform(method.begin(), method.end(), method.begin(), ::tolower);
    if (method != "sto" && method != "full") {
        std::cerr << "Error: method must be 'sto' or 'full'\n";
        return 1;
    }

    int n_files = 1;
    if (argc >= 4) {
        n_files = std::atoi(argv[3]);
        if (n_files <= 0) n_files = 1;
    }

    std::string output_file;
    if (argc == 5) {
        output_file = argv[4];
    } else {
        output_file = (method == "full") ? "lhs_pso_params_full_samples.csv"
                                         : "lhs_pso_params_sto_samples.csv";
        output_file = std::string("../output/pso_params/") + output_file;
    }

    fs::path outpath(output_file);
    fs::path outdir = outpath.parent_path();
    if (outdir.empty()) outdir = ".";
    std::error_code ec;
    fs::create_directories(outdir, ec);

    // group definitions (particles, max_iterations)
    std::vector<std::pair<int,int>> groups;
    if (method == "sto") {
        groups = {
            {640,  750},
            {1280, 650},
            {1920, 550},
            {2560, 450},
            {3200, 350},
            {3840, 300},
            {4480, 200},
            {5120, 150}
        };
    } else {
        groups = {
            {640,  360},
            {1280, 170},
            {1920, 70},
            {2560, 40},
            {3200, 30},
            {3840, 20}
        };
    }
    int group_count = static_cast<int>(groups.size());

    // parameter bounds (exclude particles column)
    double min_iterations_pso = 5.0;
    if (method == "sto") {
        min_iterations_pso = 25.0;
    }
    double min_inertia_weight = 0.1, max_inertia_weight = 10.0;
    double min_cognitive_coeff = 0.1, max_cognitive_coeff = 10.0;
    double min_social_coeff = 0.1, max_social_coeff = 10.0;
    double min_min_inertia = 0.0, max_min_inertia = 1.0;
    double min_min_cognitive = 0.0, max_min_cognitive = 1.0;
    double min_min_social = 0.0, max_min_social = 1.0;
    double min_sigmoid_alpha = 0.1, max_sigmoid_alpha = 10.0;
    double min_sigmoid_saturation = 0.5, max_sigmoid_saturation = 1.0;

    // dims for LHS excluding particles: first element is iterations
    int dims_no_particles = (method == "sto") ? 9 : 7;

    // distribute total_samples across groups (evenly, remainder to first groups)
    std::vector<int> group_totals(group_count, 0);
    int base_per_group = total_samples / group_count;
    int rem_per_group = total_samples % group_count;
    for (int g = 0; g < group_count; ++g) {
        group_totals[g] = base_per_group + (g < rem_per_group ? 1 : 0);
    }

    // create master vector of vectors (each row matches CSV columns)
    // STO rows: [particles, iterations, w, c1, c2, min_w, min_c1, min_c2, alpha, saturation]
    // FULL rows: [particles, iterations, w, c1, c2, min_w, min_c1, min_c2]
    std::vector<std::vector<double>> master_samples;
    master_samples.reserve(total_samples);

    for (int g = 0; g < group_count; ++g) {
        int group_n = group_totals[g];
        if (group_n <= 0) continue;

        int particles_fixed = groups[g].first;
        int iter_max = groups[g].second;

        std::vector<double> mins;
        std::vector<double> maxs;
        mins.push_back(min_iterations_pso); maxs.push_back(static_cast<double>(iter_max)+24.9);
        mins.push_back(min_inertia_weight); maxs.push_back(max_inertia_weight);
        mins.push_back(min_cognitive_coeff); maxs.push_back(max_cognitive_coeff);
        mins.push_back(min_social_coeff);    maxs.push_back(max_social_coeff);
        mins.push_back(min_min_inertia);     maxs.push_back(max_min_inertia);
        mins.push_back(min_min_cognitive);   maxs.push_back(max_min_cognitive);
        mins.push_back(min_min_social);      maxs.push_back(max_min_social);
        if (method == "sto") {
            mins.push_back(min_sigmoid_alpha);      maxs.push_back(max_sigmoid_alpha);
            mins.push_back(min_sigmoid_saturation); maxs.push_back(max_sigmoid_saturation);
        }

        LHS lhs(group_n, dims_no_particles);
        auto samples_mat = lhs.sampleBounded(mins, maxs);

        for (int i = 0; i < group_n; ++i) {
            int iterations_val = static_cast<int>(std::round(samples_mat[i][0]));
            // round to nearest multiple of 50 and clamp to [50, iter_max]
            double iter_round_off = 10;
            if (method == "sto") {
                iter_round_off = 50;
            }
            int iter_rounded = static_cast<int>(std::round(iterations_val / iter_round_off) * iter_round_off);

            std::vector<double> row;
            row.reserve((method == "sto") ? 10 : 8);
            row.push_back(static_cast<double>(particles_fixed));
            row.push_back(static_cast<double>(iter_rounded));
            // push remaining params (w,c1,c2,min_w,min_c1,min_c2,[alpha,saturation])
            for (int d = 1; d < dims_no_particles; ++d) {
                row.push_back(samples_mat[i][d]);
            }
            master_samples.push_back(std::move(row));
        }
    }

    if ((int)master_samples.size() != total_samples) {
        std::cerr << "Warning: generated sample count (" << master_samples.size()
                  << ") != requested total (" << total_samples << "). Proceeding with generated count.\n";
    }

    // split master_samples across n_files sequentially
    int total_generated = static_cast<int>(master_samples.size());
    int base_file = total_generated / n_files;
    int rem_file = total_generated % n_files;

    std::string fname = outpath.filename().string();
    std::string base_name, ext;
    auto pos = fname.find_last_of('.');
    if (pos == std::string::npos) { base_name = fname; ext = ".csv"; }
    else { base_name = fname.substr(0, pos); ext = fname.substr(pos); }

    int cursor = 0;
    for (int file_idx = 1; file_idx <= n_files; ++file_idx) {
        int samples_in_file = base_file + (file_idx <= rem_file ? 1 : 0);
        fs::path outfile = outdir / (base_name + "_" + std::to_string(file_idx) + ext);
        std::ofstream csv(outfile.string());
        if (!csv.is_open()) {
            std::cerr << "Failed to open " << outfile << " for writing\n";
            cursor += samples_in_file;
            continue;
        }

        if (method == "sto") {
            csv << "particles,iterations,w,c1,c2,min_w,min_c1,min_c2,alpha,saturation\n";
        } else {
            csv << "particles,iterations,w,c1,c2,min_w,min_c1,min_c2\n";
        }

        for (int s = 0; s < samples_in_file && cursor < total_generated; ++s, ++cursor) {
            auto &r = master_samples[cursor];
            // r[0]=particles, r[1]=iterations, r[2]=w, r[3]=c1, r[4]=c2, r[5]=min_w, r[6]=min_c1, r[7]=min_c2, [r[8]=alpha, r[9]=saturation]
            // particles: integer
            csv << static_cast<int>(std::round(r[0])) << ",";
            // iterations: integer multiple of 50 (no decimals)
            csv << static_cast<int>(std::round(r[1])) ;

            // w, c1, c2, min_w, min_c1, min_c2, alpha -> 1 decimal
            int start_idx = 2;
            int last_param_idx = static_cast<int>(r.size()) - 1;
            for (int k = start_idx; k <= last_param_idx; ++k) {
                csv << ",";
                if (k == 5 || k == 6 || k == 7) {
                    // min_w, min_c1, min_c2 -> 2 decimals
                    r[k] = r[k] * r[k-3]; // ensure min <= initial by scaling
                }
                // if sto and this is the saturation column (last), use 2 decimals
                if (method == "sto" && k == last_param_idx) {
                    csv << fmt_double(r[k], 2);
                } else {
                    csv << fmt_double(r[k], 1);
                }
            }
            csv << "\n";
        }
        csv.close();
        std::cout << "Wrote " << samples_in_file << " samples to " << outfile.string() << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}