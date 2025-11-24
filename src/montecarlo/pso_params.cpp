#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <regex>

namespace fs = std::filesystem;

struct PSOResult {
    std::vector<double> params;  // All parameter columns
    double avg_time;
    int n_bad_status;
    int n_runs;
    int row_number;              // row inside the file
    std::string source_file;     // filename that produced this row
};

struct BestResult {
    PSOResult result;
    std::string filename;
    std::string method;  // "full" or "sto"
};

// Function to read CSV file and parse data
std::vector<PSOResult> readCSV(const std::string& filepath, int& num_columns) {
    std::vector<PSOResult> results;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return results;
    }
    
    std::string line;
    
    // Read header to determine number of columns
    num_columns = 0;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            num_columns++;
        }
    }
    
    // Read data rows
    int row_num = 1;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string value;
        PSOResult result;
        result.row_number = row_num++;
        result.source_file = fs::path(filepath).filename().string();
        
        int col = 0;
        while (std::getline(ss, value, ',')) {
            // Remove leading/trailing whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            if (value.empty()) {
                // treat empty as zero
                value = "0";
            }
            
            double val = 0.0;
            try {
                val = std::stod(value);
            } catch (...) {
                val = 0.0;
            }
            
            // Last column is n_runs
            if (col == num_columns - 1) {
                result.n_runs = static_cast<int>(std::round(val));
            } else if (col == num_columns - 2) {
                result.n_bad_status = static_cast<int>(std::round(val));
            }
            // avg_time is 5 columns from the end
            else if (col == num_columns - 5) {
                result.avg_time = val;
            }
            else {
                result.params.push_back(val);
            }
            
            col++;
        }
        
        results.push_back(result);
    }
    
    file.close();
    return results;
}

// Find result with minimum n_bad_status (breaking ties with avg_time)
PSOResult findBestByStatus(const std::vector<PSOResult>& results) {
    if (results.empty()) {
        throw std::runtime_error("No results to analyze");
    }
    
    PSOResult best = results[0];
    
    for (const auto& result : results) {
        if (result.n_runs != 1000) continue; // skip incomplete runs
        if (result.n_bad_status < best.n_bad_status) {
            best = result;
        }
        else if (result.n_bad_status == best.n_bad_status) {
            if (result.avg_time < best.avg_time) {
                best = result;
            }
        }
    }
    
    return best;
}

// Find result with minimum avg_time (breaking ties with n_bad_status)
PSOResult findBestByTime(const std::vector<PSOResult>& results) {
    if (results.empty()) {
        throw std::runtime_error("No results to analyze");
    }
    
    PSOResult best = results[0];
    
    for (const auto& result : results) {
        if (result.n_runs != 1000) continue; // skip incomplete runs
        if (result.avg_time < best.avg_time) {
            best = result;
        }
        else if (std::abs(result.avg_time - best.avg_time) < 1e-12) {
            if (result.n_bad_status < best.n_bad_status) {
                best = result;
            }
        }
    }
    
    return best;
}

// Find result with best compromise score
// weight: 0.0 = prioritize status only, 1.0 = prioritize time only
PSOResult findBestByCompromise(const std::vector<PSOResult>& results, double weight) {
    if (results.empty()) {
        throw std::runtime_error("No results to analyze");
    }
    
    // Normalize values to compute weighted score
    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::lowest();
    int min_status = std::numeric_limits<int>::max();
    int max_status = std::numeric_limits<int>::lowest();
    
    for (const auto& result : results) {
        min_time = std::min(min_time, result.avg_time);
        max_time = std::max(max_time, result.avg_time);
        min_status = std::min(min_status, result.n_bad_status);
        max_status = std::max(max_status, result.n_bad_status);
    }
    
    double time_range = (max_time - min_time > 1e-9) ? (max_time - min_time) : 1.0;
    double status_range = (max_status - min_status > 0) ? (max_status - min_status) : 1.0;
    
    PSOResult best = results[0];
    double best_score = std::numeric_limits<double>::max();
    
    for (const auto& result : results) {
        double normalized_time = (result.avg_time - min_time) / time_range;
        double normalized_status = static_cast<double>(result.n_bad_status - min_status) / status_range;
        
        double score = (1.0 - weight) * normalized_status + weight * normalized_time;
        
        if (score < best_score) {
            best_score = score;
            best = result;
        }
    }
    
    return best;
}

// Try to load parameter line from corresponding lhs file.
// result_filename is expected like "pso_sto_tuning_3.csv" and will map to
// "lhs_pso_params_samples_3.csv" inside lhs_dir.
bool loadLHSParams(const std::string &lhs_dir, const std::string &result_filename,
                   int row_number, std::vector<double> &out_params, const std::string& method) {
    std::regex idx_re(R"(_(\d+)\.csv$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(result_filename, m, idx_re)) {
        return false; // no index found
    }
    std::string idx = m[1];
    fs::path lhs_path = fs::path(lhs_dir) / ("lhs_pso_params_" + method + "_samples_" + idx + ".csv");
    if (!fs::exists(lhs_path)) return false;

    std::ifstream f(lhs_path.string());
    if (!f.is_open()) return false;

    std::string line;
    // skip header
    if (!std::getline(f, line)) return false;

    int cur = 0;
    while (std::getline(f, line)) {
        ++cur;
        if (cur == row_number) {
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // trim
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                try { out_params.push_back(std::stod(token)); }
                catch (...) { out_params.push_back(0.0); }
            }
            return true;
        }
    }
    return false;
}

// Print result details
// added lhs_dir parameter (default to pso_params output folder)
void printResult(const std::string& label, const PSOResult& result, 
                 const std::string& method, int num_columns, 
                 bool show_compromise_score = false, double weight = 0.5,
                 const std::string& lhs_dir = "../output/pso_params/") {
    std::cout << label << ":\n";
    std::cout << "  Source File: " << result.source_file << "\n";
    std::cout << "  Row Number: " << result.row_number << "\n";

    // Try to load corresponding lhs params
    std::vector<double> lhs_params;
    bool lhs_ok = loadLHSParams(lhs_dir, result.source_file, result.row_number, lhs_params, method);

    std::cout << "  Parameters:\n";
    if (lhs_ok && !lhs_params.empty()) {
        // Expect lhs_params layout: particles, iterations, w, c1, c2, min_w, min_c1, min_c2 [, alpha, saturation]
        if (lhs_params.size() >= 8) {
            std::cout << "    Particles: " << static_cast<int>(lhs_params[0]) << "\n";
            std::cout << "    Iterations: " << static_cast<int>(lhs_params[1]) << "\n";
            std::cout << "    w: " << lhs_params[2] << "\n";
            std::cout << "    c1: " << lhs_params[3] << "\n";
            std::cout << "    c2: " << lhs_params[4] << "\n";
            std::cout << "    min_w: " << lhs_params[5] << "\n";
            std::cout << "    min_c1: " << lhs_params[6] << "\n";
            std::cout << "    min_c2: " << lhs_params[7] << "\n";
        } else {
            for (size_t i = 0; i < lhs_params.size(); ++i) {
                std::cout << "    p" << i << ": " << lhs_params[i] << "\n";
            }
        }
        if (method == "sto" && lhs_params.size() >= 10) {
            std::cout << "    alpha: " << lhs_params[8] << "\n";
            std::cout << "    saturation: " << lhs_params[9] << "\n";
        }
    } else {
        // fallback to result.params already present in result file
        if (!result.params.empty()) {
            if (result.params.size() >= 8) {
                std::cout << "    Particles: " << static_cast<int>(result.params[0]) << "\n";
                std::cout << "    Iterations: " << static_cast<int>(result.params[1]) << "\n";
                std::cout << "    w: " << result.params[2] << "\n";
                std::cout << "    c1: " << result.params[3] << "\n";
                std::cout << "    c2: " << result.params[4] << "\n";
                std::cout << "    min_w: " << result.params[5] << "\n";
                std::cout << "    min_c1: " << result.params[6] << "\n";
                std::cout << "    min_c2: " << result.params[7] << "\n";
            } else {
                for (size_t i = 0; i < result.params.size(); ++i) {
                    std::cout << "    p" << i << ": " << result.params[i] << "\n";
                }
            }
            if (method == "sto" && result.params.size() >= 10) {
                std::cout << "    alpha: " << result.params[8] << "\n";
                std::cout << "    saturation: " << result.params[9] << "\n";
            }
        } else {
            std::cout << "    (No parameter info available)\n";
        }
    }

    std::cout << "  Average Time: " << std::fixed << std::setprecision(3) 
              << result.avg_time << " s\n";
    std::cout << "  Bad Status Count: " << result.n_bad_status << "\n";
    
    if (show_compromise_score) {
        double score = (1.0 - weight) * result.n_bad_status + weight * result.avg_time * 10.0;
        std::cout << "  Weighted Score (status*(1-w) + 10*time*w): " << std::setprecision(3)
                  << score << "\n";
    }
    
    std::cout << "\n";
}

// Parse weight parameter from command line (first argument)
double parseWeight(int argc, char* argv[]) {
    double weight = 0.5;  // Default value
    
    if (argc > 1) {
        try {
            weight = std::stod(argv[1]);
            if (weight < 0.0 || weight > 1.0) {
                std::cerr << "Warning: Weight must be between 0.0 and 1.0. Using default value 0.5\n";
                weight = 0.5;
            }
        }
        catch (...) {
            std::cerr << "Warning: Invalid weight parameter. Using default value 0.5\n";
            weight = 0.5;
        }
    }
    
    return weight;
}

int main(int argc, char* argv[]) {
    std::string directory = "../output/pso_params/";
    
    double weight = parseWeight(argc, argv);
    
    if (!fs::exists(directory)) {
        std::cerr << "Error: Directory " << directory << " does not exist" << std::endl;
        return 1;
    }
    
    std::cout << "========================================\n";
    std::cout << "PSO PARAMETER TUNING AGGREGATED ANALYSIS\n";
    std::cout << "========================================\n";
    std::cout << "Compromise Weight: " << std::fixed << std::setprecision(2) << weight << "\n";
    std::cout << "  (0.0 = prioritize success rate, 1.0 = prioritize computation time)\n";
    std::cout << "========================================\n\n";
    
    // regex patterns to match files like pso_sto_tuning.csv or pso_sto_tuning_1.csv
    std::regex sto_re(R"(pso_sto_tuning(_\d+)?\.csv)", std::regex::icase);
    std::regex full_re(R"(pso_full_tuning(_\d+)?\.csv)", std::regex::icase);
    
    std::vector<PSOResult> all_sto_results;
    std::vector<PSOResult> all_full_results;
    int sto_num_columns = 0;
    int full_num_columns = 0;
    int sto_file_count = 1;
    int full_file_count = 1;
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() != ".csv") continue;
        std::string filename = entry.path().filename().string();
        
        if (std::regex_match(filename, sto_re)) {
            int num_cols = 0;
            auto vec = readCSV(entry.path().string(), num_cols);
            if (!vec.empty()) {
                sto_file_count++;
                if (sto_num_columns == 0) sto_num_columns = num_cols;
                // append and keep source_file info set inside readCSV
                all_sto_results.insert(all_sto_results.end(), vec.begin(), vec.end());
            }
        } else if (std::regex_match(filename, full_re)) {
            int num_cols = 0;
            auto vec = readCSV(entry.path().string(), num_cols);
            if (!vec.empty()) {
                full_file_count++;
                if (full_num_columns == 0) full_num_columns = num_cols;
                all_full_results.insert(all_full_results.end(), vec.begin(), vec.end());
            }
        }
    }
    
    // Process sto group if any
    if (!all_sto_results.empty()) {
        std::cout << "----------------------------------------\n";
        std::cout << "AGGREGATED sto RESULTS (" << sto_file_count - 1 << " files, "
                  << all_sto_results.size() << " total rows)\n";
        std::cout << "----------------------------------------\n\n";
        try {
            PSOResult bestByStatus = findBestByStatus(all_sto_results);
            printResult("Best (by n_bad_status) - sto aggregate", bestByStatus, "sto", sto_num_columns);
            
            PSOResult bestByCompromise = findBestByCompromise(all_sto_results, weight);
            printResult("Best (by compromise) - sto aggregate", bestByCompromise, "sto", sto_num_columns, true, weight);
            
            PSOResult bestByTime = findBestByTime(all_sto_results);
            printResult("Best (by avg_time) - sto aggregate", bestByTime, "sto", sto_num_columns);
        } catch (const std::exception& e) {
            std::cerr << "Error processing sto aggregated results: " << e.what() << "\n";
        }
    } else {
        std::cout << "No sto tuning files found.\n";
    }
    
    // Process full group if any
    if (!all_full_results.empty()) {
        std::cout << "----------------------------------------\n";
        std::cout << "AGGREGATED full RESULTS (" << full_file_count - 1 << " files, "
                  << all_full_results.size() << " total rows)\n";
        std::cout << "----------------------------------------\n\n";
        try {
            PSOResult bestByStatus = findBestByStatus(all_full_results);
            printResult("Best (by n_bad_status) - full aggregate", bestByStatus, "full", full_num_columns);
            
            PSOResult bestByCompromise = findBestByCompromise(all_full_results, weight);
            printResult("Best (by compromise) - full aggregate", bestByCompromise, "full", full_num_columns, true, weight);
            
            PSOResult bestByTime = findBestByTime(all_full_results);
            printResult("Best (by avg_time) - full aggregate", bestByTime, "full", full_num_columns);
        } catch (const std::exception& e) {
            std::cerr << "Error processing full aggregated results: " << e.what() << "\n";
        }
    } else {
        std::cout << "No full tuning files found.\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "ANALYSIS COMPLETE\n";
    std::cout << "========================================\n";
    return 0;
}