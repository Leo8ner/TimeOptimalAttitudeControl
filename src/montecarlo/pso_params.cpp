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

namespace fs = std::filesystem;

struct PSOResult {
    std::vector<double> params;  // All parameter columns
    double avg_time;
    int n_bad_status;
    int row_number;
};

struct BestResult {
    PSOResult result;
    std::string filename;
    std::string method;  // "FULL" or "STO"
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
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        num_columns = 0;
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
        
        int col = 0;
        while (std::getline(ss, value, ',')) {
            // Remove leading/trailing whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            double val = std::stod(value);
            
            // Last column is n_bad_status
            if (col == num_columns - 1) {
                result.n_bad_status = static_cast<int>(val);
            }
            // Second to last is avg_time (for STO it's column 10, for FULL it's column 10)
            else if (col == num_columns - 4) {  // avg_time is 4 columns from the end
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
        if (result.avg_time < best.avg_time) {
            best = result;
        }
        else if (std::abs(result.avg_time - best.avg_time) < 1e-9) {
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
    // Find min and max for normalization
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
    
    // Avoid division by zero
    double time_range = (max_time - min_time > 1e-9) ? (max_time - min_time) : 1.0;
    double status_range = (max_status - min_status > 0) ? (max_status - min_status) : 1.0;
    
    PSOResult best = results[0];
    double best_score = std::numeric_limits<double>::max();
    
    for (const auto& result : results) {
        // Normalize to [0, 1] range
        double normalized_time = (result.avg_time - min_time) / time_range;
        double normalized_status = static_cast<double>(result.n_bad_status - min_status) / status_range;
        
        // Compute weighted score (lower is better)
        double score = (1.0 - weight) * normalized_status + weight * normalized_time;
        
        if (score < best_score) {
            best_score = score;
            best = result;
        }
    }
    
    return best;
}

// Print result details
void printResult(const std::string& label, const PSOResult& result, 
                 const std::string& method, int num_columns, 
                 bool show_compromise_score = false, double weight = 0.5) {
    std::cout << label << ":\n";
    std::cout << "  Row Number: " << result.row_number << "\n";
    
    // Print parameters
    std::cout << "  Parameters:\n";
    std::cout << "    Particles: " << static_cast<int>(result.params[0]) << "\n";
    std::cout << "    Iterations: " << static_cast<int>(result.params[1]) << "\n";
    std::cout << "    w: " << result.params[2] << "\n";
    std::cout << "    c1: " << result.params[3] << "\n";
    std::cout << "    c2: " << result.params[4] << "\n";
    std::cout << "    min_w: " << result.params[5] << "\n";
    std::cout << "    min_c1: " << result.params[6] << "\n";
    std::cout << "    min_c2: " << result.params[7] << "\n";
    if (method == "STO") {
        std::cout << "    alpha: " << result.params[8] << "\n";
        std::cout << "    saturation: " << result.params[9] << "\n";
    }
    std::cout << "  Average Time: " << std::fixed << std::setprecision(3) 
              << result.avg_time << " s\n";
    std::cout << "  Bad Status Count: " << result.n_bad_status << "\n";
    
    if (show_compromise_score) {
        std::cout << "  Weighted Score: " << std::setprecision(3)
                  << ((1.0 - weight) * result.n_bad_status + weight * result.avg_time * 10.0) << "\n";
    }
    
    std::cout << "\n";
}

// Parse weight parameter from command line
double parseWeight(int argc, char* argv[]) {
    double weight = 0.5;  // Default value
    
    if (argc > 1) {
        try {
            weight = std::stod(argv[1]);
            
            // Validate range [0, 1]
            if (weight < 0.0 || weight > 1.0) {
                std::cerr << "Warning: Weight must be between 0.0 and 1.0. Using default value 0.5\n" << std::endl;
                weight = 0.5;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Invalid weight parameter. Using default value 0.5\n" << std::endl;
            weight = 0.5;
        }
    }
    
    return weight;
}

int main(int argc, char* argv[]) {
    std::string directory = "../output/pso_params/";
    
    // Parse weight parameter
    double weight = parseWeight(argc, argv);
    
    // Check if directory exists
    if (!fs::exists(directory)) {
        std::cerr << "Error: Directory " << directory << " does not exist" << std::endl;
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "PSO PARAMETER TUNING ANALYSIS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Compromise Weight: " << std::fixed << std::setprecision(2) << weight << std::endl;
    std::cout << "  (0.0 = prioritize success rate, 1.0 = prioritize computation time)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Iterate through all CSV files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".csv") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();
            
            int num_columns = 0;
            std::vector<PSOResult> results = readCSV(filepath, num_columns);
            
            if (results.empty()) {
                std::cerr << "Warning: No data found in " << filename << std::endl;
                continue;
            }
            
            // Determine method based on number of columns
            std::string method = (num_columns == 14) ? "STO" : "FULL";
            
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "File: " << filename << std::endl;
            std::cout << "Method: " << method << std::endl;
            std::cout << "Total Results: " << results.size() << std::endl;
            std::cout << "----------------------------------------\n" << std::endl;
            
            // Find best by status
            PSOResult bestByStatus = findBestByStatus(results);
            printResult("Configuration with highest success rate", bestByStatus, method, num_columns);
            
            // Find best by compromise
            PSOResult bestByCompromise = findBestByCompromise(results, weight);
            printResult("Configuration with best compromise (weighted)", bestByCompromise, method, num_columns, true, weight);
            
            // Find best by time
            PSOResult bestByTime = findBestByTime(results);
            printResult("Configuration with lowest average computation time", bestByTime, method, num_columns);
            
            std::cout << std::endl;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "ANALYSIS COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}