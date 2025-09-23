#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

/**
 * Simple Latin Hypercube Sampling function
 * 
 * @param n_samples Number of samples to generate
 * @param n_dims Number of dimensions
 * @param bounds Vector of pairs (min, max) for each dimension (optional)
 * @param seed Random seed for reproducibility
 * @return 2D vector containing samples [n_samples][n_dims]
 */
std::vector<std::vector<double>> lhs_sample(int n_samples, int n_dims,
                                           const std::vector<std::pair<double, double>>& bounds = {},
                                           unsigned int seed = std::random_device{}()) {
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // Initialize samples matrix
    std::vector<std::vector<double>> samples(n_samples, std::vector<double>(n_dims, 0.0));
    
    // Check bounds
    bool use_bounds = !bounds.empty();
    if (use_bounds && static_cast<int>(bounds.size()) != n_dims) {
        throw std::invalid_argument("Bounds size must match number of dimensions");
    }
    
    // Generate LHS samples for each dimension
    for (int dim = 0; dim < n_dims; ++dim) {
        // Step 1: Create stratified intervals and sample within each
        std::vector<double> intervals(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double interval_start = static_cast<double>(i) / n_samples;
            double interval_width = 1.0 / n_samples;
            intervals[i] = interval_start + uniform(rng) * interval_width;
        }
        
        // Step 2: Randomly shuffle to break correlations
        std::shuffle(intervals.begin(), intervals.end(), rng);
        
        // Step 3: Assign to samples and apply bounds
        for (int i = 0; i < n_samples; ++i) {
            if (use_bounds) {
                double min_val = bounds[dim].first;
                double max_val = bounds[dim].second;
                samples[i][dim] = min_val + intervals[i] * (max_val - min_val);
            } else {
                samples[i][dim] = intervals[i];
            }
        }
    }
    
    return samples;
}

/**
 * Print samples to console
 */
void print_samples(const std::vector<std::vector<double>>& samples) {
    std::cout << "LHS Samples (" << samples.size() << " samples, " 
              << (samples.empty() ? 0 : samples[0].size()) << " dimensions):\n";
    
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "Sample " << i << ": [";
        for (size_t j = 0; j < samples[i].size(); ++j) {
            std::cout << samples[i][j];
            if (j < samples[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main() {
    try {
        // Example 1: Basic usage - 10 samples in 3D unit cube
        std::cout << "=== Example 1: Basic LHS ===\n";
        auto samples1 = lhs_sample(10, 3, {}, 42);
        print_samples(samples1);
        
        // Example 2: With custom bounds
        std::cout << "\n=== Example 2: LHS with Custom Bounds ===\n";
        std::vector<std::pair<double, double>> bounds = {
            {0, 10},    // x: [0, 10]
            {-5, 5},    // y: [-5, 5]  
            {100, 200}  // z: [100, 200]
        };
        auto samples2 = lhs_sample(8, 3, bounds, 42);
        print_samples(samples2);
        
        // Example 3: Compare coverage with random sampling
        std::cout << "\n=== Example 3: Coverage Comparison ===\n";
        
        // LHS samples
        auto lhs_samples = lhs_sample(20, 2, {}, 42);
        
        // Random samples for comparison
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::vector<std::vector<double>> random_samples(20, std::vector<double>(2));
        for (int i = 0; i < 20; ++i) {
            random_samples[i][0] = uniform(rng);
            random_samples[i][1] = uniform(rng);
        }
        
        // Calculate how well each method covers the space
        auto calculate_coverage = [](const std::vector<std::vector<double>>& samples, int n_bins = 4) {
            std::vector<std::vector<int>> bins(n_bins, std::vector<int>(n_bins, 0));
            for (const auto& sample : samples) {
                int x_bin = std::min(static_cast<int>(sample[0] * n_bins), n_bins - 1);
                int y_bin = std::min(static_cast<int>(sample[1] * n_bins), n_bins - 1);
                bins[x_bin][y_bin]++;
            }
            
            int empty_bins = 0;
            for (const auto& row : bins) {
                for (int count : row) {
                    if (count == 0) empty_bins++;
                }
            }
            return static_cast<double>(empty_bins) / (n_bins * n_bins);
        };
        
        double lhs_empty_ratio = calculate_coverage(lhs_samples);
        double random_empty_ratio = calculate_coverage(random_samples);
        
        std::cout << "Empty bin ratio (4x4 grid, lower is better):\n";
        std::cout << "LHS: " << lhs_empty_ratio << "\n";
        std::cout << "Random: " << random_empty_ratio << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}