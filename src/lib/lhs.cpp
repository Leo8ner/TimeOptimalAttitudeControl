#include <toac/lhs.h>

LHS::LHS(int n_samples, int n_dimensions, unsigned seed)
        : n_samples_(n_samples), n_dims_(n_dimensions), rng_(seed) {}

// Generate LHS samples in [0,1]^n_dims
std::vector<std::vector<double>> LHS::sample() {

    std::vector<std::vector<double>> samples(n_samples_, std::vector<double>(n_dims_));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
    // For each dimension
    for (int d = 0; d < n_dims_; ++d) {
        // Create intervals [0, 1/n, 2/n, ..., 1]
        std::vector<int> intervals(n_samples_);
        for (int i = 0; i < n_samples_; ++i) {
            intervals[i] = i;
        }
        
        // Randomly permute intervals
        std::shuffle(intervals.begin(), intervals.end(), rng_);
        
        // Sample within each interval
        for (int i = 0; i < n_samples_; ++i) {
            double lower = static_cast<double>(intervals[i]) / n_samples_;
            double upper = static_cast<double>(intervals[i] + 1) / n_samples_;
            samples[i][d] = lower + uniform(rng_) * (upper - lower);
        }
    }
    
    return samples;
}
    
// Scale samples to [min, max] bounds
std::vector<std::vector<double>> LHS::sampleBounded(
        const std::vector<double>& mins,
        const std::vector<double>& maxs) {
        
    auto samples = sample();
    
    for (auto& sample : samples) {
        for (int d = 0; d < n_dims_; ++d) {
            sample[d] = mins[d] + sample[d] * (maxs[d] - mins[d]);
        }
    }
    
    return samples;
}