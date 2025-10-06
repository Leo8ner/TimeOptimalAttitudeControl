#include <vector>
#include <random>
#include <algorithm>

class LHS {
public:
    LHS(int n_samples, int n_dimensions, unsigned seed = std::random_device{}());
    
    // Generate LHS samples in [0,1]^n_dims
    std::vector<std::vector<double>> sample();
    
    // Scale samples to [min, max] bounds
    std::vector<std::vector<double>> sampleBounded(
        const std::vector<double>& mins,
        const std::vector<double>& maxs);

private:
    int n_samples_;
    int n_dims_;
    std::mt19937 rng_;
};