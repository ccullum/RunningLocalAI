#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <numeric>

namespace py = pybind11;

// The hyper-optimized math function
float cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b) {
    // Safety check for mismatched or empty vectors
    if (vec_a.size() != vec_b.size() || vec_a.empty()) {
        return 0.0f; 
    }

    // std::inner_product gets auto-vectorized by the MSVC /O2 compiler flag
    float dot_product = std::inner_product(vec_a.begin(), vec_a.end(), vec_b.begin(), 0.0f);
    float norm_a = std::inner_product(vec_a.begin(), vec_a.end(), vec_a.begin(), 0.0f);
    float norm_b = std::inner_product(vec_b.begin(), vec_b.end(), vec_b.begin(), 0.0f);

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }

    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// This macro creates the bridge between C++ and Python
PYBIND11_MODULE(fast_router, m) {
    m.doc() = "High-performance C++ vector math for Semantic Routing";
    
    // Bind the C++ function to Python, automatically casting Python Lists to std::vector
    m.def("cosine_similarity", &cosine_similarity, "Calculates cosine similarity between two float vectors");
}