#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "llama.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <numeric>

namespace py = pybind11;

class EmbeddingEngine {
private:
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;

public:
    EmbeddingEngine(const std::string& model_path) {
        llama_backend_init();
        
        llama_model_params model_params = llama_model_default_params();
        model = llama_load_model_from_file(model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("Failed to load model from: " + model_path);
        }

        vocab = llama_model_get_vocab(model);
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.embeddings = true; 
        
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }
    }

    ~EmbeddingEngine() {
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
    }

    std::vector<float> generate_embedding(const std::string& text) {
        std::vector<llama_token> tokens(text.length() + 1); 
        int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
        
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
        }
        tokens.resize(n_tokens);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

        if (llama_decode(ctx, batch) != 0) {
            throw std::runtime_error("Llama decode failed");
        }

        int n_embd = llama_model_n_embd(model);
        float* embd_ptr = llama_get_embeddings_seq(ctx, 0);
        
        if (!embd_ptr) {
            throw std::runtime_error("Failed to extract embeddings");
        }

        std::vector<float> embedding(embd_ptr, embd_ptr + n_embd);
        
        // Corrected memory architecture cleanup using llama_memory_t
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, -1, -1, -1);

        return embedding;
    }

    // Bare-Metal Similarity Function
    float similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vector dimensions must match.");
        }

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot += v1[i] * v2[i];
            norm_a += v1[i] * v1[i];
            norm_b += v2[i] * v2[i];
        }

        if (norm_a == 0 || norm_b == 0) return 0.0f;
        return (float)(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }
};

PYBIND11_MODULE(llama_engine, m) {
    py::class_<EmbeddingEngine>(m, "EmbeddingEngine")
        .def(py::init<const std::string&>())
        .def("generate_embedding", &EmbeddingEngine::generate_embedding)
        .def("similarity", &EmbeddingEngine::similarity);
}