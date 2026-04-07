# Lesson 12: Custom C++ Inference (The Over-Engineering Experiment)

In this lesson, we explored "Bare-Metal" optimization by moving our Semantic Router from a network-based API call to a custom-compiled C++ engine using `llama.cpp` and `pybind11`.

## 🧪 The Experiment
The goal was to eliminate the 2-second network latency of the embedding process by performing inference directly in RAM via a dedicated C++ wrapper.

## 📊 The "Reality Check" Result
| Metric | API-Based (Legacy) | Bare-Metal (C++) | Improvement |
| :--- | :--- | :--- | :--- |
| **Routing Latency** | 2.07s | **0.018s** | **115x Faster** |
| **Total User Wait Time** | 16.84s | 15.94s | **~5% Faster** |

## 💡 Key Lesson: The 99% Rule
While we successfully achieved a **115x speedup** in the routing component, the user's total wait time only improved by **5%**. This is because the true bottleneck remains the LLM's "Time to First Token" (TTFT), which averages **13.8 seconds**.

## 🛑 Final Decision: REJECTED
This implementation was **reverted** and moved to the `experiment/` directory for the following reasons:
1. **Low ROI:** A 1-second gain doesn't justify the architectural complexity.
2. **Platform Dependency:** It required a C++ build chain (MSVC/CMake), breaking our "Platform Agnostic" project rule.

---

### 📂 Folder Contents
* **[Performance_PostMortem.md](./Performance_PostMortem.md)**: Deep-dive analysis and full benchmark data.
* **[experiment/](./experiment/)**: Contains the rejected C++ source code, build files, and benchmark scripts.