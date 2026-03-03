# 📚 Lesson 02: Streaming Responses & Analysis

This module upgrades the AI's capabilities by implementing streaming responses. Instead of waiting for the entire response to generate before displaying it, these scripts handle data chunks in real-time, significantly improving the user experience.

### File Breakdown:
* **`01-StreamResponse.py`**: The core implementation for requesting and printing a streamed response from the local LLM.
* **`02-StreamTest.py`**: A testing script to validate the stability of the streaming connection under different prompt conditions.
* **`03-AnalyzeStreamTest.py`**: Processes and analyzes the chunks of data as they arrive from the stream, allowing for real-time evaluation of the model's output.
* **`04-RecommendationTest.py`**: A practical application script utilizing the streaming capabilities to generate dynamic, real-time recommendations based on user input.