# 📚 Lesson 01: Establishing Local Inference

This module focuses on the foundational mechanics of connecting a Python environment to a locally hosted Language Model (via LM Studio) using OpenAI-compatible API endpoints.

The prompt I am using for testing is: How many continents are there?

### File Breakdown:
* **`01-ConnectToLocal.py`**: 
The baseline script to establish a connection to the local server (localhost:1234) and send a standard, synchronous text prompt.
  * Lesson Learned 1: One issue I ran into was the System Prompt, while I originally did not specify a system prompt nor did I setup up a system prompt in LM Studio, a system prompt was still being processed.  By switching from PromptTemplate to ChatPromptTemplate allowed me to specify a system prompt. 
![ConnectToLocal Prompt-A](../../assets/images/Lesson-01/ConnectToLocal-A.png)
![ConnectToLocal Prompt-B](../../assets/images/Lesson-01/ConnectToLocal-B.png)

  * Lesson Learned 2: I do need to put a space in the " ", otherwise the models are inserting default system prompts.

* **`02-InstrumentedTest.py`**: 
A script that takes the methods first developed in **`01-ConnectToLocal.py`** and test it against 6 different Language Models.  This script dynamically loads each model and runs the test Prompt.  I have also added instrumentation (such as timing, logging, or error handling) to measure the performance and reliability of the local model's response.
  * google/gemma-2-9b
  * llama-3.2-3b-instruct
  * microsoft/phi-4-mini-reasoning
  * mistralai/ministral-3-3b
  * openai/gpt-oss-20b
  * qwen2.5-7b-instruct
![InstrumentedTests-A](../../assets/images/Lesson-01/InstrumentedTests.png)
  * Lesson Learned 1:  When testing for simple connection and checking for timing, using a reasoning model should be avoided.