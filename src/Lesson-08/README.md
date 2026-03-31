# Lesson 08: Multimodal Document Parsing & OCR Pipeline

### Overview
This module introduces a fully localized, multimodal Retrieval-Augmented Generation (RAG) ingestion pipeline. JARVIS is now capable of digesting complex, multi-page PDFs, extracting embedded images, executing high-fidelity Optical Character Recognition (OCR), and embedding the resulting text into the Qdrant vector database for long-term semantic retrieval.

### Core Architecture & Technologies
* **PyMuPDF (`fitz`):** Handles high-speed PDF parsing and binary image extraction.
* **Tesseract v5.x:** Deep-learning-based OCR engine configured with the Math/Equation detection module (`eng+equ`) to read charts, tables, and formulas.
* **Pillow (`PIL`):** Intercepts raw image bytes, manages CMYK-to-RGB color space conversions to prevent system crashes, and artificially upscales low-DPI embedded images by 200% (using LANCZOS resampling) to ensure OCR accuracy.
* **Nomic (`nomic-embed-text-v1.5`):** Mathematically embeds the extracted text chunks into 768-dimensional space.
* **python-dotenv:** Secures local system paths and environment variables.

### Key Features Implemented
1. **Streamlit Drop-Zone:** A new sidebar widget allows for seamless drag-and-drop ingestion of `.txt`, `.pdf`, `.png`, and `.jpg` files directly into the JARVIS memory bank.
2. **The Chunking Engine:** Extracted text is algorithmically sliced into 1,000-character blocks with a 200-character overlap to preserve sentence context, then appended with a custom FWA (Frequency, Weight, Age) payload.
3. **Non-Destructive Image Processing:** Embedded PDF images formatted for print (CMYK) are gracefully converted to digital formats (RGB) without destroying color fidelity, future-proofing the pipeline for Vision-Language Models.
4. **Heuristic Routing Override:** Hardcoded triggers intercept document-based commands (e.g., "search", "document", "pdf") to bypass stubborn local LLM intent routing and force a vector database query.

### Setup & Installation
To run this ingestion pipeline, specific system-level dependencies must be installed:

**1. Tesseract OCR Engine (Windows)**
* Download and install the 64-bit Windows executable from UB-Mannheim.
* During installation, ensure the **Math / equation detection module** is checked.
* Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**2. Python Dependencies**
It is highly recommended to install dependencies using the provided `requirements.txt` file rather than installing packages individually. This ensures version consistency across environments.
```bash
pip install -r requirements.txt
```

**3. Environment Variables (`.env`)**
Do not hardcode the Tesseract executable path in the Python scripts. Create a `.env` file in the root directory and add the following:
```env
TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```
*Note: Ensure `.env` is added to your `.gitignore`.*

### Known Limitations (The Catalyst for Lesson 09)
While the ingestion pipeline is robust, relying on local, small-parameter LLMs (like 8B models) for zero-shot intent routing is highly brittle. The LLM often misclassifies semantic document queries as casual conversation, bypassing the vector database entirely. Currently, a temporary heuristic override (trigger words) is used to force retrieval.