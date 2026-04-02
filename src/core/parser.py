import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from .colors import Colors
from .config import Config
from utils.metrics import perf_tracker # <-- Import the global perf_tracker logger

pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD_PATH

class DocumentParser:
    def __init__(self):
        print(f"{Colors.SYSTEM}[Parser] Initializing Multimodal Document Parser...{Colors.RESET}")

    def process_file(self, file_bytes: bytes, filename: str) -> str:
        """Routes the file to the correct parser based on extension."""
        ext = filename.lower().split('.')[-1]
        
        if ext == 'txt':
            return file_bytes.decode('utf-8')
        elif ext in ['png', 'jpg', 'jpeg']:
            return self._parse_image(file_bytes)
        elif ext == 'pdf':
            return self._parse_pdf_with_ocr(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @perf_tracker.measure("Image OCR Time") # <-- Track standalone image OCR
    def _parse_image(self, file_bytes: bytes) -> str:
        """Runs standard OCR on a standalone image file."""
        print(f"{Colors.SYSTEM}[Parser] Running OCR on Image...{Colors.RESET}")
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()

    @perf_tracker.measure("PDF Extraction & OCR Time") # <-- Track the heaviest CPU task
    def _parse_pdf_with_ocr(self, file_bytes: bytes) -> str:
        """
        Layout-Aware Parsing: Reads text from a PDF page, then hunts for 
        embedded images on that same page, OCRs them, and appends the data.
        """
        print(f"{Colors.SYSTEM}[Parser] Parsing PDF and scanning for embedded images...{Colors.RESET}")
        full_text = []
        image_count = 0

        # Open the PDF from bytes
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_content = [f"\n--- PAGE {page_num + 1} ---"]
            
            # 1. Extract the native text
            native_text = page.get_text()
            if native_text:
                page_content.append(native_text)
                
            # 2. Hunt for embedded images (Tables, Charts, Scans)
            image_list = page.get_images(full=True)
            if image_list:
                print(f"{Colors.SYSTEM}[Parser] Found {len(image_list)} image(s) on page {page_num + 1}. Running OCR...{Colors.RESET}")
                for img_index, img in enumerate(image_list):
                    image_count += 1
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    
                    # --- THE NON-DESTRUCTIVE FIX ---
                    if img_obj.mode == 'CMYK':
                        img_obj = img_obj.convert('RGB')
                    
                    # --- ARTIFICIAL SCALING FOR OCR ---
                    width, height = img_obj.size
                    img_obj = img_obj.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
                    
                    ocr_text = pytesseract.image_to_string(img_obj).strip()
                    
                    if ocr_text:
                        page_content.append(f"\n[EMBEDDED IMAGE/TABLE DATA]:\n{ocr_text}\n")
            
            full_text.append("\n".join(page_content))
            
        pdf_document.close()
        perf_tracker.record_value("Total Images Processed", image_count)
        return "\n".join(full_text)
    
    @perf_tracker.measure("Total Document Ingestion Time") # <-- Track the full pipeline
    def extract_and_chunk(self, file_bytes: bytes, filename: str) -> list:
        """The master method: Extracts text and slices it into overlapping chunks."""
        # 1. Extract the raw text
        raw_content = self.process_file(file_bytes, filename)
        
        # 2. The Chunking Math
        chunk_size = Config.CHUNK_SIZE
        overlap = Config.CHUNK_OVERLAP 
        
        chunks = []
        start = 0
        while start < len(raw_content):
            end = start + chunk_size
            chunks.append(raw_content[start:end])
            start += (chunk_size - overlap)
            
        return chunks