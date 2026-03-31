import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from .colors import Colors

# Load environment variables from the .env file
load_dotenv()

# Safely fetch the Tesseract path (Returns None if not found)
tesseract_path = os.getenv("TESSERACT_CMD_PATH")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

class DocumentIngestor:
    def __init__(self):
        print(f"{Colors.SYSTEM}[Ingestor] Initializing Multimodal Document Parser...{Colors.RESET}")
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

    def _parse_image(self, file_bytes: bytes) -> str:
        """Runs standard OCR on a standalone image file."""
        print(f"{Colors.SYSTEM}[Ingestor] Running OCR on Image...{Colors.RESET}")
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()

    def _parse_pdf_with_ocr(self, file_bytes: bytes) -> str:
        """
        Layout-Aware Parsing: Reads text from a PDF page, then hunts for 
        embedded images on that same page, OCRs them, and appends the data.
        """
        print(f"{Colors.SYSTEM}[Ingestor] Parsing PDF and scanning for embedded images...{Colors.RESET}")
        full_text = []
        
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
                print(f"{Colors.SYSTEM}[Ingestor] Found {len(image_list)} image(s) on page {page_num + 1}. Running OCR...{Colors.RESET}")
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    
                    # --- THE NON-DESTRUCTIVE FIX ---
                    # If the PDF image is formatted for print (CMYK), gracefully 
                    # convert it to standard digital color (RGB) so PNG doesn't crash.
                    if img_obj.mode == 'CMYK':
                        img_obj = img_obj.convert('RGB')
                    
                    # --- ARTIFICIAL SCALING FOR OCR ---
                    width, height = img_obj.size
                    img_obj = img_obj.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
                    
                    ocr_text = pytesseract.image_to_string(img_obj).strip()
                    
                    if ocr_text:
                        # Inject the OCR text directly into the page stream!
                        page_content.append(f"\n[EMBEDDED IMAGE/TABLE DATA]:\n{ocr_text}\n")
            
            full_text.append("\n".join(page_content))
            
        pdf_document.close()
        return "\n".join(full_text)