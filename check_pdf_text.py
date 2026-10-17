
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("document loader") / "Deep Learning in Neural Networks.pdf"

reader = PdfReader(str(pdf_path))

# Read the first page
first_page_text = reader.pages[0].extract_text()

print("========== FIRST PAGE TEXT ==========")
print(first_page_text)
