
from pathlib import Path
from pypdf import PdfReader

PDF_FOLDER = Path("document loader")

total_pages = 0
pdf_count = 0

for pdf_path in PDF_FOLDER.rglob("*.pdf"):
    try:
        reader = PdfReader(str(pdf_path))
        pages = len(reader.pages)

        print(f"{pdf_path.name}: {pages} pages")
        total_pages += pages
        pdf_count += 1

    except Exception as error:
        print(f"Could not read {pdf_path.name}: {error}")

print("\n========== CORPUS SUMMARY ==========")
print(f"PDF files: {pdf_count}")
print(f"Total pages: {total_pages}")

if total_pages >= 50:
    print("PASS: Corpus meets the 50-page requirement.")
else:
    print(f"NOT YET: Add at least {50 - total_pages} more pages.")
