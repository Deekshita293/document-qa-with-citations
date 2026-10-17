
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = Path("document loader")

# Load the PDF while preserving its page number
documents = []

for pdf_path in PDF_FOLDER.glob("*.pdf"):
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_number
            })

if not documents:
    raise SystemExit("No readable PDF text found. Check the document loader folder.")

# Strategy 1: Recursive character chunking
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

recursive_chunks = []

for document in documents:
    for chunk in recursive_splitter.split_text(document["text"]):
        recursive_chunks.append({
            "text": chunk,
            "source": document["source"],
            "page": document["page"]
        })

# Strategy 2: Sentence-based chunking
# Group complete sentences up to approximately 1000 characters.
import re

sentence_chunks = []

for document in documents:
    sentences = re.split(r"(?<=[.!?])\s+", document["text"].strip())
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue

        if current_chunk and len(current_chunk) + len(sentence) + 1 > 1000:
            sentence_chunks.append({
                "text": current_chunk,
                "source": document["source"],
                "page": document["page"]
            })
            current_chunk = sentence
        else:
            current_chunk = (
                f"{current_chunk} {sentence}".strip()
            )

    if current_chunk:
        sentence_chunks.append({
            "text": current_chunk,
            "source": document["source"],
            "page": document["page"]
        })

# Display comparison
print("========== CHUNKING COMPARISON ==========")
print(f"Pages with extracted text: {len(documents)}")
print(f"Recursive character chunks: {len(recursive_chunks)}")
print(f"Sentence-based chunks:      {len(sentence_chunks)}")

print("\n========== SAMPLE RECURSIVE CHUNK ==========")
print(recursive_chunks[0]["text"][:500])
print("Source:", recursive_chunks[0]["source"])
print("Page:", recursive_chunks[0]["page"])

print("\n========== SAMPLE SENTENCE CHUNK ==========")
print(sentence_chunks[0]["text"][:500])
print("Source:", sentence_chunks[0]["source"])
print("Page:", sentence_chunks[0]["page"])



# Check whether important title-page information is preserved
print("\n========== TITLE PAGE CHECK ==========")

for strategy_name, chunks in [
    ("Recursive character", recursive_chunks),
    ("Sentence-based", sentence_chunks),
]:
    print(f"\n--- {strategy_name} ---")

    for index, chunk in enumerate(chunks):
        if chunk["page"] == 1 and any(
            term in chunk["text"].lower()
            for term in [
                "deep learning in neural networks",
                "schmidhuber",
                "8 october 2014",
            ]
        ):
            print(f"\nChunk index: {index}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(chunk["text"][:700])
