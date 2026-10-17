import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PDF_FOLDER = Path("document loader")
QUESTIONS_FILE = Path("evaluation_questions.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Main experiment uses K=4.
MAIN_K = 4

# Compare retrieval depth using the same questions and chunks.
K_VALUES = [3, 4, 10]
MAX_K = max(K_VALUES)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RESULTS_FILE = Path("retrieval_results.csv")
DETAILS_FILE = Path("retrieval_evaluation_details.csv")
SUMMARY_FILE = Path("chunking_evaluation_summary.csv")


# ==========================================================
# 2. LOAD PDF DOCUMENTS AND PRESERVE PAGE METADATA
# ==========================================================

def load_documents():
    if not PDF_FOLDER.exists():
        raise SystemExit(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {PDF_FOLDER}")

    documents = []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()

                if text:
                    documents.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number
                    })

        except Exception as error:
            print(f"Warning: Could not read {pdf_path.name}: {error}")

    if not documents:
        raise SystemExit(
            "No readable PDF text was extracted. "
            "Check whether your PDFs contain selectable text."
        )

    print(f"PDF files found: {len(pdf_files)}")
    print(f"Pages with extracted text: {len(documents)}")

    total_characters = sum(len(doc["text"]) for doc in documents)
    print(f"Total extracted characters: {total_characters:,}")

    return documents


# ==========================================================
# 3. STRATEGY A: RECURSIVE CHARACTER CHUNKING
# ==========================================================

def create_recursive_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for doc in documents:
        page_chunks = splitter.split_text(doc["text"])

        for chunk in page_chunks:
            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "source": doc["source"],
                    "page": doc["page"]
                })

    return chunks


# ==========================================================
# 4. STRATEGY B: SENTENCE-BASED CHUNKING
# ==========================================================

def create_sentence_chunks(documents):
    chunks = []

    # Used only when an individual sentence exceeds the size limit.
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0
    )

    for doc in documents:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            doc["text"].strip()
        )

        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Keep even unusually long sentences within the size limit.
            if len(sentence) > CHUNK_SIZE:
                if current_sentences:
                    chunks.append({
                        "text": " ".join(current_sentences),
                        "source": doc["source"],
                        "page": doc["page"]
                    })
                    current_sentences = []
                    current_length = 0

                for part in fallback_splitter.split_text(sentence):
                    if part.strip():
                        chunks.append({
                            "text": part.strip(),
                            "source": doc["source"],
                            "page": doc["page"]
                        })

                continue

            extra_length = len(sentence)
            if current_sentences:
                extra_length += 1

            if (
                current_sentences
                and current_length + extra_length > CHUNK_SIZE
            ):
                chunks.append({
                    "text": " ".join(current_sentences),
                    "source": doc["source"],
                    "page": doc["page"]
                })

                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length = len(" ".join(current_sentences))

        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "source": doc["source"],
                "page": doc["page"]
            })

    return chunks


# ==========================================================
# 5. LOAD AND VALIDATE EVALUATION QUESTIONS
# ==========================================================

def load_questions():
    if not QUESTIONS_FILE.exists():
        raise SystemExit(
            f"Question file not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise SystemExit("The evaluation CSV has no header row.")

        headers = {
            header.strip().lower()
            for header in reader.fieldnames
            if header
        }

        required_headers = {
            "question",
            "expected_answer",
            "answerable"
        }

        missing = required_headers - headers

        if missing:
            raise SystemExit(
                "Missing CSV columns: "
                + ", ".join(sorted(missing))
                + "\nRequired columns: question,expected_answer,answerable"
            )

        questions = []

        for row in reader:
            normalized_row = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }

            question = normalized_row.get("question", "")
            expected_answer = normalized_row.get("expected_answer", "")
            answerable = normalized_row.get("answerable", "").lower()

            if not question:
                continue

            if answerable not in {"true", "false"}:
                raise SystemExit(
                    f"Invalid answerable value for question: {question}\n"
                    "Use true or false."
                )

            if not expected_answer:
                raise SystemExit(
                    f"Missing expected_answer for question: {question}"
                )

            questions.append({
                "question": question,
                "expected_answer": expected_answer,
                "answerable": answerable == "true"
            })

    if not questions:
        raise SystemExit("No valid evaluation questions were found.")

    print(f"Evaluation questions loaded: {len(questions)}")
    return questions


# ==========================================================
# 6. TEXT NORMALIZATION AND KEYWORD EVALUATION
# ==========================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "in", "is", "it", "of", "on", "or",
    "the", "to", "was", "were", "what", "who", "with"
}


def normalize_text(text):
    # Normalize common PDF Unicode encoding issues.
    text = text.replace("¨", "")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return text.lower()


def meaningful_terms(text):
    words = re.findall(
        r"\b[a-z0-9]+\b",
        normalize_text(text)
    )

    return {
        word for word in words
        if word not in STOP_WORDS
    }


def calculate_keyword_coverage(expected_answer, retrieved_text):
    expected_terms = meaningful_terms(expected_answer)
    retrieved_terms = meaningful_terms(retrieved_text)

    if not expected_terms:
        return 0.0, set(), set()

    matched_terms = expected_terms & retrieved_terms

    coverage = (
        len(matched_terms) / len(expected_terms)
    ) * 100

    return coverage, expected_terms, matched_terms


# ==========================================================
# 7. EMBEDDING AND RETRIEVAL
# ==========================================================

def retrieve_passages(strategy_name, chunks, questions, model):
    if not chunks:
        raise ValueError(f"No chunks created for {strategy_name}.")

    print(f"\nEncoding {strategy_name} chunks: {len(chunks)}")

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_vectors = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Store results for up to MAX_K passages per question.
    results = []

    for question_item in questions:
        question = question_item["question"]

        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]

        # Since vectors are normalized, dot product = cosine similarity.
        scores = chunk_vectors @ query_vector

        # Stable ordering makes ties more reproducible.
        top_indices = np.argsort(-scores, kind="stable")[:MAX_K]

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[int(index)]

            results.append({
                "strategy": strategy_name,
                "question": question,
                "expected_answer": question_item["expected_answer"],
                "answerable": question_item["answerable"],
                "rank": rank,
                "similarity": round(float(scores[index]), 6),
                "source": chunk["source"],
                "page": chunk["page"],
                "retrieved_text": chunk["text"]
            })

    return results


# ==========================================================
# 8. EVALUATE DIFFERENT K VALUES
# ==========================================================

def evaluate_results(all_results, questions):
    details = []
    summary = []

    strategies = [
        "Recursive character",
        "Sentence-based"
    ]

    for strategy in strategies:
        strategy_results = [
            result for result in all_results
            if result["strategy"] == strategy
        ]

        for k in K_VALUES:
            answerable_coverages = []
            fully_covered = 0
            unanswerable_top_scores = []

            for item in questions:
                question = item["question"]

                retrieved = [
                    result for result in strategy_results
                    if result["question"] == question
                    and result["rank"] <= k
                ]

                retrieved.sort(key=lambda result: result["rank"])

                combined_text = "\n".join(
                    result["retrieved_text"]
                    for result in retrieved
                )

                top_score = (
                    retrieved[0]["similarity"]
                    if retrieved else None
                )

                if item["answerable"]:
                    coverage, expected_terms, matched_terms = (
                        calculate_keyword_coverage(
                            item["expected_answer"],
                            combined_text
                        )
                    )

                    answerable_coverages.append(coverage)

                    if coverage == 100.0 and expected_terms:
                        fully_covered += 1

                    details.append({
                        "strategy": strategy,
                        "k": k,
                        "question": question,
                        "answerable": True,
                        "expected_answer": item["expected_answer"],
                        "expected_keyword_count": len(expected_terms),
                        "matched_keyword_count": len(matched_terms),
                        "matched_keywords": ", ".join(
                            sorted(matched_terms)
                        ),
                        "keyword_coverage_percent": round(coverage, 2),
                        "top_similarity": top_score,
                        "top_source": (
                            retrieved[0]["source"] if retrieved else ""
                        ),
                        "top_page": (
                            retrieved[0]["page"] if retrieved else ""
                        )
                    })

                else:
                    # Expected answers to unanswerable questions are not
                    # scored with keyword coverage. A retrieved passage
                    # may mention the expected answer by coincidence.
                    if top_score is not None:
                        unanswerable_top_scores.append(top_score)

                    details.append({
                        "strategy": strategy,
                        "k": k,
                        "question": question,
                        "answerable": False,
                        "expected_answer": item["expected_answer"],
                        "expected_keyword_count": "",
                        "matched_keyword_count": "",
                        "matched_keywords": "",
                        "keyword_coverage_percent": "",
                        "top_similarity": top_score,
                        "top_source": (
                            retrieved[0]["source"] if retrieved else ""
                        ),
                        "top_page": (
                            retrieved[0]["page"] if retrieved else ""
                        )
                    })

            average_coverage = (
                sum(answerable_coverages) / len(answerable_coverages)
                if answerable_coverages else 0.0
            )

            average_unanswerable_score = (
                sum(unanswerable_top_scores)
                / len(unanswerable_top_scores)
                if unanswerable_top_scores else None
            )

            summary.append({
                "strategy": strategy,
                "k": k,
                "answerable_question_count": len(answerable_coverages),
                "fully_covered_answerable_questions": fully_covered,
                "average_keyword_coverage_percent": round(
                    average_coverage, 2
                ),
                "unanswerable_question_count": sum(
                    not item["answerable"] for item in questions
                ),
                "average_top_similarity_unanswerable": (
                    round(average_unanswerable_score, 6)
                    if average_unanswerable_score is not None
                    else ""
                )
            })

    return details, summary


# ==========================================================
# 9. SAVE CSV FILES
# ==========================================================

def save_csv(file_path, rows):
    if not rows:
        print(f"No data to save: {file_path}")
        return

    with file_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {file_path.resolve()}")


# ==========================================================
# 10. MAIN PROGRAM
# ==========================================================

def main():
    documents = load_documents()

    recursive_chunks = create_recursive_chunks(documents)
    sentence_chunks = create_sentence_chunks(documents)

    print("\n========== CHUNK STATISTICS ==========")
    print(f"Recursive character chunks: {len(recursive_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")

    print(
        "Recursive average chunk length: "
        f"{np.mean([len(c['text']) for c in recursive_chunks]):.1f}"
        if recursive_chunks else "No recursive chunks"
    )

    print(
        "Sentence-based average chunk length: "
        f"{np.mean([len(c['text']) for c in sentence_chunks]):.1f}"
        if sentence_chunks else "No sentence-based chunks"
    )

    questions = load_questions()

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    recursive_results = retrieve_passages(
        "Recursive character",
        recursive_chunks,
        questions,
        model
    )

    sentence_results = retrieve_passages(
        "Sentence-based",
        sentence_chunks,
        questions,
        model
    )

    all_results = recursive_results + sentence_results

    details, summary = evaluate_results(all_results, questions)

    # Save all retrieved passages up to K=10.
    save_csv(RESULTS_FILE, all_results)
    save_csv(DETAILS_FILE, details)
    save_csv(SUMMARY_FILE, summary)

    print("\n========== MAIN EXPERIMENT: K = 4 ==========")

    for row in summary:
        if row["k"] != MAIN_K:
            continue

        print(f"\nStrategy: {row['strategy']}")
        print(f"K: {row['k']}")
        print(
            "Average keyword coverage on answerable questions: "
            f"{row['average_keyword_coverage_percent']}%"
        )
        print(
            "Fully covered answerable questions: "
            f"{row['fully_covered_answerable_questions']}/"
            f"{row['answerable_question_count']}"
        )

    print("\n========== COMPARISON: K = 3, 4, 10 ==========")

    for row in summary:
        print(
            f"{row['strategy']} | K={row['k']} | "
            f"Average keyword coverage="
            f"{row['average_keyword_coverage_percent']}%"
        )

    print("\n========== TOP PASSAGES AT K = 4 ==========")

    for strategy in ["Recursive character", "Sentence-based"]:
        print(f"\n--- {strategy} ---")

        for item in questions:
            print(f"\nQuestion: {item['question']}")

            question_results = [
                result for result in all_results
                if result["strategy"] == strategy
                and result["question"] == item["question"]
                and result["rank"] <= MAIN_K
            ]

            for result in question_results[:3]:
                print(
                    f"Rank {result['rank']} | "
                    f"Similarity: {result['similarity']:.4f} | "
                    f"Source: {result['source']} | "
                    f"Page: {result['page']}"
                )
                print(
                    result["retrieved_text"][:300].replace("\n", " ")
                )

    print("\nEvaluation completed.")

    print(
        "\nIMPORTANT:"
        "\n- K=4 is the main experiment setting."
        "\n- K=3 and K=10 are comparison settings."
        "\n- Keyword coverage is a diagnostic, not proof of answer correctness."
        "\n- Similarity scores alone cannot determine whether a question "
        "is answerable."
        "\n- Review retrieved passages before selecting the winning strategy."
        "\n- This script evaluates retrieval; it does not generate answers "
        "or implement the final 'I don't know' response."
    )


if __name__ == "__main__":
    main()