# Document Q&A with Citations

## Overview

Document Q&A with Citations is a Python-based question-answering system that answers questions using information from a PDF document. It displays the retrieved source passages so users can trace the answers back to the original document.

The system retrieves relevant text passages and uses a Groq language model to generate answers grounded in the retrieved context. When the available document evidence is insufficient, the system responds with **"I don't know based on the provided documents."**

## Features

- Extracts text from PDF documents while preserving page numbers.
- Splits document text into chunks for embedding and retrieval.
- Uses semantic similarity and lexical overlap for hybrid retrieval.
- Generates answers using retrieved document context.
- Displays source passages with filenames, page numbers, and retrieval scores.
- Refuses to answer questions when the available evidence is insufficient.
- Compares recursive character chunking with sentence-based chunking.

## Technology Stack

- **Programming Language:** Python
- **PDF Processing:** PyPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Numerical Computing:** NumPy
- **Language Model API:** Groq
- **Evaluation:** CSV-based retrieval evaluation

## Dataset

The project uses *Deep Learning in Neural Networks: An Overview*, an 88-page PDF document. This document serves as the knowledge source for answering questions and evaluating retrieval performance.

## Chunking Strategy Evaluation

Two chunking strategies were compared:

1. Recursive character chunking
2. Sentence-based chunking

The initial evaluation measured average keyword coverage across five answerable questions at three retrieval settings.

| Chunking Strategy | Top 3 Passages | Top 4 Passages | Top 10 Passages |
|---|---:|---:|---:|
| Recursive character | 61.67% | 66.67% | 83.33% |
| Sentence-based | 49.33% | 56.00% | 62.67% |

Recursive character chunking achieved higher average keyword coverage than sentence-based chunking at all three retrieval settings in this evaluation. We selected four retrieved passages as a practical balance between retrieval coverage and the amount of context supplied to the language model.

The evaluation also included three unanswerable questions. Their average top similarity scores were 0.1775 for recursive character chunking and 0.2025 for sentence-based chunking. These scores show that similarity alone does not guarantee that retrieved passages contain sufficient evidence to answer a question.

These results are preliminary: the evaluation uses only five answerable questions, and keyword coverage does not establish complete factual correctness or citation accuracy. Further testing is needed before drawing broader conclusions.

Project Structure

document-qa-with-citations/
├── document loader/
│   └── Deep Learning in Neural Networks.pdf
├── qa.py
├── retrieval_comparison.py
├── chunking_comparison.py
├── evaluation_questions.csv
├── retrieval_results.csv
├── retrieval_evaluation_details.csv
├── chunking_evaluation_summary.csv
├── check_corpus.py
├── check_pdf_text.py
├── requirements.txt
├── .gitignore
├── README.md
└── FINAL_REPORT.md

Setup

1. Clone the repository

git clone https://github.com/Deekshita293/document-qa-with-citations.git
cd document-qa-with-citations

2. Create and activate a virtual environment

On Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Configure the Groq API key

Create a file named .env in the project root directory and add your own API key:

GROQ_API_KEY=your_groq_api_key

Replace your_groq_api_key with your actual key. Never commit or publish your .env file or API key.

5. Run the question-answering system

python qa.py

Enter questions about the PDF. The application displays an answer along with retrieved source passages, filenames, page numbers, and retrieval scores.

6. Run the retrieval comparison

Stop the question-answering program if it is running, then execute:

python retrieval_comparison.py

Review the generated CSV files to inspect the retrieval evaluation results.

## Example Questions

**Questions supported by the document:**
- How do deep neural networks learn?
- What is the vanishing gradient problem?
- What is the role of LSTM in deep learning?

**Question outside the document's scope:**
- Who won the 2024 Olympic men's 100 metre final?

The final question is unrelated to the document's subject matter. The system is expected to respond with **"I don't know based on the provided documents."** when the available evidence does not support an answer.

## Limitations

- The evaluation dataset contains only a small number of questions.
- Keyword coverage does not fully measure answer correctness.
- Similarity scores alone cannot guarantee that retrieved passages contain sufficient evidence.
- Displaying citations does not guarantee that the generated answer is fully supported by the cited passages.
- The system has been evaluated using a specific PDF and requires further testing on other documents and document collections.

## Future Improvements

- Expand the evaluation dataset with more answerable and unanswerable questions.
- Evaluate answer correctness and citation support.
- Improve retrieval filtering and unsupported-question refusal behavior.
- Develop a user-friendly interface.
- Record and publish a project demonstration video.

## Security

- Store API keys in a local `.env` file.
- Add `.env` and other sensitive configuration files to `.gitignore`.
- Never commit API keys or secrets to GitHub.

## License

This project does not currently specify a license. Add an appropriate open-source license before allowing others to reuse, modify, or distribute the project under its terms.
