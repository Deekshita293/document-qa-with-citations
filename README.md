# Document Q&A with Citations

## Overview

Document Q&A with Citations is a Python-based question-answering system that answers questions using a PDF document and displays the retrieved source passages for traceability.

The system retrieves relevant text, uses a Groq language model to generate answers, and returns an "I don't know" response when the available document evidence is insufficient.

## Features

- PDF text extraction with page numbers.
- Text chunking, embedding, and semantic retrieval.
- Hybrid retrieval using semantic similarity and lexical overlap.
- Answers generated using retrieved document context.
- Source passages displayed with filenames, page numbers, and retrieval scores.
- Refusal behavior for unsupported questions.
- Comparison of recursive character chunking and sentence-based chunking.

## Technology Stack

- Python
- PyPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- NumPy
- Groq API
- CSV-based evaluation

## Dataset

The project uses *Deep Learning in Neural Networks: An Overview*, a PDF containing 88 pages.

## Chunking Strategy Evaluation

Two chunking strategies were evaluated:

1. Recursive character chunking
2. Sentence-based chunking

Initial evaluation results using average keyword coverage across five answerable questions:

| Strategy | Top 3 passages | Top 4 passages | Top 10 passages |
|---|---:|---:|---:|
| Recursive character | 61.33% | 66.67% | 83.33% |
| Sentence-based | 49.33% | 56.00% | 62.67% |

Recursive character chunking performed better on this initial evaluation set. We selected four retrieved passages as a balance between retrieval quality and the amount of context provided to the language model.

These results measure keyword coverage, not complete factual correctness or citation accuracy. Further evaluation is needed before generalizing the findings.

## Project Structure

Document Q&A with Citations/
├── document loader/
│   └── Deep Learning in Neural Networks.pdf
├── qa.py
├── retrieval_comparison.py
├── chunking_comparison.py
├── evaluation_questions.csv
├── retrieval_evaluation_details.csv
├── retrieval_results.csv
├── chunking_evaluation_summary.csv
├── check_corpus.py
├── check_pdf_text.py
├── requirements.txt
├── .env                 
├── .gitignore
└── README.md

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Document Q&A with Citations"
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project directory:

```text
GROQ_API_KEY=your_groq_api_key
```

Use the environment variable names expected by your code. Never publish your actual API key.

### 5. Run the question-answering system

```bash
python qa.py
```

Ask questions about the PDF. The application displays an answer and its retrieved source passages.

### 6. Run the retrieval comparison

```bash
python retrieval_comparison.py
```

Check the generated CSV files for the comparison results.

## Example Questions

- How do deep neural networks learn?
- What is the vanishing gradient problem?
- What is the role of LSTM in deep learning?
- Who won the 2024 Olympic men's 100 metre final?

The final question is outside the document's subject matter and should be refused when the available evidence is insufficient.

## Limitations

- The evaluation dataset is small.
- Keyword coverage does not fully measure answer correctness.
- Similarity scores alone cannot guarantee that retrieved passages support an answer.
- Citation presence does not guarantee factual support.
- The system has been tested on a specific PDF and requires broader evaluation for other document collections.

## Future Improvements

- Expand the evaluation dataset.
- Measure answer correctness and citation support.
- Improve retrieval filtering and refusal behavior.
- Add a user interface.
- Record a project demonstration video.

## Security

API keys and local environment files must not be committed to GitHub.

## License

Add an appropriate license before distributing the project publicly.
