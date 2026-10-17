# Document Q&A with Citations — Final Project Report

## 1. Abstract

This project implements a document question-answering system that answers user questions using information retrieved from a PDF document. It combines document processing, text chunking, semantic embeddings, hybrid retrieval, and a large language model accessed through the Groq API. The system displays retrieved passages with their filenames and page numbers to make answers traceable to the source document. It also includes refusal behavior for questions that cannot be answered from the available evidence.

## 2. Objectives

- Build a question-answering system for PDF documents.
- Extract text while preserving document page information.
- Compare at least two text chunking strategies.
- Retrieve relevant passages using semantic and lexical signals.
- Generate answers grounded in retrieved context.
- Display source passages for answer verification.
- Refuse unsupported questions rather than guessing.

## 3. Dataset

The project uses *Deep Learning in Neural Networks: An Overview*, a PDF containing 88 pages. The document discusses deep learning, neural networks, backpropagation, recurrent neural networks, and Long Short-Term Memory (LSTM).

The corpus exceeds the required minimum of 50 pages.

## 4. Technology Stack

- Python
- PyPDF for PDF text extraction
- Sentence Transformers using `all-MiniLM-L6-v2`
- NumPy for numerical operations
- Groq API for answer generation
- LangChain text splitters
- CSV files for recording evaluation results

ChromaDB and Streamlit are included in the project dependencies for possible vector storage and interface development.

## 5. Methodology

### 5.1 Document Processing

The PDF is loaded page by page. Extracted text is divided into chunks while preserving the source filename and page number.

### 5.2 Chunking Strategies

Two strategies were compared:

1. Recursive character chunking.
2. Sentence-based chunking.

The initial comparison evaluated retrieval using average keyword coverage across five answerable questions, with the top 3, top 4, and top 10 retrieved passages.

### 5.3 Embedding and Retrieval

The system uses the `all-MiniLM-L6-v2` sentence-transformer model to represent text as embeddings. Retrieval combines semantic similarity and lexical overlap to rank candidate passages.

The question-answering implementation uses four retrieved passages as context for answer generation.

### 5.4 Answer Generation and Citations

The Groq language model generates answers using the retrieved passages. The output includes source identifiers, filenames, page numbers, passage text, and retrieval scores.

### 5.5 Unsupported Questions

A similarity threshold is used as part of the refusal mechanism. If the evidence is insufficient, the system returns:

> I don't know based on the provided documents.

This behavior was tested using questions about the 2024 Olympic men's 100-metre final and the boiling point of pure water at sea level.

## 6. Chunking Evaluation Results

The initial evaluation produced the following average keyword coverage results:

| Chunking strategy | Top 3 | Top 4 | Top 10 |
|---|---:|---:|---:|
| Recursive character | 61.67% | 66.67% | 83.33% |
| Sentence-based | 49.33% | 56.00% | 62.67% |

### 6.1 Discussion

Recursive character chunking achieved higher keyword coverage than sentence-based chunking at all three retrieval settings in this evaluation set. Its coverage increased from 61.67% with three passages to 83.33% with ten passages.

Four passages were selected for the question-answering system as a compromise between retrieval coverage and the amount of context supplied to the language model.

These results are preliminary. The evaluation uses only five answerable questions and keyword coverage rather than a comprehensive measure of factual correctness, answer relevance, or citation support. The results therefore indicate performance on the tested questions and should not be treated as proof that recursive chunking is universally superior.

## 7. Functional Testing

### 7.1 Vanishing Gradient Problem

The system answered the question using a relevant passage from page 16. The passage explains that backpropagated error signals may shrink exponentially or grow excessively in deep networks.

**Observed result:** The retrieved passage directly supported the core answer.

### 7.2 How Deep Neural Networks Learn

The system explained credit assignment and backpropagation, using passages from pages 4 and 7.

**Observed result:** The answer was broadly consistent with the document. Further testing should confirm that each detailed claim is supported by its cited passage.

### 7.3 Role of LSTM

The system explained that LSTM uses Constant Error Carousels to help preserve error signals and learn long-term temporal dependencies. Retrieved passages from pages 19 and 20 supported the explanation.

**Observed result:** The answer was supported by relevant passages, with care needed not to overstate that errors remain unchanged in every part of the network.

### 7.4 Unsupported Questions

Two unsupported questions were tested:

- Who won the 2024 Olympic men's 100 metre final?
- What is the boiling point of pure water at sea level?

For both questions, the system returned the configured refusal response rather than answering from unrelated passages.

**Observed result:** Both refusal tests passed.

## 8. Limitations

- The initial evaluation set is small.
- Keyword coverage does not establish complete answer correctness.
- Similarity scores do not guarantee that a passage supports a claim.
- Citation presence alone does not establish citation accuracy.
- The refusal threshold may behave differently on other questions and documents.
- Evaluation on additional PDFs and a larger set of answerable and unsupported questions is needed.

## 9. Future Work

- Expand the evaluation dataset.
- Measure answer correctness and citation support.
- Improve the detection of irrelevant retrieved passages.
- Test the refusal mechanism across more unsupported questions.
- Add a user interface for uploading PDFs and asking questions.
- Record a demonstration video.
- Publish the code, documentation, and reproducible evaluation results on GitHub.

## 10. Conclusion

The project demonstrates a working PDF question-answering pipeline that combines hybrid retrieval with language-model generation and source-passage display. Initial tests show that the system can retrieve relevant passages for several deep-learning questions and refuse two tested questions outside the document's subject matter.

Recursive character chunking achieved higher keyword coverage than sentence-based chunking in the initial five-question comparison. The system provides a useful foundation for document-grounded question answering, while broader evaluation remains necessary to establish reliable answer accuracy and citation quality.

## 11. Reproducibility and Security

Install the dependencies listed in `requirements.txt`, configure the required Groq API key in a local `.env` file, and run the question-answering script with Python.

The `.env` file and virtual environment must not be committed to GitHub. The final repository should include setup instructions, evaluation outputs, and the final report.
