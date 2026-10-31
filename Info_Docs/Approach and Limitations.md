# Approach and Methodology

## 1. Anomaly Detection
Experimented with statistical baselines, Rolling Median and MAD, residual-based detection, and Isolation Forest. **Rolling Median and MAD** was selected due to its stable, interpretable results on the available unlabelled dataset.

Its key strengths include historical behaviour-based detection, robustness to extreme values, and no requirement for labelled training data.

## 2. RAG Pipeline
Implemented the following workflow:

- **Parsing:** Docling
- **Chunking:** LlamaIndex Markdown Splitter
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Database:** ChromaDB
- **Retrieval:** Hybrid BM25 and vector search
- **Reranking:** Cross-Encoder
- **Generation:** Structured prompts with OpenAI GPT-5.4-mini for contextual diagnostics and recommendations.

# Limitations

- **Dataset:** Small, unlabelled telemetry data limits reliable anomaly detection evaluation.
- **Detection:** Rolling Median and MAD depends on window size and historical data; sudden or gradual operating changes may affect detection.
- **Evaluation:** Few anomalous test observations and limited evaluation references constrain RAG metric reliability.
- **Backend:** Complete FastAPI integration with the Copilot interface remains pending.

## Future Improvements
Expand labelled datasets, explore adaptive detection methods, improve RAG evaluation references, and complete end-to-end backend integration.