mini_alphafold/
├─ README.md
├─ requirements.txt
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ data/
│  │  └─ dataset_builder.py
│  ├─ preprocess/
│  │  └─ tokenizer_and_utils.py   # shared token/sequence functions
│  ├─ embeddings/
│  │  └─ embedder.py              # code to call ESM/ProtBERT or load cached embeddings
│  ├─ models/
│  │  ├─ classifier.py            # model architectures & load/save
│  │  └─ trainer.py               # training loop (CPU-friendly)
│  ├─ inference/
│  │  └─ predictor.py             # single function predict(sequence)->json
│  └─ metrics/
│     └─ eval.py
├─ notebooks/                      # experiments, EDA
├─ models/                         # small weights, or pointers to HF hub
├─ tests/
├─ .github/workflows/ci.yml
