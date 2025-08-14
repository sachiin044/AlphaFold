# 🧬 Mini AlphaFold – Small-Scale Protein Structure & Drug Discovery AI

A lightweight, Streamlit-powered AI system for **protein family classification**, inspired by DeepMind’s AlphaFold but optimized for **smaller datasets**, **fewer parameters**, and **accessible compute environments**.

---

## 1️⃣ Project Overview

**Mini AlphaFold** is designed to bring protein structure–related AI capabilities to researchers, students, and developers without access to supercomputers.  
Instead of tackling full 3D folding like AlphaFold, this project focuses on **Protein Family Classification** using amino acid sequences.

Key highlights:
- Runs locally or on modest cloud instances.
- Built on **transfer learning** from pre-trained protein language models (e.g., ESM-2, ProtBERT).
- Interactive **Streamlit web app** for real-time predictions.

---

## 2️⃣ Motivation

AlphaFold’s breakthrough in protein structure prediction earned global recognition, but its **high computational requirements** limit accessibility for smaller research teams.

This project aims to **democratize protein AI** by:
- Using **lightweight deep learning models** (≤50M parameters).
- Narrowing the focus to **specific, high-impact bioinformatics tasks**.
- Providing an **easy-to-use web interface** for fast experimentation.

---

## 3️⃣ Features

- 🧪 **Protein Family Classification** from amino acid sequences.
- ⚡ **Fast Inference** on CPU or GPU.
- 📊 **Prediction Confidence Scores** displayed alongside results.
- 🌐 **Interactive Web Interface** via Streamlit.
- 🛠 **Modular Codebase** for extending to other protein tasks.
- 📦 **Lightweight Deployment** for local or cloud environments.

---

## 4️⃣ Tech Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Model Architecture:** ECM-based classifier + embeddings from ESM-2 / ProtBERT
- **Dataset:** [SCOPe](http://scop.berkeley.edu/) – Structural Classification of Proteins
- **Visualization:** Streamlit, Matplotlib
- **Utilities:** NumPy, Pandas, scikit-learn, BioPython, tqdm

---

## 5️⃣ Demo & Screenshots

**🔗 Live App:** [Open on Streamlit Cloud](https://sachiin044-hacknation-appstreamlit-app-tqvhzh.streamlit.app)

**📺 Demo Video:** [Watch here](https://drive.google.com/file/d/19PDryaIRcEpJHtE3EIWpYO5bTXdf9g-c/view?usp=drivesdk)

**🖼 Screenshots:**
![Protein Family Classifier Screenshot](https://github.com/user-attachments/assets/525bc008-f882-47ec-803b-b4b33045dfed)

---

## 6️⃣ Installation

1. **Clone the repository**
```bash
git clone https://github.com/sachiin044/AlphaFold.git
cd AlphaFold
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 7️⃣ Usage
**Run locally**
```bash
streamlit run app/streamlit_app.py
```

**Run on Streamlit Cloud**

This project is deployed and accessible online:
[Click here](https://sachiin044-hacknation-appstreamlit-app-tqvhzh.streamlit.app) to try it now

---

## 8️⃣ Input & Output Format

**Input:**

  - A protein sequence in plain text (amino acids in single-letter format, e.g., GDTRPRFLWQLK...).

**Output:**

  - Predicted protein family name.
  - Confidence score (percentage).

```bash
Input Sequence:  GDTRPRFLWQLK...
Predicted Family:  c.47.1.10
Confidence:  92.45%
```

---

## 9️⃣ Project Structure
```bash
AlphaFold/
│
├── app/
│   └── streamlit_app.py       # Streamlit UI for prediction
│
├── src/
│   ├── models/
│   │   └── infer_wrapper.py   # Prediction wrapper logic
│   ├── data/                  # Data loading & preprocessing scripts
│   └── utils/                 # Utility functions
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── ...
```

---

## 🔟 Dataset
- Name: SCOPe (Structural Classification of Proteins)

- Description: A curated dataset for structural and functional classification of proteins.

- Source: http://scop.berkeley.edu/

- Usage in this project: Used to train and evaluate the protein family classifier.

  ---

## 1️⃣1️⃣ Model Architecture

- **Embedding Layer:**  
  Uses **ESM-2** or **ProtBERT** pre-trained protein language model embeddings.  
- **Classification Layer:**  
  An ECM-based (Efficient Convolutional Model) classifier with ≤50M parameters.  
- **Training:**  
  - Optimizer: AdamW  
  - Loss: CrossEntropyLoss  
  - Batch size: Tuned for CPU/GPU constraints  
- **Deployment:**  
  Packaged with Streamlit for interactive use.

---

## 1️⃣2️⃣ Results & Examples

| Sequence Example | Predicted Family | Confidence |
|------------------|------------------|------------|
| GDTRPRFLWQLK...  | c.47.1.10        | 92.45%     |
| MLEKFTYLSVAV...  | a.25.1.2         | 88.10%     |

**Screenshot of Output:**
![Output Example](https://github.com/user-attachments/assets/525bc008-f882-47ec-803b-b4b33045dfed)

---

## 1️⃣3️⃣ Why It Matters

Protein research is the foundation for **drug discovery**, **disease understanding**, and **biotechnology innovation**.  
While large-scale AlphaFold models push the boundaries of science, **Mini AlphaFold** makes protein AI more **accessible**, **cost-effective**, and **practical** for:  
- Academic labs with limited budgets  
- Bioinformatics courses and student projects  
- Rapid prototyping in biotech startups

---

## 1️⃣4️⃣ Future Work / Roadmap

- [ ] Extend to **secondary structure prediction** (alpha-helix, beta-sheet, coil).  
- [ ] Add **ligand–protein binding affinity** prediction module.  
- [ ] Support **batch predictions** for multiple sequences.  
- [ ] Integrate 3D structure visualization via Mol* Viewer or PyMOL.  
- [ ] Improve training pipeline with more protein families.

---

## 1️⃣5️⃣ Contributing

We welcome contributions!  
To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes and push the branch.
4. Open a Pull Request.

Please check our **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## 1️⃣6️⃣ License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 1️⃣7️⃣ Acknowledgements

- **AlphaFold** by DeepMind – Inspiration for this project.  
- **Meta AI’s ESM-2** and **ProtBERT** – Pre-trained protein language models.  
- **SCOPe Dataset** – Protein family classification dataset.  
- Streamlit team for the interactive app framework.

---

## 1️⃣8️⃣ Contact

**Author:** Sachin Gupta  
**GitHub:** [sachiin044](https://github.com/sachiin044)  
**Email:** royalsachingupta@gmail.com  
**LinkedIn:** [sachiin04](https://www.linkedin.com/in/sachiin04)  
**Live Demo:** [Streamlit App](https://sachiin044-hacknation-appstreamlit-app-tqvhzh.streamlit.app/)  

---



