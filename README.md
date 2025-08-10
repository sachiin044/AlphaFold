# 🧬 Mini AlphaFold – Small-Scale Protein Structure & Drug Discovery AI

A lightweight, Streamlit-powered AI system for protein structure–related tasks, inspired by AlphaFold but optimized for smaller datasets, fewer parameters, and modest compute environments.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Demo & Media](#demo--media)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples / Results](#examples--results)
- [Why It Matters](#why-it-matters)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview
AlphaFold has revolutionized biology by predicting protein structures with unprecedented accuracy, but its massive compute requirements limit accessibility.  
**Mini AlphaFold** flips the script — delivering **specialized, small-scale models** that tackle targeted protein structure or drug discovery problems efficiently.

This implementation focuses on **Protein Family Classification** using amino acid sequences, enabling researchers and students to experiment with protein structure AI without supercomputers.

---

## 🎥 Demo & Media

**Live App**: [Open Mini AlphaFold on Streamlit Cloud](https://sachiin044-hacknation-appstreamlit-app-tqvhzh.streamlit.app/)  

**Demo Video**:  
[![Watch Demo](images/demo-thumbnail.png)](YOUR_VIDEO_LINK)  

**Screenshots**:  
![App Screenshot](images/app-screenshot.png)  
![Prediction Output](images/output-screenshot.png)  

---

## ✨ Features
- 🧪 **Protein Family Classification** from amino acid sequences.  
- ⚡ **Lightweight Model Training** using ≤50M parameter architectures.  
- 📊 **Prediction Visualization** with confidence scores and graphical output.  
- 🌐 **Interactive Web Interface** via Streamlit.  
- 🛠 **Modular Codebase** for extending to other bioinformatics tasks.

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Model Architecture:** ECM-based + transfer learning from ESM-2 / ProtBERT embeddings  
- **Dataset:** SCOPe (Protein structural classification)  
- **Visualization:** Matplotlib, Streamlit charts  
- **Utilities:** NumPy, Pandas, scikit-learn, BioPython, tqdm  

---

## ⚡ Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/sachiin044/hacknation.git
cd hacknation
```

2️⃣ Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
### Local Run
```bash
streamlit run app.py
```
### Cloud Deployment
This project is already deployed on Streamlit Cloud.  [Click here to try it now](https://sachiin044-hacknation-appstreamlit-app-tqvhzh.streamlit.app/) 

---

## 📂 Project Structure
<img width="695" height="555" alt="image" src="https://github.com/user-attachments/assets/c723cc6b-15bf-4e2f-ad90-2eb8e45947e8" />





















<img width="695" height="555" alt="image" src="https://github.com/user-attachments/assets/c723cc6b-15bf-4e2f-ad90-2eb8e45947e8" />
## project overview
This project is a lightweight version of AlphaFold designed for protein structure related tasking using smaller dataset.

## Features 
predicting family for the user given protein sequence 
## tech stack used
python
pytorch
ECM model architecture
SCOPe dataset
