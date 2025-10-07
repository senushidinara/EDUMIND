# 🎓✨ EduMind AI - Adaptive Knowledge Tracing via Personalized Federated Learning (PFL)

This project simulates a cutting-edge **Personalized Federated Learning (PFL)** system designed for educational applications (**Knowledge Tracing**). Unlike traditional centralized models, this architecture trains a unique, highly accurate prediction model for every student while preserving user privacy 🔒 and leveraging collective data for better generalization 🌐.

The core principle demonstrated is **Transfer Learning (Federated Personalization)**: a Global Model captures general learning patterns, and then its weights are fine-tuned locally using each student's minimal personal data to create a high-fidelity **Personalized Local Model** 🧠.

---

## 🚀 Key Features

- 🧩 **Model-Per-Student Personalization**: A unique `LogisticRegression` model is fine-tuned for each student to predict their probability of success on any given topic.  
- 🌍 **Federated Learning (PFL) Simulation**: Simulates training a shared Global Model followed by Local Model Personalization (Transfer Learning).  
- ⏱️ **Adaptive Scheduling**: Generates personalized **Spaced Repetition** intervals based on a student-specific Forgetting Rate.  
- 🛡️ **Robust Data Handling**: Ensures class variety (both successes ✅ and failures ❌) exists in training data.  
- 🔎 **Interpretability**: Visualizes the difference between Global Model weights and Local Model weights, explaining why each student's recommendations are unique.

---

## 🛠️ Prerequisites

This simulation is written in Python and requires the standard data science stack:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

---

## ⚙️ How to Run the Simulation

1. 📋 **Copy the Code**: Copy the full Python code block for the EduMind AI simulation into your environment (Google Colab, Jupyter Notebook, or any Python IDE).  
2. ▶️ **Run All Cells**: Execute the entire script sequentially. Running all cells will:  
   - Generate simulated student data with personalized learning and forgetting rates  
   - Train the **Global Model** for overall learning patterns  
   - Fine-tune **Local Models** for each student using only their personal data  
   - Compute adaptive review intervals based on each student’s ForgettingRate  
   - Produce visualizations showing differences in feature importance between global and local models  
   - Highlight topics that need more frequent review for each student

---

## 📊 Key Code Sections and Principles

### 1️⃣ Data Simulation
- ⚡ **LearningRate**: How quickly a student’s mastery increases after success  
- 🧠 **ForgettingRate**: How quickly scores drop due to DaysSinceLast session  
- ✅ **Data Fixes**: Ensures every student’s local data contains both successes (Correct=1) and failures (Correct=0), which is necessary for binary classification

### 2️⃣ Global Model Training
- 🏗️ **Model**: `LogisticRegression` with `max_iter=500`  
- 🌐 **Principle**: Federated Averaging (FedAvg) baseline  
- Establishes a common baseline of knowledge and difficulty across all students

### 3️⃣ Local Model Personalization
For each student:  
- 🆕 Initialize a new local model  
- 🔄 Copy global model weights (`coef_` and `intercept_`) to the local model (Transfer Learning)  
- 📝 Fine-tune using only that student’s subset of data  
- **Principle**: Local models start with global knowledge and adapt to individual patterns for high accuracy with minimal data

### 4️⃣ Adaptive Recommendations
- 🎯 **Weakest Link**: Topic with minimum predicted success probability (`preds.idxmin()`)  
- ⏳ **Spaced Repetition**: Interval (`review_days`) calculated using personalized ForgettingRate, enabling adaptive review

---

## 🔍 Model Interpretability (Visualization)

- 🌟 **Global Weights**: Average feature importance (Topic_encoded, DaysSinceLast) for the entire cohort  
- 🧑‍🎓 **Local Weights**: Personalized importance for each student (e.g., S5)  
- 📊 Visualizations highlight why some students need more frequent review than others

---

## 🛑 Notes and Limitations

- 🔒 **Privacy**: Raw data never leaves the client device; only encrypted model updates are shared. This simulation emulates transfer and fine-tuning.  
- 🧩 **Model Simplification**: This simulation uses `LogisticRegression` because it is fast, stable, and easy to interpret, especially for small or synthetic datasets. Real-world Knowledge Tracing often uses Deep Knowledge Tracing (DKT) with RNNs or LSTMs to capture sequential learning patterns. LogisticRegression cannot model dependencies on previous topics.  
- 🔁 **Federated Rounds**: True Personalized Federated Learning (PFL) is iterative. Multiple rounds of global aggregation collect updates from clients, followed by local fine-tuning on each student’s device. Each round improves both global generalization and local personalization. This simulation performs **one global training round and one local fine-tuning round** to illustrate the concept without multi-round complexity.

---

This completes the full EduMind AI README. Copy **Blocks 1, 2, and 3 in order** into a single `.md` file, and it will be a fully unbroken Markdown document.
