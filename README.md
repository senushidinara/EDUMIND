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
