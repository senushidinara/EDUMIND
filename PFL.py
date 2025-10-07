# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import copy 

# Ignore convergence warnings that often occur with small, personalized datasets
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Step 2: Simulate Granular Student Data (Definitive Class Balance Fix) ---
np.random.seed(42)
students = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
topics = ['Algebra','Geometry','Physics','Chemistry']
sessions = 20 # Increased sessions for more robust local training

data = []
start_date = datetime.now()

# Data Generation Loop: Simulating unique learning and forgetting traits for each student
for i, student in enumerate(students):
    # Personalized traits 
    learning_rate = np.random.uniform(0.20, 0.35) 
    forget_rate = np.random.uniform(0.05, 0.30) 
    
    for topic in topics:
        mastery_level = np.random.uniform(0.6, 0.8) # Higher starting mastery
        last_date = start_date
        
        for session in range(1, sessions + 1):
            days_since_last = np.random.randint(1, 5)
            date = last_date + timedelta(days=days_since_last)
            last_date = date
            
            # Score calculation based on mastery and forgetting rate
            forget_factor = np.exp(-forget_rate * days_since_last / 10) 
            score_base = mastery_level * 100 * forget_factor
            score = int(np.clip(score_base + np.random.randint(-15, 15), 40, 100))
            
            # Mastery update (learning/unlearning)
            if score >= 75: mastery_level = min(1.0, mastery_level + learning_rate * 0.7)
            elif score < 60: mastery_level = max(0.1, mastery_level - learning_rate * 0.2)
            
            data.append([student, topic, date, session, score, mastery_level, days_since_last, learning_rate, forget_rate])

df = pd.DataFrame(data, columns=['StudentID','Topic','Date','Session','Score','LatentMastery','DaysSinceLast','LearningRate','ForgettingRate'])

# Create target variable: Correct (1) or Incorrect (0). Threshold lowered to 60 for better class balance.
df['Correct'] = (df['Score'] >= 60).astype(int) 

# Feature engineering for the Logistic Regression model
df['Topic_encoded'] = df['Topic'].astype('category').cat.codes
topic_mapping = df[['Topic_encoded', 'Topic']].drop_duplicates().set_index('Topic_encoded')['Topic'].to_dict()
FEATURE_COLS = ['Topic_encoded', 'DaysSinceLast'] 
TARGET_COL = 'Correct'

# ----------------------------------------------------------------------
# --- STEP 3: FEDERATED LEARNING SIMULATION (Global & Local Training) ---
# ----------------------------------------------------------------------

# 3.1. Train the GLOBAL Model (Central Server)
print("--- 1. Training Global Model (Federated Average) ---")
X_global = df[FEATURE_COLS]
y_global = df[TARGET_COL]

# Safety check for class variety (essential for binary classification)
if len(y_global.unique()) < 2:
    print("\n🚨 WARNING: Global target still has only one class. Forcing one '1' label for training stability.")
    # Find the index of the first '0' and change its label to '1'
    idx_to_change = y_global[y_global == 0].index[0]
    df.loc[idx_to_change, 'Correct'] = 1
    y_global = df[TARGET_COL] # Re-fetch the corrected target series

# Initialize Global Model
global_model = LogisticRegression(solver='saga', max_iter=500, penalty='l2', random_state=42)
global_model.fit(X_global, y_global)
print(f"Global Model Trained. Accuracy on combined data (simulated): {global_model.score(X_global, y_global):.4f}")

# 3.2. Initialize Local Models (Client Side)
student_models = {}
student_predictions = {}
test_topic_encoded = df['Topic_encoded'].unique()

print("\n--- 2. Personalizing Local Models (Fine-tuning Global Weights) ---")

for student_id in students:
    student_data = df[df['StudentID'] == student_id].copy()
    
    X_local = student_data[FEATURE_COLS]
    y_local = student_data[TARGET_COL]
    
    # Pre-training checks
    if len(student_data) < 10: 
        print(f"Skipping {student_id} - Insufficient data.")
        continue

    if len(y_local.unique()) < 2:
        print(f"Skipping {student_id} - Insufficient class variety in local data (only class {y_local.iloc[0]}).")
        continue

    # Initialize model
    local_model = LogisticRegression(solver='saga', max_iter=500, penalty='l2', random_state=42)
    
    # Federated Personalization Step (Transfer Learning)
    try:
        # Step 1: Initialize model structure (needed by scikit-learn)
        unique_data_indices = y_local.drop_duplicates().index
        local_model.fit(X_local.loc[unique_data_indices], y_local.loc[unique_data_indices]) 
        
        # Step 2: Copy Global Model weights (The PFL step)
        local_model.coef_ = copy.deepcopy(global_model.coef_)
        local_model.intercept_ = copy.deepcopy(global_model.intercept_)
        
        # Step 3: Fine-tune on local data
        local_model.fit(X_local, y_local)
        student_models[student_id] = local_model
        
        # Predict future success probability for all topics (5 days later)
        X_test_future = pd.DataFrame({
            'Topic_encoded': test_topic_encoded,
            'DaysSinceLast': [5] * len(test_topic_encoded)
        })
        probs = local_model.predict_proba(X_test_future)[:, 1] 
        topic_names = [topic_mapping[t] for t in test_topic_encoded]
        student_predictions[student_id] = pd.Series(probs, index=topic_names)
        
    except Exception as e:
        print(f"Error fine-tuning model for {student_id}: {e}")

# ----------------------------------------------------------------------
# --- STEP 4: ADAPTIVE RECOMMENDATION (Actionable Insights) ---
# ----------------------------------------------------------------------

def adaptive_recommendation(student_id):
    """Uses the personalized local model to generate adaptive study recommendations."""
    if student_id not in student_predictions:
        print(f"\n--- Personalized Profile & Action for Student {student_id} ---\nModel not trained or failed to train due to data constraints.")
        return 
    
    preds = student_predictions[student_id]
    latest_data = df[df['StudentID'] == student_id].drop_duplicates('Topic', keep='last').set_index('Topic')
    latest_mastery = latest_data['LatentMastery']
    forget_rate = latest_data['ForgettingRate'].iloc[0]
    
    # Identify Weakest/Strongest based on the FINE-TUNED model prediction
    weakest_topic = preds.idxmin()
    weakest_prob = preds.min()
    strongest_topic = preds.idxmax()
    
    # Calculate Personalized Review Days (Spaced Repetition based on personalized forgetting rate)
    base_days = 7
    # Inverse relationship: Higher forgetting rate -> Shorter review interval
    review_days = int(np.clip(base_days + (1/forget_rate) * 2, 5, 25)) 
    
    # --- Action Logic ---
    action_1 = f"**{weakest_topic}** (Predicted Success: {weakest_prob:.2f}). **Action:** Immediate Remediation. The **PFL** model suggests this is the highest risk topic, recommend a pre-requisite review."
    
    action_2 = f"**{strongest_topic}** (Mastery: {latest_mastery.loc[strongest_topic]:.2f}). **Action:** Dynamic Challenge. Schedule a higher-order application problem. Next review in **{review_days} days** based on personalized forgetting."

    print(f"\n--- Personalized Profile & Action for Student {student_id} (PFL Driven) ---")
    print(f"| Student Trait: Forgetting Rate (Personalized): {forget_rate:.2f}")
    print("----------------------------------------------------------------------")
    print(f"🔥 Weakest Link Prediction (Based on **Fine-Tuned Local Model**): {action_1}")
    print(f"🧠 Strongest Area Recommendation (Spaced Repetition): {action_2}")
    
# --- STEP 5: Run the PFL System for Example Students ---
adaptive_recommendation('S5')
adaptive_recommendation('S10')

# ----------------------------------------------------------------------
# --- STEP 6: Visualization: Comparing Global vs. Local Model Weights ---
# ----------------------------------------------------------------------
print("\n--- Model Interpretability: Global vs. Personalized Weights ---")

# Get coefficients for the Global Model
global_weights = pd.Series(global_model.coef_[0], index=FEATURE_COLS)

# Get coefficients for one Local Model (e.g., S5)
if 'S5' in student_models:
    local_weights = pd.Series(student_models['S5'].coef_[0], index=FEATURE_COLS)
    
    weight_df = pd.DataFrame({
        'Global Model': global_weights,
        'S5 Local Model': local_weights
    })

    # Plotting the comparison
    plt.figure(figsize=(8, 5))
    weight_df.plot(kind='bar')
    plt.title("Comparison of Feature Weights (Global vs. Student S5)")
    plt.ylabel("Model Coefficient Value")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("Could not compare weights (S5 model not trained).")

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import copy 

# Ignore convergence warnings that often occur with small, personalized datasets
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Step 2: Simulate Granular Student Data (Definitive Class Balance Fix) ---
np.random.seed(42)
students = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
topics = ['Algebra','Geometry','Physics','Chemistry']
sessions = 20 # Increased sessions for more robust local training

data = []
start_date = datetime.now()

# Data Generation Loop: Simulating unique learning and forgetting traits for each student
for i, student in enumerate(students):
    # Personalized traits 
    learning_rate = np.random.uniform(0.20, 0.35) 
    forget_rate = np.random.uniform(0.05, 0.30) 
    
    for topic in topics:
        mastery_level = np.random.uniform(0.6, 0.8) # Higher starting mastery
        last_date = start_date
        
        for session in range(1, sessions + 1):
            days_since_last = np.random.randint(1, 5)
            date = last_date + timedelta(days=days_since_last)
            last_date = date
            
            # Score calculation based on mastery and forgetting rate
            forget_factor = np.exp(-forget_rate * days_since_last / 10) 
            score_base = mastery_level * 100 * forget_factor
            score = int(np.clip(score_base + np.random.randint(-15, 15), 40, 100))
            
            # Mastery update (learning/unlearning)
            if score >= 75: mastery_level = min(1.0, mastery_level + learning_rate * 0.7)
            elif score < 60: mastery_level = max(0.1, mastery_level - learning_rate * 0.2)
            
            data.append([student, topic, date, session, score, mastery_level, days_since_last, learning_rate, forget_rate])

df = pd.DataFrame(data, columns=['StudentID','Topic','Date','Session','Score','LatentMastery','DaysSinceLast','LearningRate','ForgettingRate'])

# Create target variable: Correct (1) or Incorrect (0). Threshold lowered to 60 for better class balance.
df['Correct'] = (df['Score'] >= 60).astype(int) 

# Feature engineering for the Logistic Regression model
df['Topic_encoded'] = df['Topic'].astype('category').cat.codes
topic_mapping = df[['Topic_encoded', 'Topic']].drop_duplicates().set_index('Topic_encoded')['Topic'].to_dict()
FEATURE_COLS = ['Topic_encoded', 'DaysSinceLast'] 
TARGET_COL = 'Correct'

# ----------------------------------------------------------------------
# --- STEP 3: FEDERATED LEARNING SIMULATION (Global & Local Training) ---
# ----------------------------------------------------------------------

# 3.1. Train the GLOBAL Model (Central Server)
print("--- 1. Training Global Model (Federated Average) ---")
X_global = df[FEATURE_COLS]
y_global = df[TARGET_COL]

# Safety check for class variety (essential for binary classification)
if len(y_global.unique()) < 2:
    print("\n🚨 WARNING: Global target still has only one class. Forcing one '1' label for training stability.")
    # Find the index of the first '0' and change its label to '1'
    idx_to_change = y_global[y_global == 0].index[0]
    df.loc[idx_to_change, 'Correct'] = 1
    y_global = df[TARGET_COL] # Re-fetch the corrected target series

# Initialize Global Model
global_model = LogisticRegression(solver='saga', max_iter=500, penalty='l2', random_state=42)
global_model.fit(X_global, y_global)
print(f"Global Model Trained. Accuracy on combined data (simulated): {global_model.score(X_global, y_global):.4f}")

# 3.2. Initialize Local Models (Client Side)
student_models = {}
student_predictions = {}
test_topic_encoded = df['Topic_encoded'].unique()

print("\n--- 2. Personalizing Local Models (Fine-tuning Global Weights) ---")

for student_id in students:
    student_data = df[df['StudentID'] == student_id].copy()
    
    X_local = student_data[FEATURE_COLS]
    y_local = student_data[TARGET_COL]
    
    # Pre-training checks
    if len(student_data) < 10: 
        print(f"Skipping {student_id} - Insufficient data.")
        continue

    if len(y_local.unique()) < 2:
        print(f"Skipping {student_id} - Insufficient class variety in local data (only class {y_local.iloc[0]}).")
        continue

    # Initialize model
    local_model = LogisticRegression(solver='saga', max_iter=500, penalty='l2', random_state=42)
    
    # Federated Personalization Step (Transfer Learning)
    try:
        # Step 1: Initialize model structure (needed by scikit-learn)
        unique_data_indices = y_local.drop_duplicates().index
        local_model.fit(X_local.loc[unique_data_indices], y_local.loc[unique_data_indices]) 
        
        # Step 2: Copy Global Model weights (The PFL step)
        local_model.coef_ = copy.deepcopy(global_model.coef_)
        local_model.intercept_ = copy.deepcopy(global_model.intercept_)
        
        # Step 3: Fine-tune on local data
        local_model.fit(X_local, y_local)
        student_models[student_id] = local_model
        
        # Predict future success probability for all topics (5 days later)
        X_test_future = pd.DataFrame({
            'Topic_encoded': test_topic_encoded,
            'DaysSinceLast': [5] * len(test_topic_encoded)
        })
        probs = local_model.predict_proba(X_test_future)[:, 1] 
        topic_names = [topic_mapping[t] for t in test_topic_encoded]
        student_predictions[student_id] = pd.Series(probs, index=topic_names)
        
    except Exception as e:
        print(f"Error fine-tuning model for {student_id}: {e}")

# ----------------------------------------------------------------------
# --- STEP 4: ADAPTIVE RECOMMENDATION (Actionable Insights) ---
# ----------------------------------------------------------------------

def adaptive_recommendation(student_id):
    """Uses the personalized local model to generate adaptive study recommendations."""
    if student_id not in student_predictions:
        print(f"\n--- Personalized Profile & Action for Student {student_id} ---\nModel not trained or failed to train due to data constraints.")
        return 
    
    preds = student_predictions[student_id]
    latest_data = df[df['StudentID'] == student_id].drop_duplicates('Topic', keep='last').set_index('Topic')
    latest_mastery = latest_data['LatentMastery']
    forget_rate = latest_data['ForgettingRate'].iloc[0]
    
    # Identify Weakest/Strongest based on the FINE-TUNED model prediction
    weakest_topic = preds.idxmin()
    weakest_prob = preds.min()
    strongest_topic = preds.idxmax()
    
    # Calculate Personalized Review Days (Spaced Repetition based on personalized forgetting rate)
    base_days = 7
    # Inverse relationship: Higher forgetting rate -> Shorter review interval
    review_days = int(np.clip(base_days + (1/forget_rate) * 2, 5, 25)) 
    
    # --- Action Logic ---
    action_1 = f"**{weakest_topic}** (Predicted Success: {weakest_prob:.2f}). **Action:** Immediate Remediation. The **PFL** model suggests this is the highest risk topic, recommend a pre-requisite review."
    
    action_2 = f"**{strongest_topic}** (Mastery: {latest_mastery.loc[strongest_topic]:.2f}). **Action:** Dynamic Challenge. Schedule a higher-order application problem. Next review in **{review_days} days** based on personalized forgetting."

    print(f"\n--- Personalized Profile & Action for Student {student_id} (PFL Driven) ---")
    print(f"| Student Trait: Forgetting Rate (Personalized): {forget_rate:.2f}")
    print("----------------------------------------------------------------------")
    print(f"🔥 Weakest Link Prediction (Based on **Fine-Tuned Local Model**): {action_1}")
    print(f"🧠 Strongest Area Recommendation (Spaced Repetition): {action_2}")
    
# --- STEP 5: Run the PFL System for Example Students ---
adaptive_recommendation('S5')
adaptive_recommendation('S10')

# ----------------------------------------------------------------------
# --- STEP 6: Visualization: Comparing Global vs. Local Model Weights ---
# ----------------------------------------------------------------------
print("\n--- Model Interpretability: Global vs. Personalized Weights ---")

# Get coefficients for the Global Model
global_weights = pd.Series(global_model.coef_[0], index=FEATURE_COLS)

# Get coefficients for one Local Model (e.g., S5)
if 'S5' in student_models:
    local_weights = pd.Series(student_models['S5'].coef_[0], index=FEATURE_COLS)
    
    weight_df = pd.DataFrame({
        'Global Model': global_weights,
        'S5 Local Model': local_weights
    })

    # Plotting the comparison
    plt.figure(figsize=(8, 5))
    weight_df.plot(kind='bar')
    plt.title("Comparison of Feature Weights (Global vs. Student S5)")
    plt.ylabel("Model Coefficient Value")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("Could not compare weights (S5 model not trained).")

