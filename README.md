# --- Step 1: Install Libraries ---
!pip install pandas numpy scikit-learn matplotlib seaborn

# --- Step 2: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 3: Simulate Student Data ---
np.random.seed(42)
students = ['S1','S2','S3','S4','S5']
topics = ['Algebra','Geometry','Physics','Chemistry','Biology']

data = []
for student in students:
    for topic in topics:
        score = np.random.randint(40,100)  # Random scores 40-100
        time_spent = np.random.randint(20,60)  # Minutes spent
        # NEW FEATURE: Include a difficulty level (1=Easy, 2=Medium, 3=Hard)
        difficulty = np.random.choice([1, 2, 3])
        data.append([student, topic, score, time_spent, difficulty])

df = pd.DataFrame(data, columns=['StudentID','Topic','Score','TimeSpent', 'Difficulty'])

# Encode topic for ML
df['Topic_encoded'] = df['Topic'].astype('category').cat.codes

# Define performance: Weak=0 (<60), Medium=1 (60-79), Strong=2 (80+)
df['Performance'] = df['Score'].apply(lambda x: 0 if x<60 else (1 if x<80 else 2))
print("Sample Data:")
print(df.head())

# --- Step 4: Split Features and Labels (Now including Difficulty) ---
X = df[['Topic_encoded','TimeSpent', 'Difficulty']] # Added 'Difficulty' as a feature
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Train Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 6: Predict Weak Topics for a New Student ---
# Simulate a new student's study time per topic and current difficulty
new_student_data = pd.DataFrame({
    'Topic_encoded': [0, 1, 2, 3, 4],  # Algebra, Geometry, Physics, Chemistry, Biology
    'TimeSpent': [25, 35, 30, 20, 40],
    'Difficulty': [2, 1, 3, 2, 1] # Current assumed difficulty level for the student
})
new_student_topics = ['Algebra', 'Geometry', 'Physics', 'Chemistry', 'Biology']

pred = model.predict(new_student_data)
# NEW FEATURE: Predict the probability distribution (confidence)
pred_proba = model.predict_proba(new_student_data)

pred_labels = ['Weak','Medium','Strong']
pred_text = [pred_labels[i] for i in pred]
print("\n--- Predicted Performance ---")
print("Predicted performance per topic:", pred_text)

# ----------------------------------------------------------------------
# --- NEW FEATURE 1: Confidence Score & Adaptive Difficulty ---
# ----------------------------------------------------------------------

print("\n--- AI Confidence & Next Steps ---")
difficulty_map = {1: 'Easy', 2: 'Medium', 3: 'Hard'}

for i, topic in enumerate(new_student_topics):
    confidence = np.max(pred_proba[i]) * 100
    current_diff = new_student_data.loc[i, 'Difficulty']

    # Adaptive Difficulty Logic
    next_action = ""
    if pred[i] == 2 and confidence > 80: # Strong and High Confidence
        next_diff = min(current_diff + 1, 3) # Max difficulty is 3 (Hard)
        next_action = f"Level Up! Recommended Difficulty: {difficulty_map[next_diff]}."
    elif pred[i] == 0: # Weak
        next_action = "Focus on basics."
    else: # Medium or Strong but low confidence
        next_action = f"Continue practice at current difficulty ({difficulty_map[current_diff]})."

    print(f"| {topic}: Performance={pred_text[i]} | Confidence={confidence:.1f}% | Action: {next_action}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 2: NLP-Based Dynamic Question Generator Function ---
# (Carried over from the previous update)
# ----------------------------------------------------------------------

def generate_question(topic, difficulty_level):
    """Generates a unique practice question based on topic and difficulty."""
    
    templates = {
        1: f"Easy: What is the primary function of a **{topic}** concept?",
        2: f"Medium: Solve this typical **{topic}** problem: [X]",
        3: f"Hard: Analyze and contrast two advanced principles of **{topic}** using a real-world scenario."
    }
    
    return templates.get(difficulty_level, f"Practice problem for {topic}: Review {difficulty_map[difficulty_level]}.")


# --- Step 7: Dynamic Question Recommendation ---
print("\n--- Dynamic Question Recommendation for Weak/Medium Topics ---")

for i, topic in enumerate(new_student_topics):
    current_diff = new_student_data.loc[i, 'Difficulty']
    
    if pred[i] == 0:  # Weak
        # Suggest an easier question for a weak topic
        recommended_diff = max(1, current_diff - 1)
        question = generate_question(topic, recommended_diff)
        print(f"🛑 Recommended for {topic} (Weak): Dynamic Question (Difficulty {recommended_diff}): {question}")
    elif pred[i] == 1: # Medium
        # Suggest a question at current difficulty
        question = generate_question(topic, current_diff)
        print(f"🔶 Recommended for {topic} (Medium): Dynamic Question (Difficulty {current_diff}): {question}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 3: Gamification Metrics ---
# ----------------------------------------------------------------------

print("\n--- Gamification Metrics (Simulated) ---")
student_xp = 5500
current_level = int(student_xp / 1000)
daily_streak = 7
topic_mastery = {t: df[df['Topic'] == t]['Score'].mean() for t in topics}

print(f"🌟 Student Level: {current_level} (XP: {student_xp})")
print(f"🔥 Daily Streak: {daily_streak} days!")
print("🏆 Topic Mastery Scores:")
for t, score in topic_mastery.items():
    badge = '🥇' if score >= 80 else ('🥈' if score >= 60 else '🥉')
    print(f"  - {t}: {badge} {score:.1f}")


# --- Step 8: Visualize Student Progress ---
print("\n--- Student Performance Visualization ---")
df_visual = df.pivot(index='StudentID', columns='Topic', values='Score')
plt.figure(figsize=(10,6))
sns.heatmap(df_visual, annot=True, cmap='YlGnBu', fmt="d", linewidths=.5, cbar_kws={'label': 'Score'})
plt.title("Student Performance Heatmap (80+ is Strong)")
plt.show()
# --- Step 1: Install Libraries ---
!pip install pandas numpy scikit-learn matplotlib seaborn

# --- Step 2: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 3: Simulate Student Data ---
np.random.seed(42)
students = ['S1','S2','S3','S4','S5']
topics = ['Algebra','Geometry','Physics','Chemistry','Biology']

data = []
for student in students:
    for topic in topics:
        score = np.random.randint(40,100)  # Random scores 40-100
        time_spent = np.random.randint(20,60)  # Minutes spent
        # NEW FEATURE: Include a difficulty level (1=Easy, 2=Medium, 3=Hard)
        difficulty = np.random.choice([1, 2, 3])
        data.append([student, topic, score, time_spent, difficulty])

df = pd.DataFrame(data, columns=['StudentID','Topic','Score','TimeSpent', 'Difficulty'])

# Encode topic for ML
df['Topic_encoded'] = df['Topic'].astype('category').cat.codes

# Define performance: Weak=0 (<60), Medium=1 (60-79), Strong=2 (80+)
df['Performance'] = df['Score'].apply(lambda x: 0 if x<60 else (1 if x<80 else 2))
print("Sample Data:")
print(df.head())

# --- Step 4: Split Features and Labels (Now including Difficulty) ---
X = df[['Topic_encoded','TimeSpent', 'Difficulty']] # Added 'Difficulty' as a feature
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Train Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 6: Predict Weak Topics for a New Student ---
# Simulate a new student's study time per topic and current difficulty
new_student_data = pd.DataFrame({
    'Topic_encoded': [0, 1, 2, 3, 4],  # Algebra, Geometry, Physics, Chemistry, Biology
    'TimeSpent': [25, 35, 30, 20, 40],
    'Difficulty': [2, 1, 3, 2, 1] # Current assumed difficulty level for the student
})
new_student_topics = ['Algebra', 'Geometry', 'Physics', 'Chemistry', 'Biology']

pred = model.predict(new_student_data)
# NEW FEATURE: Predict the probability distribution (confidence)
pred_proba = model.predict_proba(new_student_data)

pred_labels = ['Weak','Medium','Strong']
pred_text = [pred_labels[i] for i in pred]
print("\n--- Predicted Performance ---")
print("Predicted performance per topic:", pred_text)

# ----------------------------------------------------------------------
# --- NEW FEATURE 1: Confidence Score & Adaptive Difficulty ---
# ----------------------------------------------------------------------

print("\n--- AI Confidence & Next Steps ---")
difficulty_map = {1: 'Easy', 2: 'Medium', 3: 'Hard'}

for i, topic in enumerate(new_student_topics):
    confidence = np.max(pred_proba[i]) * 100
    current_diff = new_student_data.loc[i, 'Difficulty']

    # Adaptive Difficulty Logic
    next_action = ""
    if pred[i] == 2 and confidence > 80: # Strong and High Confidence
        next_diff = min(current_diff + 1, 3) # Max difficulty is 3 (Hard)
        next_action = f"Level Up! Recommended Difficulty: {difficulty_map[next_diff]}."
    elif pred[i] == 0: # Weak
        next_action = "Focus on basics."
    else: # Medium or Strong but low confidence
        next_action = f"Continue practice at current difficulty ({difficulty_map[current_diff]})."

    print(f"| {topic}: Performance={pred_text[i]} | Confidence={confidence:.1f}% | Action: {next_action}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 2: NLP-Based Dynamic Question Generator Function ---
# (Carried over from the previous update)
# ----------------------------------------------------------------------

def generate_question(topic, difficulty_level):
    """Generates a unique practice question based on topic and difficulty."""
    
    templates = {
        1: f"Easy: What is the primary function of a **{topic}** concept?",
        2: f"Medium: Solve this typical **{topic}** problem: [X]",
        3: f"Hard: Analyze and contrast two advanced principles of **{topic}** using a real-world scenario."
    }
    
    return templates.get(difficulty_level, f"Practice problem for {topic}: Review {difficulty_map[difficulty_level]}.")


# --- Step 7: Dynamic Question Recommendation ---
print("\n--- Dynamic Question Recommendation for Weak/Medium Topics ---")

for i, topic in enumerate(new_student_topics):
    current_diff = new_student_data.loc[i, 'Difficulty']
    
    if pred[i] == 0:  # Weak
        # Suggest an easier question for a weak topic
        recommended_diff = max(1, current_diff - 1)
        question = generate_question(topic, recommended_diff)
        print(f"🛑 Recommended for {topic} (Weak): Dynamic Question (Difficulty {recommended_diff}): {question}")
    elif pred[i] == 1: # Medium
        # Suggest a question at current difficulty
        question = generate_question(topic, current_diff)
        print(f"🔶 Recommended for {topic} (Medium): Dynamic Question (Difficulty {current_diff}): {question}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 3: Gamification Metrics ---
# ----------------------------------------------------------------------

print("\n--- Gamification Metrics (Simulated) ---")
student_xp = 5500
current_level = int(student_xp / 1000)
daily_streak = 7
topic_mastery = {t: df[df['Topic'] == t]['Score'].mean() for t in topics}

print(f"🌟 Student Level: {current_level} (XP: {student_xp})")
print(f"🔥 Daily Streak: {daily_streak} days!")
print("🏆 Topic Mastery Scores:")
for t, score in topic_mastery.items():
    badge = '🥇' if score >= 80 else ('🥈' if score >= 60 else '🥉')
    print(f"  - {t}: {badge} {score:.1f}")


# --- Step 8: Visualize Student Progress ---
print("\n--- Student Performance Visualization ---")
df_visual = df.pivot(index='StudentID', columns='Topic', values='Score')
plt.figure(figsize=(10,6))
sns.heatmap(df_visual, annot=True, cmap='YlGnBu', fmt="d", linewidths=.5, cbar_kws={'label': 'Score'})
plt.title("Student Performance Heatmap (80+ is Strong)")
plt.show()
# --- Step 1: Install Libraries ---
!pip install pandas numpy scikit-learn matplotlib seaborn

# --- Step 2: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 3: Simulate Student Data ---
np.random.seed(42)
students = ['S1','S2','S3','S4','S5']
topics = ['Algebra','Geometry','Physics','Chemistry','Biology']

data = []
for student in students:
    for topic in topics:
        score = np.random.randint(40,100)  # Random scores 40-100
        time_spent = np.random.randint(20,60)  # Minutes spent
        # NEW FEATURE: Include a difficulty level (1=Easy, 2=Medium, 3=Hard)
        difficulty = np.random.choice([1, 2, 3])
        data.append([student, topic, score, time_spent, difficulty])

df = pd.DataFrame(data, columns=['StudentID','Topic','Score','TimeSpent', 'Difficulty'])

# Encode topic for ML
df['Topic_encoded'] = df['Topic'].astype('category').cat.codes

# Define performance: Weak=0 (<60), Medium=1 (60-79), Strong=2 (80+)
df['Performance'] = df['Score'].apply(lambda x: 0 if x<60 else (1 if x<80 else 2))
print("Sample Data:")
print(df.head())

# --- Step 4: Split Features and Labels (Now including Difficulty) ---
X = df[['Topic_encoded','TimeSpent', 'Difficulty']] # Added 'Difficulty' as a feature
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Train Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 6: Predict Weak Topics for a New Student ---
# Simulate a new student's study time per topic and current difficulty
new_student_data = pd.DataFrame({
    'Topic_encoded': [0, 1, 2, 3, 4],  # Algebra, Geometry, Physics, Chemistry, Biology
    'TimeSpent': [25, 35, 30, 20, 40],
    'Difficulty': [2, 1, 3, 2, 1] # Current assumed difficulty level for the student
})
new_student_topics = ['Algebra', 'Geometry', 'Physics', 'Chemistry', 'Biology']

pred = model.predict(new_student_data)
# NEW FEATURE: Predict the probability distribution (confidence)
pred_proba = model.predict_proba(new_student_data)

pred_labels = ['Weak','Medium','Strong']
pred_text = [pred_labels[i] for i in pred]
print("\n--- Predicted Performance ---")
print("Predicted performance per topic:", pred_text)

# ----------------------------------------------------------------------
# --- NEW FEATURE 1: Confidence Score & Adaptive Difficulty ---
# ----------------------------------------------------------------------

print("\n--- AI Confidence & Next Steps ---")
difficulty_map = {1: 'Easy', 2: 'Medium', 3: 'Hard'}

for i, topic in enumerate(new_student_topics):
    confidence = np.max(pred_proba[i]) * 100
    current_diff = new_student_data.loc[i, 'Difficulty']

    # Adaptive Difficulty Logic
    next_action = ""
    if pred[i] == 2 and confidence > 80: # Strong and High Confidence
        next_diff = min(current_diff + 1, 3) # Max difficulty is 3 (Hard)
        next_action = f"Level Up! Recommended Difficulty: {difficulty_map[next_diff]}."
    elif pred[i] == 0: # Weak
        next_action = "Focus on basics."
    else: # Medium or Strong but low confidence
        next_action = f"Continue practice at current difficulty ({difficulty_map[current_diff]})."

    print(f"| {topic}: Performance={pred_text[i]} | Confidence={confidence:.1f}% | Action: {next_action}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 2: NLP-Based Dynamic Question Generator Function ---
# (Carried over from the previous update)
# ----------------------------------------------------------------------

def generate_question(topic, difficulty_level):
    """Generates a unique practice question based on topic and difficulty."""
    
    templates = {
        1: f"Easy: What is the primary function of a **{topic}** concept?",
        2: f"Medium: Solve this typical **{topic}** problem: [X]",
        3: f"Hard: Analyze and contrast two advanced principles of **{topic}** using a real-world scenario."
    }
    
    return templates.get(difficulty_level, f"Practice problem for {topic}: Review {difficulty_map[difficulty_level]}.")


# --- Step 7: Dynamic Question Recommendation ---
print("\n--- Dynamic Question Recommendation for Weak/Medium Topics ---")

for i, topic in enumerate(new_student_topics):
    current_diff = new_student_data.loc[i, 'Difficulty']
    
    if pred[i] == 0:  # Weak
        # Suggest an easier question for a weak topic
        recommended_diff = max(1, current_diff - 1)
        question = generate_question(topic, recommended_diff)
        print(f"🛑 Recommended for {topic} (Weak): Dynamic Question (Difficulty {recommended_diff}): {question}")
    elif pred[i] == 1: # Medium
        # Suggest a question at current difficulty
        question = generate_question(topic, current_diff)
        print(f"🔶 Recommended for {topic} (Medium): Dynamic Question (Difficulty {current_diff}): {question}")


# ----------------------------------------------------------------------
# --- NEW FEATURE 3: Gamification Metrics ---
# ----------------------------------------------------------------------

print("\n--- Gamification Metrics (Simulated) ---")
student_xp = 5500
current_level = int(student_xp / 1000)
daily_streak = 7
topic_mastery = {t: df[df['Topic'] == t]['Score'].mean() for t in topics}

print(f"🌟 Student Level: {current_level} (XP: {student_xp})")
print(f"🔥 Daily Streak: {daily_streak} days!")
print("🏆 Topic Mastery Scores:")
for t, score in topic_mastery.items():
    badge = '🥇' if score >= 80 else ('🥈' if score >= 60 else '🥉')
    print(f"  - {t}: {badge} {score:.1f}")


# --- Step 8: Visualize Student Progress ---
print("\n--- Student Performance Visualization ---")
df_visual = df.pivot(index='StudentID', columns='Topic', values='Score')
plt.figure(figsize=(10,6))
sns.heatmap(df_visual, annot=True, cmap='YlGnBu', fmt="d", linewidths=.5, cbar_kws={'label': 'Score'})
plt.title("Student Performance Heatmap (80+ is Strong)")
plt.show()
# EDUMIND
