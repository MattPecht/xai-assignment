import numpy as num
import lime
import numpy as np
import sklearn
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from anchor import anchor_text
import spacy
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import accuracy_score

# Load
df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

# Label
df_fake['label'] = 0
df_true['label'] = 1

# Merge Data
df = pd.concat([df_true, df_fake]).reset_index(drop=True)

# SOLVE REUTERS ISSUE
df['text'] = df['text'].str.replace('Reuters', '', case=False)

# Take 1000 random rows from each category for testing
df_sampled = df.groupby('label').sample(n=1000, random_state=42)
X = df_sampled['text']
Y = df_sampled['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create/Fit Pipeline
pipeline = make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), RandomForestClassifier(n_estimators=100, n_jobs=-1))
pipeline.fit(X_train, y_train)

# Model accuracy check
y_pred = pipeline.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)} \n")

# LIME initialization
lime_explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

# Individual LIME
idx = 0
text_to_explain = X_test.iloc[idx]
lime_exp = lime_explainer.explain_instance(text_to_explain, pipeline.predict_proba, num_features=6)
print(f"LIME Result for index {idx}:")
print(lime_exp.as_list(), '\n')

# Run LIME on 50 articles to find patterns
global_feature_importance = defaultdict(float)
for i in range(50):
    text = X_test.iloc[i]
    exp = lime_explainer.explain_instance(text, pipeline.predict_proba, num_features=5)
    
    for word, weight in exp.as_list():
        global_feature_importance[word] += weight

# Sort and see the "Global" patterns discovered by LIME
sorted_features = sorted(global_feature_importance.items(), key=lambda x: x[1], reverse=True)
print("Top words pushing towards 'Real':", sorted_features[:5])
print("Top words pushing towards 'Fake':", sorted_features[-5:], '\n')
exp.save_to_file('lime_explanation.html')


# ANCHORS
nlp = spacy.load('en_core_web_sm')
anchor_explainer = anchor_text.AnchorText(nlp, ['Fake', 'Real'], use_unk_distribution=True)
anchor_exp = anchor_explainer.explain_instance(text_to_explain, pipeline.predict, threshold=0.95)
print(f"Anchor Rule: {' AND '.join(anchor_exp.names())}")
print(f"Precision: {anchor_exp.precision()}")


# Manual Coverage Calculation
anchor_words = anchor_exp.names()
matches = X_test.apply(lambda x: all(word.lower() in x.lower() for word in anchor_words))
empirical_coverage = matches.mean()
print(f"Empirical Coverage on Test Set: {empirical_coverage:.4f}")
print(f"Number of articles matching this rule: {matches.sum()} out of {len(X_test)}")