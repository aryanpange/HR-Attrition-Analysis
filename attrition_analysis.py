# 📦 Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📥 Step 2: Load dataset
df = pd.read_csv("HR_Employee_Attrition.csv")
print("✅ Dataset loaded successfully.\n")
print(df.head())

# 📊 Step 3: Dataset overview
print("\n🧾 Dataset Info:")
print(df.info())

print("\n📊 Summary Statistics:")
print(df.describe())

print("\n🔍 Missing Values:")
print(df.isnull().sum())

# 📈 Step 4: Exploratory Data Visualization
sns.countplot(x='Attrition', data=df, palette='Set2')
plt.title("Employee Attrition Count")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, bins=30)
plt.title("Age Distribution by Attrition")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income by Attrition")
plt.show()

sns.countplot(x='OverTime', hue='Attrition', data=df, palette='coolwarm')
plt.title("Attrition based on Overtime")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='Department', hue='Attrition', data=df, palette='pastel')
plt.title("Attrition across Departments")
plt.xticks(rotation=20)
plt.show()

# 🧼 Step 5: Preprocessing
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})  # Encode target
df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)  # Drop constants

# 🎯 Step 6: Feature matrix & target variable
X = pd.get_dummies(df.drop('Attrition', axis=1), drop_first=True)
y = df['Attrition']

# 📏 Step 7: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Shape of features after encoding and scaling:", X_scaled.shape)
print("✅ Shape of target:", y.shape)

# 🔀 Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🤖 Step 9: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print(f"\n✅ Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.2f}")
print("🔍 Classification Report (LogReg):\n", classification_report(y_test, log_pred))

# 📉 Confusion Matrix - Logistic Regression
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 🌲 Step 10: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"\n🌲 Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print("🔍 Classification Report (Random Forest):\n", classification_report(y_test, rf_pred))

# 📉 Confusion Matrix - Random Forest
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 📌 Step 11: Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.head(10), y=feature_importance.head(10).index, palette='viridis')
plt.title("🔥 Top 10 Features Driving Attrition")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Combine X_test (as DataFrame) with predictions and true labels
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['Actual_Attrition'] = y_test.values
X_test_df['Predicted_Attrition'] = rf_pred
