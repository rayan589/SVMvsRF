import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import time  # Import the time module

# Importing dataset
df = pd.read_csv('/Users/macbookpro/Documents/umich/CIS405/Project/diabetes_prediction_dataset.csv')

df['is_male'] = (df['gender'] == 'Male').astype(int)
df = df.drop(columns=['gender'])
df['is_smoker'] = df['smoking_history'].isin(['current', 'former']).astype(int)
df = df.drop(columns=['smoking_history'])
X = df.drop('diabetes', axis=1)
y = df['diabetes']

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaling_x = StandardScaler()
X_train = scaling_x.fit_transform(X_train)
X_test = scaling_x.transform(X_test)

rfc = RandomForestClassifier()

# Start the timer
start_time = time.time()

# Train the model
rfc.fit(X_train, y_train)

# End the timer
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")

y_pred = rfc.predict(X_test)

print("Accuracy: ", rfc.score(X_test, y_test))

target_names = ['Diabetes', 'Normal']
print(classification_report(y_test, y_pred, target_names=target_names))

y_pred_proba = rfc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr, label='RF')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()

mat = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(7, 5))
sns.heatmap(mat, annot=True, fmt=".2%", cmap="Blues")
plt.show()
