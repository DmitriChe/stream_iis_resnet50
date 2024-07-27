import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Пример данных обучения
epochs = range(15)
train_loss = [0.539, 0.488, 0.485, 0.455, 0.466, 0.448, 0.439, 0.429, 0.423, 0.438, 0.434, 0.424, 0.432, 0.401, 0.411]
val_loss = [0.434, 0.319, 0.306, 0.261, 0.288, 0.256, 0.263, 0.243, 0.242, 0.253, 0.246, 0.288, 0.243, 0.235, 0.238]
train_metric = [0.639, 0.667, 0.669, 0.681, 0.663, 0.678, 0.682, 0.693, 0.692, 0.688, 0.687, 0.695, 0.683, 0.710, 0.693]
val_metric = [0.829, 0.854, 0.869, 0.891, 0.898, 0.886, 0.868, 0.892, 0.880, 0.880, 0.893, 0.866, 0.887, 0.883, 0.899]
train_time = 600  

dataset_info = {
    'num_samples': 660,
    'class_distribution': {'Benign': 360, 'Malignant': 300}
}

confusion_matrix = np.array([[333, 27], [34, 266]])
classification_report = """
              precision    recall  f1-score   support

      Benign       0.91      0.93      0.92       360
   Malignant       0.91      0.89      0.90       300

    accuracy                           0.91       660
   macro avg       0.91      0.91      0.91       660
weighted avg       0.91      0.91      0.91       660
"""
f1_score = 0.91
accuracy = 0.9076
total_mismatches = 61

st.title('Training Information')

st.header('Training Time')
st.write(f'Total training time: {train_time / 3600:.2f} hours')

st.header('Dataset Composition')
st.write(f'Number of samples: {dataset_info["num_samples"]}')
st.write('Class distribution:')
st.bar_chart(dataset_info['class_distribution'])

st.header('Learning Curves')
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(epochs, train_loss, label='Train Loss')
ax[0].plot(epochs, val_loss, label='Validation Loss')
ax[0].set_title('Loss Curve')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(epochs, train_metric, label='Train Metric')
ax[1].plot(epochs, val_metric, label='Validation Metric')
ax[1].set_title('Metric Curve')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Metric')
ax[1].legend()

st.pyplot(fig)

st.header('F1 Score')
st.write(f'F1 Score: {f1_score:.4f}')

st.header('Confusion Matrix')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

st.header('Classification Report')
st.text(classification_report)

st.write(f'Accuracy: {accuracy:.4f}')
st.write(f'Total mismatches: {total_mismatches} out of {dataset_info["num_samples"]}')
