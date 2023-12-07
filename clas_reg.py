import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

df = pd.read_excel("/Users/melihasecilozturk/Desktop/miuul/ödevler/classification_reg/clas_denem.xlsx")
df.head()

# - Eşik değerini 0.5 alarak confusion matrix oluşturunuz.
# - Accuracy,Recall,Precision,F1Skorlarını hesaplayınız.


# - Eşik değerini 0.5 alarak yeni tahminler yap

df["y_pred"] = df["1_sınıfta_olma_olasılık"].apply(lambda x: 0 if x < 0.5 else 1)

df

# - Accuracy,Recall,Precision,F1 Skorlarını hesaplayınız.
y = df["gercek_deger"]
y_pred = df["y_pred"]

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))
#              precision    recall  f1-score   support
#           1       0.83      0.83      0.83         6
#    accuracy                           0.80        10

