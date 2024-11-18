import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

veri = load_breast_cancer()
X = pd.DataFrame(veri.data, columns = veri.feature_names)
y = pd.Series(veri.target)

print("Veri Özellikleri:\n", X.head())
print("\nHedef Değişken Dağılımı:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

dogruluk = accuracy_score(y_test, y_pred)
print("Model Doğruluk Oranı:", dogruluk)
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation="nearest", cmap = plt.cm.Blues)
plt.title("Karışıklık Matrisi")
plt.colorbar()
plt.xticks([0,1], ["Benign", "Malignant"])
plt.yticks([0,1], ["Benign", "Malignant"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment = "center", color = "white" if cm[i, j] > cm.max()/ 2 else "black")
plt.show()
