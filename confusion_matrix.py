from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
table = [[3, 1, 1], [3, 18, 7], [4, 3, 10]]
y_pred = [0 for i in range(sum(table[0]))] + [1 for i in range(sum(table[1]))] + [2 for i in range(sum(table[2]))]
y_true = []
for i in range(3):
    for j in range(3):
        y_true = y_true + [j for k in range(table[i][j])]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
disp.plot()
plt.show()