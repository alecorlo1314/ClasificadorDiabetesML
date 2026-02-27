import os
import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_curve,
)


def guardar_modelo(modelo, path="./Modelo/pipeline.skops"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.dump(modelo, path)


def guardar_metricas(y_true, y_pred, y_proba=None, path="./Resultados/metricas.txt"):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Guardar métricas en un archivo de texto
    with open(path, "w") as f:
        f.write("=== Metricas de Clasificacion ===\n")
        f.write(f"Precision = {round(precision, 2)}\n")
        f.write(f"Recall = {round(recall, 2)}\n")
        f.write(f"F1 Score = {round(f1, 2)}\n")
        f.write(f"Accuracy = {round(accuracy, 2)}\n\n")
        f.write("=== Matriz de Confusion ===\n")
        f.write(str(cm))
        f.write("\n")

        # Curva ROC (si se pasa y_proba)
        if y_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)  # Guardar curva ROC como imagen
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"Curva ROC (area = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Curva ROC")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(os.path.dirname(path), "roc_curve.png"))
            plt.close()


def guardar_matriz_confusion(
    modelo, X_test, y_test, path="./Resultados/matriz_confusion.png"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    predictions = modelo.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=modelo.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
    disp.plot()
    plt.savefig(path, dpi=120)
    plt.close()
