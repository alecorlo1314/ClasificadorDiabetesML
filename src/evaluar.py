from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report


def evaluar_modelo(modelo, X_test, y_test):
    
    try:
        predictions = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        recall = recall_score(y_test, predictions, average="macro")
        precision = precision_score(y_test, predictions, average="macro")
    except Exception as e:
        print("Error al evaluar el modelo:", e)
        return None, None, None, None, None, None
    return accuracy, f1, recall, precision, predictions, y_proba
