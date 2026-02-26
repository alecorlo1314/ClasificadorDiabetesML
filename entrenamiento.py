from src.guardar import guardar_matriz_confusion, guardar_metricas, guardar_modelo
from src.datos import cargar_datos
from src.entrenar import construir_pipeline
from src.evaluar import evaluar_modelo
from sklearn.model_selection import train_test_split


def main():
    print("Cargando datos...")
    diabetes_df = cargar_datos("Datos/clasificador_diabetes_ml_csv_v2.0.0__diabetes_clasificacion_prediction_dataset.csv")

    print("Dividiendo datos...")
    X = diabetes_df.drop("diabetes", axis=1).values
    y = diabetes_df.diabetes.values

    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Construyendo pipeline...")
    pipe = construir_pipeline()
    print("Entrenando el modelo...")
    pipe.fit(X_train, y_train)

    print("Evaluando el modelo...")
    accuracy, f1, recall, precision, predictions, y_proba = evaluar_modelo(pipe, X_test, y_test)

    print("Resultados de la evaluación")
    print("Accuracy:", str(round(accuracy, 2) * 100) + "%", 
          "F1:", round(f1, 2), 
          "Recall:", round(recall, 2), 
          "Precision:", round(precision, 2))
    
    print("Guardando resultados...")
    guardar_matriz_confusion(pipe, X_test, y_test)
    guardar_metricas(y_test, predictions, y_proba, path="./Resultados/metricas.txt")
    guardar_modelo(pipe)

if __name__ == "__main__":
    main()