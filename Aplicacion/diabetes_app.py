import gradio as gr
from skops.io import get_untrusted_types, load

unsafe = get_untrusted_types(file="Modelo/pipeline.skops")
print("Tipos no confiables:", unsafe)

pipeline = load("Modelo/pipeline.skops", trusted=unsafe)


def prediccion(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    HbA1c_level,
    blood_glucose_level,
):
    """
    Args:
        gender (str): Género de la persona (ej. 'Male', 'Female').
        age (int): Edad en años.
        hypertension (int): 1 si la persona tiene hipertensión, 0 si no.
        heart_disease (int): 1 si la persona tiene enfermedad cardíaca, 0 si no.
        smoking_history (str): Historial de tabaquismo (ej. 'never', 'former', 'current').
        bmi (float): Índice de masa corporal.
        HbA1c_level (float): Nivel de hemoglobina glicosilada.
        blood_glucose_level (float): Nivel de glucosa en sangre.

    Returns:
        int: 1 si el modelo predice que la persona es diabética, 0 si no lo es.
    """

    caracteristicas = [
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_history,
        bmi,
        HbA1c_level,
        blood_glucose_level,
    ]
    diabetes_predicha = pipeline.predict([caracteristicas])[0]

    label = f"La persona es {'diabética' if diabetes_predicha == 1 else 'no diabética'}"
    return label


entradas = [
    gr.Dropdown(choices=["Male", "Female", "Other"], label="Genero"),
    gr.Number(label="Edad"),
    gr.Radio(choices=[0, 1], label="Hipertension"),  # 1 = Sí, 0 = No
    gr.Radio(choices=[0, 1], label="Enfermedad Cardíaca"),  # 1 = Sí, 0 = No
    gr.Dropdown(
        choices=["never", "former", "current", "not current", "ever", "No Info"],
        label="Historial de Tabaco",
    ),  # never = Nunca, former = Exfumador, current = Fumador actual
    gr.Number(label="IMC"),
    gr.Number(label="HbA1c Level"),
    gr.Number(label="Blood Glucose Level"),
]
salida = [gr.Label(num_top_classes=8, label="Predicción de Diabetes")]


# Ejemplos de prueba
ejemplos = [
    ["Male", 45, 1, 0, "former", 28.5, 6.2, 150],
    ["Female", 30, 0, 0, "never", 22.1, 5.4, 95],
    ["Male", 60, 1, 1, "current", 31.0, 7.8, 200],
]

# Título y descripción
titulo = "Clasificador de Diabetes con ML"
descripcion = "Recibe parámetros clínicos y predice si un paciente es diabético (1) o no (0)."
articulo = "Proyecto de Machine Learning con DVC, GitHub y DagsHub."

# Interfaz
demo = gr.Interface(
    fn=prediccion,  # tu función de predicción
    inputs=entradas,
    outputs=salida,
    examples=ejemplos,
    title=titulo,
    description=descripcion,
    article=articulo,
)

demo.launch(theme=gr.themes.Soft())

"""
Abrir la terminal y ejecutar el siguiente comando para iniciar la aplicación:
python -m Aplicacion.diabetes_app
"""
