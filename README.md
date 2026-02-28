# 🩺 Clasificador de Diabetes con Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-MLPClassifier-orange?logo=scikit-learn)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?logo=huggingface)

Proyecto de Machine Learning que predice si una persona tiene diabetes a partir de parámetros clínicos. Utiliza un pipeline completo con control de versiones de datos (DVC), integración continua (CI/CD) con GitHub Actions, y despliegue automático en Hugging Face Spaces.

## 🔗 Links del Proyecto

| Recurso | Link |
|---|---|
| 📦 Repositorio GitHub | [ClasificadorDiabetesML](https://github.com/alecorlo1314/ClasificadorDiabetesML) |
| 🗄️ Repositorio DagsHub | [alecorlo1234/ClasificadorDiabetesML](https://dagshub.com/alecorlo1234/ClasificadorDiabetesML) |
| 🤗 Aplicación en Hugging Face | [ClasificadorDiabetesML Space](https://huggingface.co/spaces/alecorlo1234/ClasificadorDiabetesML) |
| 📊 Dataset (Kaggle) | [Diabetes Risk Prediction Dataset](https://www.kaggle.com/datasets/vishardmehta/diabetes-risk-prediction-dataset) |

---

## 📋 Tabla de Contenidos

- [Descripción del Problema](#-descripción-del-problema)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Estructura de Carpetas](#-estructura-de-carpetas)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Requisitos Previos](#-requisitos-previos)
- [Configuración del Entorno](#-configuración-del-entorno)
- [Configuración de Secretos](#-configuración-de-secretos)
- [Uso con Makefile](#-uso-con-makefile)
- [Pipeline CI/CD](#-pipeline-cicd)
- [Métricas del Modelo](#-métricas-del-modelo)
- [Aplicación Gradio](#-aplicación-gradio)

---

## 🎯 Descripción del Problema

El dataset de entrenamiento presenta un **desbalance de clases**: la mayoría de los registros corresponden a personas no diabéticas, mientras que los casos diabéticos son la minoría. Para abordar esto se utilizó `imbalanced-learn` y se priorizó la métrica **F1-Score** durante la evaluación, ya que esta penaliza tanto los falsos positivos como los falsos negativos, siendo más representativa en escenarios desbalanceados.

---

## 🏗️ Arquitectura del Proyecto

```
Dataset (Kaggle)
      ↓
DVC + DagsHub  →  Control de versiones de datos
      ↓
Entrenamiento  →  MLPClassifier + Pipeline de sklearn
      ↓
Evaluación     →  Métricas con reporte en Pull Request (CML)
      ↓
CI/CD          →  GitHub Actions (CI → CD)
      ↓
Despliegue     →  Hugging Face Spaces (Gradio)
```

---

## 📁 Estructura de Carpetas

```
└── 📁ClasificadorDiabetesML
    └── 📁.github
        └── 📁workflows
            ├── ci.yml          # Integración continua
            └── cd.yml          # Despliegue continuo
    └── 📁Aplicacion
        ├── diabetes_app.py     # Interfaz Gradio
        ├── README.md           # Configuración del Space en HF
        └── requirements.txt    # Dependencias del Space
    └── 📁Datos
        └── *.dvc               # Referencia al dataset versionado con DVC
    └── 📁Modelo
        └── pipeline.skops      # Modelo entrenado serializado
    └── 📁src
        ├── datos.py            # Carga y preprocesamiento
        ├── entrenar.py         # Construcción del pipeline y entrenamiento
        ├── evaluar.py          # Cálculo de métricas
        ├── guardar.py          # Serialización del modelo
        └── predecir.py         # Inferencia
    ├── entrenamiento.py        # Script principal de entrenamiento
    ├── Makefile                # Comandos del proyecto
    ├── notebook.ipynb          # Exploración y experimentación
    └── requirements.txt        # Dependencias del proyecto
```

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.11 | Lenguaje principal |
| Scikit-learn | MLPClassifier y pipeline de ML |
| imbalanced-learn | Manejo del desbalance de clases |
| skops | Serialización segura del modelo |
| pandas / numpy | Manipulación de datos |
| DVC | Control de versiones del dataset |
| DagsHub | Almacenamiento remoto de DVC |
| Gradio | Interfaz web de la aplicación |
| GitHub Actions | CI/CD automatizado |
| Hugging Face Spaces | Despliegue de la aplicación |
| CML | Reporte de métricas en Pull Requests |

---

## ✅ Requisitos Previos

Antes de comenzar asegúrate de tener instalado:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [DVC](https://dvc.org/doc/install)
- Una cuenta en [DagsHub](https://dagshub.com/)
- Una cuenta en [Hugging Face](https://huggingface.co/)
- Una cuenta en [GitHub](https://github.com/)

---

## ⚙️ Configuración del Entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/alecorlo1314/ClasificadorDiabetesML.git
cd ClasificadorDiabetesML
```

### 2. Crear y activar el entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
make install
```

### 4. Configurar DVC con DagsHub

```bash
dvc remote add -f diabetes_storage https://dagshub.com/TU_USUARIO/TU_REPO.dvc
dvc remote default diabetes_storage
dvc remote modify diabetes_storage auth basic
dvc remote modify diabetes_storage user TU_USUARIO_DAGSHUB
dvc remote modify diabetes_storage password TU_TOKEN_DAGSHUB
```

> 💡 Puedes obtener tu token en [dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)

### 5. Descargar los datos

```bash
dvc pull -r diabetes_storage
```

### 6. Entrenar el modelo

```bash
make train
```

---

## 🔐 Configuración de Secretos

Para que el CI/CD funcione correctamente debes configurar los siguientes secretos en tu repositorio de GitHub en **Settings → Secrets and variables → Actions**:

| Secreto | Descripción | Dónde obtenerlo |
|---|---|---|
| `DAGSHUB_TOKEN` | Token de acceso a DagsHub | [dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens) |
| `HF_DIABETES` | Token de acceso a Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

> ⚠️ El secreto `GITHUB_TOKEN` es generado automáticamente por GitHub Actions, no necesitas crearlo manualmente.

---

## 🧰 Uso con Makefile

El proyecto incluye un `Makefile` con los comandos principales:

| Comando | Descripción |
|---|---|
| `make install` | Instala todas las dependencias |
| `make format` | Formatea el código con Black |
| `make lint` | Analiza la calidad del código |
| `make train` | Entrena el modelo |
| `make eval` | Evalúa el modelo y genera reporte |
| `make deploy HF=<token>` | Despliega la app en Hugging Face |

---

## 🔄 Pipeline CI/CD

El proyecto tiene dos workflows automatizados:

### Integración Continua (`ci.yml`)
Se dispara en cada push o pull request a `main`:

```
Checkout → Instalar dependencias → Formatear código → Analizar código
    → Configurar DVC → Descargar datos → Entrenar modelo → Evaluar modelo
```

Al finalizar, CML publica automáticamente un reporte con las métricas del modelo como comentario en el Pull Request.

### Despliegue Continuo (`cd.yml`)
Se dispara automáticamente cuando el CI termina exitosamente:

```
Checkout → Login en Hugging Face → Subir Aplicacion → Subir Modelo
```

---

## 📊 Métricas del Modelo

El modelo fue evaluado priorizando el **F1-Score** debido al desbalance de clases en el dataset. Las métricas reportadas son:

| Métrica | Descripción |
|---|---|
| **F1-Score** | Métrica principal — balance entre precisión y recall |
| **Accuracy** | Porcentaje de predicciones correctas |
| **Precision** | De los predichos como diabéticos, cuántos realmente lo son |
| **Recall** | De los diabéticos reales, cuántos fueron detectados |

> Los reportes detallados de cada entrenamiento se generan automáticamente en los Pull Requests gracias a CML.

---

## 🖥️ Aplicación Gradio

La aplicación permite ingresar los siguientes parámetros clínicos para obtener una predicción:

| Parámetro | Tipo | Descripción |
|---|---|---|
| Género | Dropdown | Male, Female, Other |
| Edad | Número | Edad en años |
| Hipertensión | Radio | 0 = No, 1 = Sí |
| Enfermedad Cardíaca | Radio | 0 = No, 1 = Sí |
| Historial de Tabaco | Dropdown | never, former, current, etc. |
| IMC | Número | Índice de masa corporal |
| HbA1c Level | Número | Nivel de hemoglobina glicosilada |
| Blood Glucose Level | Número | Nivel de glucosa en sangre |

Puedes probar la aplicación en vivo en: [https://huggingface.co/spaces/alecorlo1234/ClasificadorDiabetesML](https://huggingface.co/spaces/alecorlo1234/ClasificadorDiabetesML)

---

## 📄 Licencia

Este proyecto es de uso educativo y libre. Si lo usas o adaptas, se agradece dar crédito al autor original.
