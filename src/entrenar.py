from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


def construir_pipeline():

    preprocesamiento = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [1, 5, 6]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [0, 4])
        ],
        remainder="passthrough"
    )

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocesamiento),
            ("smote", SMOTE(random_state=42)),
            ("modelo", MLPClassifier(max_iter=1000, 
                                     early_stopping=True,
                                     random_state=42, 
                                     activation="logistic", 
                                     hidden_layer_sizes=(100,),
                                     learning_rate = "constant",
                                     learning_rate_init=0.001))
        ]
    )

    return pipe