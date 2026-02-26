install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format-check:
	black --check .

pull_dvc:
	dvc pull -r diabetes_storage

train:
	python entrenamiento.py

lint:
	flake8 --ignore E203 --max-line-length 100 src Aplicacion

eval:
	test -f ./Resultados/metricas.txt
	echo "## Metricas del Modelo" > reporte.md
	cat ./Resultados/metricas.txt >> reporte.md
	echo '\n## Matriz de Confusion' >> reporte.md
	echo '![Matriz de Confusion](./Resultados/matriz_confusion.png)' >> reporte.md
	cml comment create reporte.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Actualizando los nuevos resultados"
	git push --force origin HEAD:update

hf-login:
	git fetch origin
	git switch -c update --track origin/update || git switch update
	pip install -U "huggingface_hub[cli]"
	git config --global credential.helper store
	hf auth login --token $(HF) --add-to-git-credential
	hf auth whoami

push-hub:
	hf upload alecorlo1234/ClasificadorDiabetesML ./Aplicacion/diabetes_prediction_dataset.py /diabetes_prediction_dataset.py --repo-type space --commit-message="Sincronizando diabetes_prediction_dataset.py"
	hf upload alecorlo1234/ClasificadorDiabetesML ./Modelo /Modelo --repo-type space --commit-message="Sincronizando Modelo"

deploy: hf-login push-hub