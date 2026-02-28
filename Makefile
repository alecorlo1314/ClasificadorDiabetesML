install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black .

configuracion_DVC_remoto:
	dvc remote add -f diabetes_storage https://dagshub.com/alecorlo1234/ClasificadorDiabetesML.dvc
	dvc remote default diabetes_storage
	dvc remote modify diabetes_storage auth basic
	dvc remote modify diabetes_storage user alecorlo1234

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

	echo '\n## Curva ROC' >> reporte.md
	echo '![Curva ROC](./Resultados/roc_curve.png)' >> reporte.md

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
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	hf upload alecorlo1234/ClasificadorDiabetesML ./Aplicacion --repo-type=space --commit-message="Sincronizar archivos de Aplicacion"
	hf upload alecorlo1234/ClasificadorDiabetesML ./Modelo /Modelo --repo-type=space --commit-message="Sincronizar archivos del modelo"
	
deploy: hf-login push-hub