run_preprocess:
	python -c "from titanic.api.main import preprocess_data; preprocess_data()"

run_train:
	python -c 'from titanic.api.main import train_rfc; train_rfc()'

run_pred:
	python -c 'from titanic.api.main import pred; pred()'

run_api:
	uvicorn titanic.api.fast:app --reload
