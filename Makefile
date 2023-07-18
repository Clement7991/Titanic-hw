run_preprocess:
	python -c "from titanic.interface.main import preprocess_data; preprocess_data()"

run_train:
	python -c 'from titanic.interface.main import train_rfc; train_rfc()'

run_pred:
	python -c 'from titanic.interface.main import predict; predict()'

run_all: run_preprocess run_train run_pred

run_api:
	uvicorn titanic.api.fast:app --reload
