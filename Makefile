lint:
	black .
	isort .
	autoflake -r coordination --in-place --remove-all-unused-imports
	flake8 coordination

test:
	pytest

app:
	PYTHONPATH="." streamlit run coordination/webapp/app.py --server.port=${APP_PORT} \
		--server.address=localhost

vocalic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/data_mapping.json" \
		--model_params_dict_filepath="data/config/params_dict.json"

vocalic_2d_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_2d" \
		--data_mapping_filepath="data/config/data_mapping.json" \
		--experiment_ids="T000612,T000613"

vocalic_2d_asist_unconcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data_unconcat.csv" \
		--model_name="vocalic_2d" \
		--data_mapping_filepath="data/config/data_mapping.json" \
		--experiment_ids="T000612,T000613"
