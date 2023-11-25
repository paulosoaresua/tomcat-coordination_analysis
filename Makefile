lint:
	black .
	isort .
	autoflake -r coordination --in-place --remove-all-unused-imports
	flake8 coordination

vocalic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/data_mapping.json" \
		--model_params_dict_filepath="data/config/params_dict.json"
