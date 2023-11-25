lint:
	black .
	isort .
	autoflake -r coordination --in-place --remove-all-unused-imports
	flake8 coordination

app:
	PYTHONPATH="." streamlit run coordination/webapp/app.py --server.port=8080 \
		--server.address=0.0.0.0

vocalic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/data_mapping.json" \
		--model_params_dict_filepath="data/config/params_dict.json"
