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

# Regular
vocalic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json"

vocalic_semantic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json"

vocalic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json"

vocalic_semantic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json"

# PPA
vocalic_asist_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1

vocalic_semantic_asist_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1

vocalic_tomcat_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1

vocalic_semantic_tomcat_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1

# PPA - No coordination
vocalic_no_coordination_asist_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

vocalic_semantic_no_coordination_asist_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

vocalic_no_coordination_tomcat_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

vocalic_semantic_no_coordination_tomcat_ppa:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_semantic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1
