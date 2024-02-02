lint:
	black .
	isort .
	autoflake -r coordination --in-place --remove-all-unused-imports
	flake8 coordination

# Run unit tests
test:
	pytest

# Start the webapp
app:
	PYTHONPATH="." streamlit run coordination/webapp/app.py --server.port=${APP_PORT} \
		--server.address=localhost

# To run inference on the ASIST dataset with the Voc model
vocalic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json"

# To run inference on the ASIST dataset with the Link model
vocalic_semantic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json"

# To run inference on the ToMCAT dataset with the Voc model
vocalic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json"

# To run inference on the ToMCAT dataset with the Link model
vocalic_semantic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json"

######################################################################################
######################################################################################
######################################################################################

# RMSE on the ASIST dataset with the Voc model
vocalic_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

# RMSE on the ASIST dataset with the Link model
vocalic_semantic_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

# RMSE on ASIST data with the x-model
vocalic_no_coordination_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

# RMSE on the ToMCAT dataset with the Voc model
vocalic_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

# RMSE on the ToMCAT dataset with the Link model
vocalic_semantic_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

# RMSE on ToMCAT data with the x-model
vocalic_no_coordination_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

######################################################################################
######################################################################################
######################################################################################

# Generate x-data dn c-data with x-model and c-models
generate_ppa_synthetic_data:
	PYTHONPATH="." ./bin/generate_ppa_synthetic_data

x_model_x_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/x_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

x_model_c_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/c_data_low.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

x_model_random_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/random_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1

c_model_x_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/x_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

c_model_c_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/c_data_low.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_params_dict.json" \
		--do_ppa=1

c_model_random_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/vocalic/synthetic/random_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/vocalic/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/vocalic/synthetic/config/params/vocalic_params_dict.json" \
		--do_ppa=1