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
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="T000672"

# To run inference on the ASIST dataset with the Link model
vocalic_semantic_asist:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="T000672"

# To run inference on the ToMCAT dataset with the Voc model
vocalic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="d141cab6-2940-47ac-8aaa-e0c3a425e31c"

# To run inference on the ToMCAT dataset with the Link model
vocalic_semantic_tomcat:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="d141cab6-2940-47ac-8aaa-e0c3a425e31c"

######################################################################################
######################################################################################
######################################################################################

# RMSE on the ASIST dataset with the Voc model
vocalic_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="T000672" --num_time_points_ppa=1

# RMSE on the ASIST dataset with the Link model
vocalic_semantic_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="T000672" --num_time_points_ppa=1

# RMSE on ASIST data with the x-model
vocalic_no_coordination_asist_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/asist_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="T000672" --num_time_points_ppa=1

# RMSE on the ToMCAT dataset with the Voc model
vocalic_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10-- experiment_ids="d141cab6-2940-47ac-8aaa-e0c3a425e31c" --num_time_points_ppa=1

# RMSE on the ToMCAT dataset with the Link model
vocalic_semantic_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic_semantic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="d141cab6-2940-47ac-8aaa-e0c3a425e31c" --num_time_points_ppa=1

# RMSE on ToMCAT data with the x-model
vocalic_no_coordination_tomcat_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/tomcat_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="d141cab6-2940-47ac-8aaa-e0c3a425e31c" --num_time_points_ppa=1

######################################################################################
######################################################################################
######################################################################################

# Generate x-data dn c-data with x-model and c-models
generate_ppa_synthetic_data:
	PYTHONPATH="." ./bin/generate_ppa_synthetic_data

x_model_x_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/synthetic/x_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/synthetic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="exp0" --num_time_points_ppa=1

x_model_c_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/synthetic/x_data_low.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/synthetic/config/params/vocalic_no_coordination_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="exp0" --num_time_points_ppa=1

c_model_x_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/synthetic/x_data.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/synthetic/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="exp0" --num_time_points_ppa=1

c_model_c_data_rmse:
	PYTHONPATH="." ./bin/run_inference --evidence_filepath="data/synthetic/x_data_low.csv" \
		--model_name="vocalic" \
		--data_mapping_filepath="data/synthetic/config/mappings/vocalic_data_mapping.json" \
		--model_params_dict_filepath="data/synthetic/config/params/vocalic_params_dict.json" \
		--do_ppa=1 \
		--burn_in=10 --num_samples=10 --num_chains=10 --experiment_ids="exp0" --num_time_points_ppa=1