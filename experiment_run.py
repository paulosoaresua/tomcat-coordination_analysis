# List of run_id values
run_ids = [
    "2024.11.14--09.27.27", "2024.11.14--09.30.23", "2024.11.14--09.33.28",
    "2024.11.09--14.42.46", "2024.11.09--14.43.02", "2024.11.13--13.23.32",
    "2024.11.13--13.21.13", "2024.11.14--09.28.29", "2024.11.14--09.30.31",
    "2024.11.14--09.33.40", "2024.11.09--14.43.15", "2024.11.09--14.43.24",
    "2024.11.13--13.24.05", "2024.11.13--13.21.27", "2024.11.14--09.28.50",
    "2024.11.14--09.30.36", "2024.11.14--09.33.47", "2024.11.09--14.43.31",
    "2024.11.09--14.43.40", "2024.11.13--13.24.40", "2024.11.13--13.21.41"
]

# Base command template
base_command = (
    'inferences_dir=/space/mlyang721/coordination/inferences '
    'evaluations_dir=/space/mlyang721/coordination/evaluations '
    'data_dir=data/brain/synthetic PYTHONPATH="." python coordination/evaluation/ppa.py --run_id={run_id}'
)

# Loop through each run_id and print the corresponding command
for run_id in run_ids:
    command = base_command.format(run_id=run_id)
    print(command)
    import os
    os.system(command)


