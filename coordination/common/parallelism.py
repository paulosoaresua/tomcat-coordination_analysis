from os import environ

# Global variables that control program behaviour when running in multithread or multiprocess


def display_inner_progress_bar():
    # Enables it if the environment variable INNER_PB to be set to 1
    return environ.get('INNER_PB', '0') == '1'

