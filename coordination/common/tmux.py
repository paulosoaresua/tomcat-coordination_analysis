import os
import time


class TMUX:

    def __init__(self, session_name: str):
        self.session_name = session_name

    def create_window(self, window_name: str):
        if TMUX.has_session(self.session_name):
            os.system(f"tmux new-window -n {window_name}")
        else:
            os.system(f"tmux new-session -d -s {self.session_name} -n {window_name}")

    def kill_session(self):
        os.system(f"tmux kill-session -t {self.session_name}")

    def run_command(self, command: str):
        os.system(f'tmux send-keys -t {self.session_name} "{command}" "C-m"')
        time.sleep(2)

    @staticmethod
    def has_session(session_name: str):
        return os.system(f"tmux has-session -t {session_name} 2>/dev/null") == 0
