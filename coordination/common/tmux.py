import os
import time


class TMUX:
    """
    This class handles tmux sessions.
    """

    def __init__(self, session_name: str):
        """
        Creates a TMUX object.

        @param session_name: name of the tmux session.
        """
        self.session_name = session_name

    def create_window(self, window_name: str):
        """
        Creates a new window in a tmux session. If the session does not exist, it will be created
        along with the window.

        @param window_name: name of the window.
        """
        if TMUX.has_session(self.session_name):
            os.system(f"tmux new-window -n {window_name}")
        else:
            os.system(f"tmux new-session -d -s {self.session_name} -n {window_name}")

    def kill_session(self):
        """
        Kills a tmux session.
        """
        os.system(f"tmux kill-session -t {self.session_name}")

    def run_command(self, command: str):
        """
        Runs a command in a tmux session. This sends letter by letter of the command to the tmux
        session, so we need to wait a bit before for the command to be fully delivered and
        executed.

        @param command: command.
        """
        os.system(f'tmux send-keys -t {self.session_name} "{command}" "C-m"')
        time.sleep(2)

    @staticmethod
    def has_session(session_name: str):
        """
        Checks if a tmux session with the same name already exists.

        @param session_name: name of the session.
        @return:
        """
        return os.system(f"tmux has-session -t {session_name} 2>/dev/null") == 0
