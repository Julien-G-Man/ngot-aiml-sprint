import sys
import os

class Tee:
    """Write to both console and file simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()
    
    # Provide isatty and fileno so libraries (uvicorn logging) can query terminal capabilities
    def isatty(self):
        try:
            return getattr(self.files[0], "isatty", lambda: False)()
        except Exception:
            return False

    def fileno(self):
        try:
            return getattr(self.files[0], "fileno")()
        except Exception:
            return 1
    
    def writelines(self, lines):
        for f in self.files:
            f.writelines(lines)

    @property
    def encoding(self):
        return getattr(self.files[0], 'encoding', 'utf-8')

    @property
    def errors(self):
        return getattr(self.files[0], 'errors', None)

    @property
    def buffer(self):
        return getattr(self.files[0], 'buffer', None)

if __name__ == "__main__":
    # Ensure logs directory exists and log to logs/server_logs.txt
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "server_logs.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    tee_stdout = Tee(sys.__stdout__, log_file)
    tee_stderr = Tee(sys.__stderr__, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    # Ensure logging module also writes to the same file
    import logging

    root_logger = logging.getLogger()
    abs_log_path = os.path.abspath(log_path)
    has_file = any(isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_path for h in root_logger.handlers)
    if not has_file:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    # Import uvicorn after stdout/stderr and logging are configured so
    # uvicorn's initialization picks up the correct streams/terminal state.
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
    )
