
import os
import sys
import socket
from pathlib import Path

# Ensure RoadSafety_ML is on sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app import create_app
from app.config import Config

app = create_app(Config)


def find_open_port(start_port: int = 5000, max_attempts: int = 20) -> int:
    """Finds an open port starting from start_port to prevent address conflicts."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    default_port = int(os.environ.get("PORT", 5000))
    port = find_open_port(default_port)
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")

    print("==================================================")
    print("  RoadSafety AI Web Studio & REST API Server")
    print(f"  Access URL: http://{host}:{port}")
    print("==================================================")

    app.run(host=host, port=port, debug=debug)
