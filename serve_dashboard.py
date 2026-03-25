"""
Simple HTTP server to serve the static ML dashboard on port 8502.
Designed to run as a background task via Windows Task Scheduler.

Usage: python serve_dashboard.py
"""

import http.server
import os
import sys

PORT = 8502
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_site")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Suppress request logging for background operation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # silent


def main():
    os.chdir(STATIC_DIR)
    with http.server.HTTPServer(("0.0.0.0", PORT), QuietHandler) as server:
        print(f"Serving dashboard at http://localhost:{PORT}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
