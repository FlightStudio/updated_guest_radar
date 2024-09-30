import sys
from app import app

if __name__ == "__main__":
    try:
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        raise
