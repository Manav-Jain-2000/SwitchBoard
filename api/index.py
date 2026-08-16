import sys
import os
import importlib.util

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

main_path = os.path.join(src_dir, "api", "main.py")
spec = importlib.util.spec_from_file_location("src_api_main", main_path)
switchboard_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switchboard_api)

app = switchboard_api.app
