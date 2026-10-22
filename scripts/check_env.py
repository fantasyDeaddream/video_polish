import importlib.util
import json
import sys

required = ['torch', 'cv2', 'numpy', 'fastapi', 'uvicorn', 'multipart']
missing = [p for p in required if importlib.util.find_spec(p) is None]
print(json.dumps({'python': sys.executable, 'version': sys.version.split()[0], 'missing': missing}))
raise SystemExit(bool(missing) or sys.version_info < (3, 10))
