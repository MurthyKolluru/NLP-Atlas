import nbformat as nbf
from pathlib import Path

nb_path = Path("notebooks/00_explore_svd.ipynb")
nb = nbf.read(nb_path, as_version=4)

bootstrap = nbf.v4.new_code_cell("""# Make project root importable so 'from src...' works
import sys
from pathlib import Path

# If running from notebooks/, project root is parent
ROOT = Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("PYTHONPATH[0]:", sys.path[0])
""")

# Only add if not already present (idempotent)
first_src = "\\n".join(nb["cells"][0]["source"]) if nb["cells"] else ""
if "Make project root importable" not in first_src:
    nb["cells"].insert(0, bootstrap)

nbformat = 4
nbf.write(nb, nb_path)
print("Patched:", nb_path)
