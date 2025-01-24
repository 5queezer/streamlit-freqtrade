import json
import zipfile


def load_json(filepath: str):
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    with z.open(name) as f:
                        return json.load(f)
        return None
    else:
        with open(filepath, "r") as f:
            return json.load(f)
