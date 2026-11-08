"""LightGBM refinements reuse original fixed parameters and save arbitrary schemas."""
import json
from pathlib import Path

from experiments.property_groups.models import GroupModel


class LandModel(GroupModel):
    def __init__(self):
        super().__init__("global", "lightgbm")

    def save_land(self, folder, cutoff, training_n, variant):
        self.save_bundle(folder, cutoff, training_n)
        path = Path(folder) / "schema.json"
        schema = json.loads(path.read_text())
        schema["variant"] = variant
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
