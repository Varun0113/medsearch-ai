import pandas as pd
import re
from pathlib import Path

class BrandMatcher:
    def __init__(self):
        metadata_path = Path("indexes/medicine_metadata.pkl")

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Metadata file not found. Run build_index.py first."
            )

        df = pd.read_pickle(metadata_path)

        # Precompute lowercase brand names
        self.brands = (
            df["brand_name"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

    def extract(self, text: str):
        text = text.lower()
        found = []

        for brand in self.brands:
            # Word boundary match to avoid partial hits
            pattern = r"\b" + re.escape(brand) + r"\b"
            if re.search(pattern, text):
                found.append(brand)

        return found
