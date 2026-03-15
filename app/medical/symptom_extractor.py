from app.data_loader import MedicalDataLoader, get_data_loader


class SymptomExtractor:
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()

    def extract(self, text: str) -> list[str]:
        return self.data_loader.extract_symptoms(text)
