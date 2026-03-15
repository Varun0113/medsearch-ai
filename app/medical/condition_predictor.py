from app.data_loader import MedicalDataLoader, get_data_loader


class ConditionPredictor:
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()

    def predict(self, symptoms: list[str], text: str, limit: int = 5) -> list[dict[str, object]]:
        return self.data_loader.get_conditions(symptoms, text, limit=limit)
