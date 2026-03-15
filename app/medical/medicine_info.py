from app.data_loader import MedicalDataLoader, get_data_loader


class MedicineInfoService:
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()

    def mentioned_medicine_keys(self, text: str) -> list[str]:
        return self.data_loader.mentioned_medicine_keys(text)

    def build_medicine_list(
        self,
        predicted_conditions: list[dict[str, object]],
        patient: dict[str, object],
        query_text: str,
    ) -> tuple[list[dict[str, object]], list[str]]:
        condition_keys = [
            str(condition["condition_key"])
            for condition in predicted_conditions[:3]
            if condition.get("condition_key")
        ]
        payload = self.data_loader.get_medicines(condition_keys, patient, query_text)
        return list(payload["all_medicines"]), list(payload["warnings"])
