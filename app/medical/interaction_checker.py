from app.data_loader import MedicalDataLoader, get_data_loader


class DrugInteractionChecker:
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()

    def check(
        self,
        medicine_entries: list[dict[str, object]],
        patient: dict[str, object],
    ) -> list[dict[str, object]]:
        return self.data_loader.get_interaction_warnings(medicine_entries, patient)
