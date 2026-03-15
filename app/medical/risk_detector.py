from app.data_loader import MedicalDataLoader, get_data_loader


class RiskDetector:
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()

    def detect(self, text: str) -> dict[str, object]:
        matches = self.data_loader.detect_urgent_risks(text)
        if not matches:
            return {
                "urgent": False,
                "message": "",
                "matched_signals": [],
                "rationale": [],
            }

        banner = self.data_loader.get_system_message("urgent_banner")
        action = self.data_loader.get_system_message("urgent_action")
        return {
            "urgent": True,
            "message": f"{banner}\n\n{action}".strip(),
            "matched_signals": [match["label"] for match in matches],
            "rationale": [match["message"] for match in matches],
        }
