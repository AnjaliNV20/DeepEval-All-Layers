import json
from pathlib import Path


def save_report(report_file: Path, results: list) -> None:
    # Save complete evaluation report.
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(results,file,indent=4,ensure_ascii=False
        )

        
def save_test_result(report_file: Path, test_result: dict) -> None:
    # Add or update one test result in the evaluation report.
    report_file.parent.mkdir(parents=True, exist_ok=True)
    existing_results = []
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as file:
                existing_results = json.load(file)

        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []

    # Remove old result for same test case
    existing_results = [
        result
        for result in existing_results
        if result.get("test_case_id") != test_result.get("test_case_id")
    ]

    existing_results.append(test_result)

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump( existing_results,file,indent=4,ensure_ascii=False
        )



def load_report(report_file: Path) -> list:
    # Load existing evaluation report.
    
    if not report_file.exists():
        return []

    with open(report_file, "r", encoding="utf-8") as file:
        return json.load(file)