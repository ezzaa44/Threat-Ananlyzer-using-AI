import pandas as pd


def parse_snort_alerts(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["is_malicious_alert"] = 1
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parse_snort_alerts("snort/snort_alerts.csv", "snort/snort_alerts_parsed.csv")
    print("Parsed alerts saved to snort/snort_alerts_parsed.csv")
