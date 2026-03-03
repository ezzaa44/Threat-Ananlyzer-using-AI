import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import config
from src.predict import predict_file


def main():
    result = predict_file(str(config.SAMPLE_PATH))
    print(result)


if __name__ == "__main__":
    main()
