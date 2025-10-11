import os
import pandas as pd


def main() -> None:
    path = os.path.join(os.path.dirname(__file__), "hamamatsu.csv")
    if not os.path.exists(path):
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path, header=None, dtype=str)
    df = df[[0, 2, 7]].rename(columns={0: "pid", 2: "age_band", 7: "date"})

    switched = (df.groupby("pid")["age_band"].nunique() > 1).sum()
    print("people who change age band:", int(switched))


if __name__ == "__main__":
    main()


