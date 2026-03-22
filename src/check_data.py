import pandas as pd
import os

def check_dataset(path, name):
    print(f"\n{'='*40}")
    print(f"Checking {name} dataset...")
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        print(f"  Rows    : {df.shape[0]}")
        print(f"  Columns : {df.shape[1]}")
        print(f"  Columns : {list(df.columns)}")
        missing = df.isnull().sum()[df.isnull().sum()>0]
        if len(missing) > 0:
            print(f"  Missing :\n{missing}")
        else:
            print(f"  Missing : None ✅")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    check_dataset("data/movies/tmdb_5000_movies.csv", "Movies")
    check_dataset("data/food/RAW_recipes.csv",         "Food")
    check_dataset("data/fashion/styles.csv",           "Fashion")
