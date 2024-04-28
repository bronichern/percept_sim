import argparse
from glob import glob
import os
import pandas as pd


def unify_portion_files(dataset, input_dir,prefix):
    """
    Unify a list of files into a single file.
    """
    output_path = os.path.join(input_dir, f"{prefix}.csv")
    files = glob(os.path.join(input_dir, f"{prefix}_portion*.csv"))
    expected_files = 4 if dataset == 'kr' else 9
    assert len(files) == expected_files, f"Expected {expected_files} files for the {dataset} dataset, but got {len(files)}"
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    unified_df = pd.concat(dfs)
    print(f"Saving unified file to {output_path}")
    unified_df.to_csv(output_path, index=False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, required=True, help="Dataset name")
argparser.add_argument("--input_dir", type=str, required=True, help="Input directory")
argparser.add_argument("--prefix", type=str, required=True, help="Prefix of the portion files")
args = argparser.parse_args()
if __name__ == "__main__":
    unify_portion_files(args.dataset, args.input_dir, args.prefix)