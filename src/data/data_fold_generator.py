import csv
import argparse
import os

def split_csv_k_fold(input_file, k=5):
    base = os.path.splitext(os.path.basename(input_file))[0]  # e.g., 'data.csv' → 'data'
    output_files = [open(f"{base}_fold{i+1}.csv", "w", newline='') for i in range(k)]
    writers = [csv.writer(f) for f in output_files]

    with open(input_file, "r", newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for writer in writers:
            writer.writerow(header)

        for i, row in enumerate(reader):
            writers[i % k].writerow(row)

    for f in output_files:
        f.close()

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into K folds (round-robin).")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    args = parser.parse_args()

    split_csv_k_fold(args.input_file, k=args.k)

if __name__ == "__main__":
    main()
