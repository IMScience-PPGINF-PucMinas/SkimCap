import sys
from HieTaSumm import HieTaSumm

def main():
    if len(sys.argv) != 5:
        print("Usage: python -m HieTaSumm.One_Line_Function [features_path] [summary_path] [gen_summary_method] [hierarchy]")
        sys.exit(1)

    features_path = sys.argv[1]
    summary_path = sys.argv[2]
    gen_summary_method = {"method": sys.argv[3]}
    hierarchy = sys.argv[4]

    HieTaSumm(
        features_path=features_path,
        percent=15,
        rate=30,
        time=4,
        alpha=100,
        selected_model='vgg16',
        summary_path=summary_path,
        gen_summary_method=gen_summary_method,
        hierarchy=hierarchy
    )

if __name__ == "__main__":
    main()
