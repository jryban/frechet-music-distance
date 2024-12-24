import argparse

from .fmd import FrechetMusicDistance


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmd", description="A script for calculating Frechet Music Distance[FMD]")
    parser.add_argument("reference_dataset", nargs="?", help="Path to reference dataset")
    parser.add_argument("test_dataset", nargs="?", help="Path to test dataset")
    parser.add_argument("--model", "-m", choices=["clamp2", "clamp"], default="clamp2", help="Embedding model name")
    parser.add_argument("--reference_ext", "-r", help="Music file extension in referene dataset (e.g. .midi)")
    parser.add_argument("--test_ext", "-t", help="Music file extension in test dataset (e.g. .midi)")
    parser.add_argument("--inf", action="store_true", help="Use FMD-Inf extrapolation")
    parser.add_argument("--steps", "-s", default=25, type=int, help="Number of steps when calculating FMD-Inf")
    parser.add_argument("--min_n", "-n", default=500, type=int, help="Mininum sample size when calculating FMD-Inf (Must be smaller than the size of test dataset)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear precomputed cache")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    metric = FrechetMusicDistance(model_name=args.model, verbose=True)
    if args.clear_cache:
        metric.clear_cache()

    else:
        if not args.reference_dataset or not args.test_dataset:
            parser.error("The following arguments are required: reference_dataset, test_dataset")

        if args.inf:
            result = metric.score_inf(args.reference_dataset, args.test_dataset, args.reference_ext, args.test_ext,
                                    steps=args.steps, min_n=args.min_n, method="mle")
            print(f"Frechet Music Distance [FMD-Inf]: {result.score}; R^2 = {result.r2}")

        else:
            score = metric.score(args.reference_dataset, args.test_dataset, args.reference_ext, args.test_ext, method="mle")
            print(f"Frechet Music Distance [FMD]: {score}")


if __name__ == "__main__":
    main()
