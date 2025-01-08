import argparse

from .fmd import FrechetMusicDistance


def create_parser() -> argparse.ArgumentParser:
    """
    Create the top-level parser and subparsers for the 'score' and 'clear' commands.
    """
    parser = argparse.ArgumentParser(prog="fmd", description="A script for calculating Frechet Music Distance[FMD]")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")
    # ------------------------
    # Subparser: "score"
    # ------------------------
    score_parser = subparsers.add_parser("score", help="Compute Frechet Music Distance")
    score_parser.add_argument("reference_dataset", nargs="?", help="Path to reference dataset")
    score_parser.add_argument("test_dataset", nargs="?", help="Path to test dataset")
    score_parser.add_argument(
        "--model", "-m", choices=["clamp2", "clamp"], default="clamp2", help="Embedding model name"
    )
    score_parser.add_argument("--reference_ext", "-r", help="Music file extension in referene dataset (e.g. .midi)")
    score_parser.add_argument("--test_ext", "-t", help="Music file extension in test dataset (e.g. .midi)")
    score_parser.add_argument("--inf", action="store_true", help="Use FMD-Inf extrapolation")
    score_parser.add_argument("--steps", "-s", default=25, type=int, help="Number of steps when calculating FMD-Inf")
    score_parser.add_argument(
        "--min_n",
        "-n",
        default=500,
        type=int,
        help="Mininum sample size when calculating FMD-Inf (Must be smaller than the size of test dataset)",
    )
    score_parser.add_argument("--clear-cache", action="store_true", help="Clear precomputed cache")

    # ------------------------
    # Subparser: "clear"
    # ------------------------
    subparsers.add_parser("clear", help="Clear precomputed cache")

    return parser


def run_score(parser, args, metric):
    if args.clear_cache:
        metric.clear_cache()
    if not args.reference_dataset or not args.test_dataset:
        parser.error("The following arguments are required: reference_dataset, test_dataset")

    if args.inf:
        result = metric.score_inf(
            args.reference_dataset,
            args.test_dataset,
            args.reference_ext,
            args.test_ext,
            steps=args.steps,
            min_n=args.min_n,
            method="mle",
        )
        print(f"Frechet Music Distance [FMD-Inf]: {result.score}; R^2 = {result.r2}")

    else:
        score = metric.score(args.reference_dataset, args.test_dataset, args.reference_ext, args.test_ext, method="mle")
        print(f"Frechet Music Distance [FMD]: {score}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.command == "clear":
        FrechetMusicDistance.clear_cache()
    elif args.command == "score":
        metric = FrechetMusicDistance(model_name=args.model, verbose=True)
        run_score(parser, args, metric)


if __name__ == "__main__":
    main()
