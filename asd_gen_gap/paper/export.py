"""CLI wrapper for exporting manuscript tables and figures."""

from __future__ import annotations

import argparse

from asd_gen_gap.paper.export_figures import export_paper_figures
from asd_gen_gap.paper.export_tables import export_paper_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-out")
    args = parser.parse_args(argv)
    export_paper_tables(args.results_dir, output_path=args.tables_out)
    export_paper_figures(args.results_dir)


if __name__ == "__main__":
    main()
