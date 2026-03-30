"""Click CLI for survey-cleaner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from survey_cleaner.models import CleaningResult

import click

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    _rich = True
except ImportError:  # pragma: no cover
    _rich = False


def _console() -> "Console":  # type: ignore[name-defined]
    return Console(stderr=False)


@click.group()
@click.version_option(package_name="survey-cleaner")
def cli() -> None:
    """Survey response quality filter — remove speeders, straight-liners, and gibberish."""


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Path for the cleaned CSV output. Skipped if not provided.",
)
@click.option(
    "--report",
    "-r",
    default=None,
    help="Path for the JSON cleaning report.",
)
@click.option(
    "--min-time",
    default=120,
    show_default=True,
    help="Minimum acceptable completion time in seconds (speeder threshold).",
)
@click.option(
    "--timing-col",
    default=None,
    help="Column name containing completion time in seconds (auto-detected if omitted).",
)
@click.option(
    "--open-end-cols",
    default=None,
    help="Comma-separated list of open-end text column names (auto-detected if omitted).",
)
@click.option(
    "--likert-cols",
    default=None,
    help="Comma-separated list of Likert-scale column names (auto-detected if omitted).",
)
@click.option(
    "--id-col",
    default=None,
    help="Column to use as respondent identifier in the report.",
)
@click.option(
    "--variance-threshold",
    default=0.0,
    show_default=True,
    help="Likert variance threshold for straight-liner detection (0 = pure straight-liners only).",
)
@click.option(
    "--gibberish-threshold",
    default=0.5,
    show_default=True,
    help="Composite gibberish score cutoff (0–1).",
)
@click.option("--no-speeders", is_flag=True, help="Disable speeder detection.")
@click.option("--no-straight-liners", is_flag=True, help="Disable straight-liner/pattern-liner detection.")
@click.option("--no-gibberish", is_flag=True, help="Disable open-end gibberish detection.")
@click.option(
    "--llm-backend",
    type=click.Choice(["openai", "anthropic", "gemini"]),
    default=None,
    help="Optional LLM backend for open-end quality enrichment.",
)
@click.option("--dry-run", is_flag=True, help="Report flags without writing any output files.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress terminal output.")
def clean(
    input_file: str,
    output: Optional[str],
    report: Optional[str],
    min_time: int,
    timing_col: Optional[str],
    open_end_cols: Optional[str],
    likert_cols: Optional[str],
    id_col: Optional[str],
    variance_threshold: float,
    gibberish_threshold: float,
    no_speeders: bool,
    no_straight_liners: bool,
    no_gibberish: bool,
    llm_backend: Optional[str],
    dry_run: bool,
    quiet: bool,
) -> None:
    """Clean a survey CSV by removing low-quality responses."""
    import pandas as pd

    from survey_cleaner.cleaner import SurveyCleaner

    df = pd.read_csv(input_file)

    oe_cols = [c.strip() for c in open_end_cols.split(",")] if open_end_cols else None
    lk_cols = [c.strip() for c in likert_cols.split(",")] if likert_cols else None

    cleaner = SurveyCleaner(
        timing_col=timing_col,
        min_seconds=min_time,
        likert_cols=lk_cols,
        variance_threshold=variance_threshold,
        open_end_cols=oe_cols,
        min_chars=10,
        gibberish_score_threshold=gibberish_threshold,
        id_col=id_col,
        run_speeders=not no_speeders,
        run_straight_liners=not no_straight_liners,
        run_gibberish=not no_gibberish,
        llm_backend=llm_backend,
    )

    cleaned_df, result = cleaner.clean_df(df)

    if not quiet:
        _print_result(result, input_file)

    if not dry_run:
        if output:
            cleaned_df.to_csv(output, index=False)
            if not quiet:
                click.echo(f"✓ Cleaned CSV written to {output}")

        if report:
            Path(report).write_text(json.dumps(result.to_dict(), indent=2))
            if not quiet:
                click.echo(f"✓ Report written to {report}")

    # Exit 1 if flag rate > 20% (useful in CI pipelines)
    if result.flag_rate > 0.20:
        sys.exit(1)


# ---------------------------------------------------------------------------
# report command (report-only, no cleaning)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--min-time", default=120, show_default=True)
@click.option("--timing-col", default=None)
@click.option("--open-end-cols", default=None)
@click.option("--likert-cols", default=None)
@click.option("--id-col", default=None)
@click.option("--variance-threshold", default=0.0, show_default=True)
@click.option("--gibberish-threshold", default=0.5, show_default=True)
@click.option("--json-out", "-j", default=None, help="Write JSON report to this path.")
def report(
    input_file: str,
    min_time: int,
    timing_col: Optional[str],
    open_end_cols: Optional[str],
    likert_cols: Optional[str],
    id_col: Optional[str],
    variance_threshold: float,
    gibberish_threshold: float,
    json_out: Optional[str],
) -> None:
    """Analyse a survey CSV and print a quality report without modifying any data."""
    import pandas as pd

    from survey_cleaner.cleaner import SurveyCleaner

    df = pd.read_csv(input_file)

    oe_cols = [c.strip() for c in open_end_cols.split(",")] if open_end_cols else None
    lk_cols = [c.strip() for c in likert_cols.split(",")] if likert_cols else None

    cleaner = SurveyCleaner(
        timing_col=timing_col,
        min_seconds=min_time,
        likert_cols=lk_cols,
        variance_threshold=variance_threshold,
        open_end_cols=oe_cols,
        gibberish_score_threshold=gibberish_threshold,
        id_col=id_col,
    )

    result = cleaner.clean(df)
    _print_result(result, input_file)

    if json_out:
        Path(json_out).write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"✓ Report written to {json_out}")


# ---------------------------------------------------------------------------
# internal rendering
# ---------------------------------------------------------------------------


def _print_result(result: "CleaningResult", source: str) -> None:
    if _rich:
        _print_rich(result, source)
    else:
        _print_plain(result, source)


def _print_rich(result: "CleaningResult", source: str) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    flag_pct = f"{result.flag_rate * 100:.1f}%"
    summary = (
        f"[bold]{result.total_responses}[/] responses analysed · "
        f"[red]{result.flagged_count} flagged[/] ({flag_pct}) · "
        f"[green]{result.clean_count} clean[/]"
    )
    console.print()
    console.print(Panel(summary, title=f"[bold]survey-cleaner[/] · {Path(source).name}", expand=False))

    if result.flagged_count == 0:
        console.print("[green]✓ No quality issues detected.[/]")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Row", style="dim", width=6)
    table.add_column("ID", style="cyan", width=14)
    table.add_column("Flag(s)", style="yellow")
    table.add_column("Detail", style="dim")
    table.add_column("Sev.", justify="right", width=5)

    for fr in result.flagged:
        codes = ", ".join(set(r.code.value for r in fr.reasons))
        detail = fr.reasons[0].detail[:60] + ("…" if len(fr.reasons[0].detail) > 60 else "")
        table.add_row(
            str(fr.row_index),
            fr.respondent_id or "—",
            codes,
            detail,
            f"{fr.max_severity:.2f}",
        )

    console.print(table)

    breakdown = (
        f"  Speeders: [red]{result.speeder_count}[/]  "
        f"Straight-liners: [red]{result.straight_liner_count}[/]  "
        f"Pattern-liners: [red]{result.pattern_liner_count}[/]  "
        f"Gibberish/short: [red]{result.gibberish_count}[/]"
    )
    console.print(breakdown)
    console.print()


def _print_plain(result: "CleaningResult", source: str) -> None:
    click.echo(f"\nsurvey-cleaner — {Path(source).name}")
    click.echo(
        f"  Total: {result.total_responses} | "
        f"Flagged: {result.flagged_count} ({result.flag_rate*100:.1f}%) | "
        f"Clean: {result.clean_count}"
    )
    click.echo(
        f"  Speeders: {result.speeder_count} | "
        f"Straight-liners: {result.straight_liner_count} | "
        f"Pattern-liners: {result.pattern_liner_count} | "
        f"Gibberish: {result.gibberish_count}"
    )
    for fr in result.flagged:
        codes = ", ".join(r.code.value for r in fr.reasons)
        click.echo(f"  Row {fr.row_index}: {codes}")
