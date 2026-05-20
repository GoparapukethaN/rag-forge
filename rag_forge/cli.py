"""CLI for rag-forge."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_forge.bench import BenchConfig, load_documents, load_qa_pairs, run_benchmark
from rag_forge.gate import (
    GateThresholds,
    evaluate_regression_gate,
    load_json_report,
    write_gate_json,
    write_gate_markdown,
)
from rag_forge.report import generate_json_report, generate_markdown_report, generate_pareto_plot

app = typer.Typer(help="rag-forge: find the best RAG config for your documents.")
console = Console()


@app.command()
def run(
    docs: str = typer.Option(..., help="Path to directory with .txt/.md documents"),
    qa: str = typer.Option(..., help="Path to CSV with 'question' and 'answer' columns"),
    output: str = typer.Option("./results", help="Output directory for results"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve per query"),
    skip_openai: bool = typer.Option(False, help="Skip OpenAI embedder even if key is set"),
    skip_reranker: bool = typer.Option(False, help="Skip cross-encoder reranking"),
):
    """Run the benchmark: test all chunking × embedding × retrieval × reranking combos."""
    console.print("\n[bold]rag-forge[/bold] — finding your optimal RAG config\n")

    # load data
    console.print(f"Loading documents from {docs}...")
    documents = load_documents(docs)
    console.print(f"  → {len(documents)} documents loaded")

    console.print(f"Loading QA pairs from {qa}...")
    questions, ground_truths = load_qa_pairs(qa)
    console.print(f"  → {len(questions)} QA pairs loaded\n")

    # configure
    config = BenchConfig(top_k=top_k)
    if skip_openai:
        config.embedders = [e for e in config.embedders if e != "openai"]
    if skip_reranker:
        config.rerankers = ["none"]

    # run
    results = run_benchmark(documents, questions, ground_truths, config, verbose=True)

    # save results
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "results.md"
    generate_markdown_report(results, str(report_path))
    console.print(f"\n[green]✓[/green] Report saved to {report_path}")

    json_path = out_dir / "results.json"
    generate_json_report(results, str(json_path))
    console.print(f"[green]✓[/green] JSON report saved to {json_path}")

    try:
        plot_path = out_dir / "pareto.png"
        generate_pareto_plot(results, str(plot_path))
        console.print(f"[green]✓[/green] Pareto plot saved to {plot_path}")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not generate plot: {e}")

    # show top 5
    console.print("\n[bold]Top 5 Configurations:[/bold]\n")
    _print_results_table(results[:5])


@app.command()
def report(
    results_file: str = typer.Option("./results/results.md", help="Path to results markdown"),
):
    """Display a previously generated benchmark report."""
    path = Path(results_file)
    if not path.exists():
        console.print(f"[red]Error:[/red] {results_file} not found. Run `rag-forge run` first.")
        raise typer.Exit(1)
    console.print(path.read_text())


@app.command()
def gate(
    baseline: str = typer.Option(..., help="Baseline results.json file"),
    current: str = typer.Option(..., help="Current results.json file"),
    output: str = typer.Option("./results/gate.json", help="Path for machine-readable gate output"),
    markdown: str = typer.Option("./results/gate.md", help="Path for Markdown gate summary"),
    max_hit_rate_drop: float = typer.Option(0.02, help="Maximum allowed hit rate drop"),
    max_mrr_drop: float = typer.Option(0.02, help="Maximum allowed MRR drop"),
    max_latency_increase_pct: float = typer.Option(
        25.0,
        help="Maximum allowed latency increase percentage",
    ),
):
    """Compare two benchmark reports and fail when retrieval quality regresses."""
    baseline_report = load_json_report(baseline)
    current_report = load_json_report(current)
    gate_report = evaluate_regression_gate(
        baseline_report,
        current_report,
        thresholds=GateThresholds(
            max_hit_rate_drop=max_hit_rate_drop,
            max_mrr_drop=max_mrr_drop,
            max_latency_increase_pct=max_latency_increase_pct,
        ),
    )
    write_gate_json(gate_report, output)
    write_gate_markdown(gate_report, markdown)

    verdict = gate_report["verdict"]
    if verdict == "fail":
        console.print(
            f"[red]Regression gate failed[/red]. Reports saved to {output} and {markdown}."
        )
        raise typer.Exit(1)
    if verdict == "warn":
        console.print(
            f"[yellow]Regression gate passed with warnings[/yellow]. Reports saved to {output}."
        )
        return
    console.print(f"[green]Regression gate passed[/green]. Reports saved to {output}.")


def _print_results_table(results):
    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Chunker")
    table.add_column("Embedder")
    table.add_column("Retriever")
    table.add_column("Reranker")
    table.add_column("Hit Rate", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Latency", justify="right")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.chunker,
            r.embedder,
            r.retriever,
            r.reranker,
            f"{r.eval.hit_rate:.3f}",
            f"{r.eval.mrr:.3f}",
            f"{r.latency_ms:.0f}ms",
        )
    console.print(table)


if __name__ == "__main__":
    app()
