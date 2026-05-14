"""distillarium CLI — distill / taste / bottle subcommands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


LOGO = r"""
   ___ _ _ _
  |   \ (_) | |               ___           _   _  _ _
  | |) | _| | | ___ _ __ _  _/ __|_ __ ___ | |_| || | |_ _
  |  _/  | | | / -_) '_/ || \__ \ '_(_-(_-<| ' \ || |  _| |
  |_|   _|_|_|_\___|_|  \_, |___/_| /__/__/|_||_\_, /\__|_|
                        |__/                    |__/
"""


@click.group(help="⚗  The Distillery — distill any task into a pocket-sized Spirit.")
@click.version_option()
def main():
    pass


@main.command(help="Run a recipe end-to-end and bottle the resulting Spirit.")
@click.argument("recipe_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", default="spirits/", help="Output directory")
@click.option("--mash", default=None, help="Pre-existing mash JSONL (skip teacher call)")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def distill(recipe_path, out, mash, quiet):
    """Distill a Spirit from a Recipe."""
    from distillarium import Recipe, distill as run_distill

    recipe = Recipe.from_file(recipe_path)

    if not quiet:
        console.print(Panel.fit(
            f"[bold gold1]⚗  Distilling [yellow]{recipe.name}[/yellow] v{recipe.version}[/bold gold1]\n"
            f"   Teacher: [cyan]{recipe.teacher.provider}:{recipe.teacher.model}[/cyan]\n"
            f"   Student: [cyan]{recipe.student.arch}[/cyan] "
            f"(d={recipe.student.d_model}, h={recipe.student.n_heads}, L={recipe.student.n_layers})\n"
            f"   Mash:    [cyan]{recipe.mash.total_examples} examples[/cyan]\n"
            f"   Still:   [cyan]{recipe.still.epochs} epochs @ lr={recipe.still.lr}[/cyan]",
            title="THE STILL",
            border_style="orange3",
        ))

    spirit = run_distill(recipe, out_dir=out, mash_path=mash, verbose=not quiet)

    out_path = Path(out) / f"{recipe.name.replace('.', '_')}.pt"
    spirit.save(out_path)

    if not quiet:
        _print_tasting_notes(spirit)
        console.print(f"\n[bold green]🍾 Bottled:[/bold green] {out_path}")
        console.print(f"[dim]   {spirit.n_params/1e6:.2f}M params · "
                      f"{spirit.proof}° proof · "
                      f"final loss {spirit.loss_curve[-1]:.3f}[/dim]")


@main.command(help="Taste a Spirit — run held-out eval and print notes.")
@click.argument("spirit_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--mash", required=True, help="Held-out JSONL to taste against")
@click.option("--held-out", "-n", default=100, help="Number of examples to evaluate")
@click.option("--json", "as_json", is_flag=True, help="Output metrics as JSON")
def taste(spirit_path, mash, held_out, as_json):
    """Re-taste a Spirit against a held-out cut."""
    from distillarium import load_spirit, taste as run_taste

    spirit = load_spirit(spirit_path)
    metrics = run_taste(spirit, mash, held_out=held_out)

    if as_json:
        click.echo(json.dumps(metrics, indent=2))
    else:
        _print_tasting_notes(spirit)


@main.command(help="Bottle a Spirit into a deployable format (onnx, gguf, ...).")
@click.argument("spirit_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "-f", default="pytorch",
              type=click.Choice(["pytorch", "onnx"]),
              help="Export format")
@click.option("--out", "-o", default=None, help="Output path (defaults to spirits/<name>.<ext>)")
def bottle(spirit_path, format, out):
    """Export a Spirit to a deployable format."""
    from distillarium import load_spirit, bottle as run_bottle

    spirit = load_spirit(spirit_path)
    path = run_bottle(spirit, format=format, out=out)
    console.print(f"[bold green]🍾 Bottled:[/bold green] {path}")


@main.command(help="Show what's in your local Cellar.")
@click.option("--dir", "cellar_dir", default="spirits/",
              type=click.Path(file_okay=False),
              help="Cellar directory")
def cellar(cellar_dir):
    """List Spirits in the local Cellar."""
    from distillarium import load_spirit

    cellar_path = Path(cellar_dir)
    if not cellar_path.exists():
        console.print(f"[yellow]No Cellar at {cellar_dir} — run a distillation first.[/yellow]")
        return

    bottles = sorted(cellar_path.glob("*.pt"))
    if not bottles:
        console.print(f"[yellow]Cellar at {cellar_dir} is empty.[/yellow]")
        return

    table = Table(title="🍾 Your Cellar", title_style="bold gold1")
    table.add_column("Spirit", style="cyan")
    table.add_column("Proof", justify="right", style="yellow")
    table.add_column("Params", justify="right")
    table.add_column("Final loss", justify="right")
    table.add_column("Path", style="dim")

    for b in bottles:
        try:
            s = load_spirit(b)
            table.add_row(
                s.name,
                f"{s.proof}°",
                f"{s.n_params/1e6:.1f}M",
                f"{s.loss_curve[-1]:.3f}" if s.loss_curve else "—",
                str(b),
            )
        except Exception as e:
            table.add_row(b.stem, "?", "?", "?", f"[red]error: {e}[/red]")

    console.print(table)


def _print_tasting_notes(spirit):
    """Pretty-print the Tasting Notes for a Spirit."""
    m = spirit.metrics

    panel_body = (
        f"[bold]Tool-name accuracy:[/bold]  "
        f"[yellow]{m.get('tool_name_accuracy', 0)*100:.1f}%[/yellow]\n"
        f"[bold]Exact-call accuracy:[/bold] "
        f"[yellow]{m.get('exact_call_accuracy', 0)*100:.1f}%[/yellow]\n"
        f"[bold]Arg-key F1:[/bold]          "
        f"[yellow]{m.get('arg_key_f1', 0):.3f}[/yellow] "
        f"(p={m.get('arg_key_precision', 0):.2f}, r={m.get('arg_key_recall', 0):.2f})\n"
        f"[bold]Evaluated on:[/bold]        "
        f"[cyan]{m.get('n_evaluated', 0)} held-out cuts[/cyan]"
    )

    console.print(Panel(panel_body, title="📝  TASTING NOTES", border_style="gold1"))

    samples = m.get("samples", [])[:4]
    if samples:
        table = Table(title="Sample Predictions", show_header=True, header_style="bold")
        table.add_column("Utterance", style="white", max_width=40)
        table.add_column("Gold tool", style="green")
        table.add_column("Predicted", style="cyan")
        table.add_column("Verdict")

        for s in samples:
            pred_str = s["predicted"]["name"] if s["predicted"] else "—"
            verdict = s["verdict"]
            color = {"exact": "green", "tool_only": "yellow", "wrong": "red"}.get(verdict, "white")
            table.add_row(
                s["utterance"][:40] + ("…" if len(s["utterance"]) > 40 else ""),
                s["gold"]["name"],
                pred_str,
                f"[{color}]{verdict}[/{color}]",
            )
        console.print(table)


if __name__ == "__main__":
    main()
