# mgt_eval/cli_help_formatter.py
"""
Custom help formatter for beautiful CLI help messages using Rich.
"""

import argparse
import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    _HAS_RICH = True
    console = Console()
except ImportError:
    _HAS_RICH = False
    console = None


def print_beautiful_help():
    """Print a beautiful help message for mgteval-cli."""
    if not _HAS_RICH or console is None:
        # Fallback to simple text
        print_simple_help()
        return

    # Banner
    try:
        from cli_helpers import print_banner
        print_banner()
    except Exception:
        pass

    # Banner
    console.print()
    banner = Text()
    banner.append("MGTEval CLI", style="bold magenta on black")
    banner.append(" - ", style="white")
    banner.append("Machine-Generated Text Detection Framework", style="bold cyan")

    console.print(Panel.fit(
        banner,
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    # Description
    console.print(Panel(
        "[white]A unified command-line interface for building datasets, applying attacks, "
        "running detection, calibrating detectors, and training custom models.[/white]",
        title="[bold yellow]📖 Description[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Commands table
    commands_table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
        box=box.ROUNDED,
        title="[bold magenta]🚀 Available Commands[/bold magenta]",
        title_style="bold magenta",
    )
    commands_table.add_column("Command", style="green", width=15)
    commands_table.add_column("Description", style="white", width=60)

    commands = [
        ("list", "📋 List all available detectors"),
        ("info", "ℹ️  Show detailed information about a specific detector"),
        ("examples", "💡 Display usage examples for common tasks"),
        ("troubleshoot", "🔧 Show troubleshooting guide for common issues"),
        ("", ""),  # Separator
        ("detect", "🔍 Run detector evaluation on a dataset"),
        ("build", "🏗️  Build dataset by generating machine text from human samples"),
        ("attack", "⚔️  Apply adversarial text attacks to existing datasets"),
        ("train", "🏋️  Train or fine-tune a custom detector model"),
    ]

    for cmd, desc in commands:
        if cmd == "":
            commands_table.add_row("", "")
        else:
            commands_table.add_row(cmd, desc)

    console.print(commands_table)
    console.print()

    # Quick start examples
    examples = [
        ("Discovery", [
            "mgteval-cli list",
            "mgteval-cli info binoculars",
            "mgteval-cli examples",
        ]),
        ("Detection", [
            "mgteval-cli detect config.yaml",
            "mgteval-cli detect --detector binoculars \\",
            "    --data test.jsonl --model1 gpt2",
        ]),
        ("Dataset", [
            "mgteval-cli build config.yaml",
            "mgteval-cli attack config.yaml",
        ]),
    ]

    console.print(Panel(
        _format_quick_examples(examples),
        title="[bold cyan]⚡ Quick Start[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Common options
    options_content = (
        "[green]Global Options:[/green]\n"
        "  [cyan]-h, --help[/cyan]           Show this help message\n"
        "  [cyan]--version[/cyan]            Show version information\n\n"
        "[green]Command-specific Help:[/green]\n"
        "  [cyan]mgteval-cli <command> --help[/cyan]\n"
        "  Example: [dim]mgteval-cli detect --help[/dim]"
    )

    console.print(Panel(
        options_content,
        title="[bold yellow]⚙️  Options & Help[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Tips and notes
    tips_content = (
        "• Use [cyan]YAML configs[/cyan] for reproducible runs\n"
        "• Set [cyan]HF_ENDPOINT=\"https://hf-mirror.com\"[/cyan] for faster downloads\n"
        "• Use [cyan]--sample_k[/cyan] for quick testing on data subsets\n"
        "• Labels: [green]0=human[/green], [red]1=machine[/red]\n"
        "• Run [cyan]mgteval-cli troubleshoot[/cyan] if you encounter issues"
    )

    console.print(Panel(
        tips_content,
        title="[bold magenta]💡 Tips & Conventions[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Footer
    console.print("[dim]For detailed documentation, see README.md and CLI_ENHANCEMENTS.md[/dim]")
    console.print()


def _format_quick_examples(examples):
    """Format quick start examples."""
    lines = []
    for category, cmds in examples:
        lines.append(f"[bold yellow]{category}:[/bold yellow]")
        for cmd in cmds:
            if cmd.startswith(" "):
                lines.append(f"[dim]  {cmd}[/dim]")
            else:
                lines.append(f"  [green]$ {cmd}[/green]")
        lines.append("")
    return "\n".join(lines[:-1])  # Remove last empty line


def print_command_help(command: str):
    """Print beautiful help for a specific command."""
    if not _HAS_RICH or console is None:
        return None  # Let argparse handle it

    help_info = {
        "list": {
            "title": "📋 List Detectors",
            "description": "Display all available detection methods registered in MGTEval.",
            "usage": "mgteval-cli list [OPTIONS]",
            "options": [
                ("-v, --verbose", "Show detailed information with descriptions"),
            ],
            "examples": [
                "mgteval-cli list",
                "mgteval-cli list --verbose",
            ],
        },
        "info": {
            "title": "ℹ️  Detector Information",
            "description": "Show comprehensive details about a specific detector including paper reference, parameters, and usage examples.",
            "usage": "mgteval-cli info <detector_name>",
            "arguments": [
                ("detector_name", "Name of the detector (e.g., binoculars, fastdetectgpt)"),
            ],
            "examples": [
                "mgteval-cli info binoculars",
                "mgteval-cli info fastdetectgpt",
            ],
        },
        "detect": {
            "title": "🔍 Run Detection",
            "description": "Evaluate a detector on a dataset and compute performance metrics.",
            "usage": "mgteval-cli detect [config.yaml] [OPTIONS]",
            "key_options": [
                ("--detector", "Detector name (required if not in config)"),
                ("--data", "Input dataset path (required if not in config)"),
                ("--model1", "Primary model path/name (for metric-based detectors)"),
                ("--model2", "Secondary model path/name (optional)"),
                ("--batch_size", "Batch size for evaluation (default: 8)"),
                ("--threshold", "Classification threshold (default: 0.5)"),
                ("--out", "Output directory for results"),
                ("--sample_k", "Subsample K examples for quick testing"),
            ],
            "examples": [
                "# Using YAML config (recommended)",
                "mgteval-cli detect examples/detect/binoculars.yaml",
                "",
                "# Using command-line arguments",
                "mgteval-cli detect --detector binoculars --data test.jsonl \\",
                "    --model1 gpt-neo-2.7B --model2 gpt-neo-2.7B-instruct \\",
                "    --batch_size 16 --out results/",
            ],
        },
        "build": {
            "title": "🏗️  Build Dataset",
            "description": "Generate machine-written text from human samples using language models.",
            "usage": "mgteval-cli build <config.yaml>",
            "examples": [
                "mgteval-cli build examples/build/build_dataset.yaml",
            ],
            "notes": "Use YAML config for dataset building. See examples/build/ for templates.",
        },
        "attack": {
            "title": "⚔️  Apply Attacks",
            "description": "Apply adversarial text perturbations to test detector robustness.",
            "usage": "mgteval-cli attack <config.yaml>",
            "examples": [
                "mgteval-cli attack examples/attack/build_attack_dataset.yaml",
            ],
            "notes": "Supports 18+ attack methods. See examples/attack/ for configurations.",
        },
        "calibrate": {
            "title": "🎯 Calibrate Detector",
            "description": "Fit calibrator to map raw detector scores to calibrated probabilities.",
            "usage": "mgteval-cli calibrate [OPTIONS]",
            "key_options": [
                ("--detector", "Detector name (required)"),
                ("--data", "Calibration dataset path (required)"),
                ("--model1", "Primary model path/name (required for metric-based)"),
                ("--out", "Output path for calibrator JSON"),
            ],
            "examples": [
                "mgteval-cli calibrate --detector binoculars \\",
                "    --data dev.jsonl --model1 gpt-neo-2.7B \\",
                "    --out calibration_results/binoculars.json",
            ],
        },
        "train": {
            "title": "🏋️  Train Detector",
            "description": "Train or fine-tune a custom detection model on your dataset.",
            "usage": "mgteval-cli train [config.yaml] [OPTIONS]",
            "key_options": [
                ("--detector", "Detector/model name (required if not in config)"),
                ("--data", "Training dataset path (required if not in config)"),
                ("--epochs", "Number of training epochs"),
                ("--lr", "Learning rate"),
                ("--batch_size", "Training batch size"),
                ("--out", "Output directory for checkpoints"),
            ],
            "examples": [
                "mgteval-cli train examples/train/seqcls.yaml",
                "",
                "mgteval-cli train --detector roberta-base \\",
                "    --data train.jsonl --epochs 3 --lr 2e-5",
            ],
        },
    }

    if command not in help_info:
        return None  # Let argparse handle it

    info = help_info[command]

    console.print()
    console.print(Panel.fit(
        f"[bold magenta]{info['title']}[/bold magenta]",
        border_style="bright_blue",
        box=box.DOUBLE,
    ))
    console.print()

    # Description
    console.print(Panel(
        f"[white]{info['description']}[/white]",
        title="[bold yellow]📖 Description[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Usage
    console.print(Panel(
        f"[green]{info['usage']}[/green]",
        title="[bold cyan]📝 Usage[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Arguments (if any)
    if "arguments" in info:
        args_lines = []
        for arg, desc in info["arguments"]:
            args_lines.append(f"[green]{arg}[/green]")
            args_lines.append(f"  [dim]{desc}[/dim]")
        console.print(Panel(
            "\n".join(args_lines),
            title="[bold yellow]Arguments[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Options (if any)
    if "options" in info or "key_options" in info:
        opts = info.get("options", info.get("key_options", []))
        opts_lines = []
        for opt, desc in opts:
            opts_lines.append(f"[cyan]{opt}[/cyan]")
            opts_lines.append(f"  [dim]{desc}[/dim]")
            opts_lines.append("")
        console.print(Panel(
            "\n".join(opts_lines[:-1]),
            title="[bold yellow]⚙️  Options[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Examples
    if "examples" in info:
        example_lines = []
        for ex in info["examples"]:
            if ex.startswith("#"):
                example_lines.append(f"[dim]{ex}[/dim]")
            elif ex == "":
                example_lines.append("")
            else:
                example_lines.append(f"[green]$ {ex}[/green]")

        console.print(Panel(
            "\n".join(example_lines),
            title="[bold cyan]💡 Examples[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Notes (if any)
    if "notes" in info:
        console.print(Panel(
            f"[yellow]{info['notes']}[/yellow]",
            title="[bold yellow]📝 Notes[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    console.print("[dim]For more detailed options, add --help to see argparse output[/dim]")
    console.print()

    return True  # Handled


def print_simple_help():
    """Fallback help without Rich."""
    help_text = """
MGTEval CLI - Machine-Generated Text Detection Framework

USAGE:
    mgteval-cli <command> [options]

COMMANDS:
    list          List all available detectors
    info          Show detailed information about a detector
    examples      Display usage examples
    troubleshoot  Show troubleshooting guide

    detect        Run detector evaluation
    build         Build dataset with machine text
    attack        Apply adversarial attacks
    train         Train custom detector

QUICK START:
    mgteval-cli list
    mgteval-cli info binoculars
    mgteval-cli detect config.yaml
    mgteval-cli examples

For detailed help on a command:
    mgteval-cli <command> --help

For more information, see README.md
"""
    print(help_text)


class RichHelpAction(argparse.Action):
    """Custom action to intercept --help and show beautiful output."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print_beautiful_help()
        parser.exit()


class RichHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter that uses Rich for beautiful output (not actively used, kept for compatibility)."""
    pass
