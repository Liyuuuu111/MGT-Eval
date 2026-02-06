# mgt_eval/cli_helpers.py
"""
Enhanced CLI helpers for beautiful output and improved user experience.
"""
from typing import Any, Dict, List, Optional, Tuple
import sys
import os
from pathlib import Path

# Rich imports for beautiful CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    from rich.prompt import Prompt, Confirm
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# Initialize console
if _HAS_RICH:
    console = Console()
    console_err = Console(stderr=True)
else:
    console = None
    console_err = None

# =======================
# Detector Metadata Registry
# =======================

DETECTOR_METADATA = {
    # Metric-based detectors
    "binoculars": {
        "name": "Binoculars",
        "type": "Metric-based",
        "description": "Likelihood ratio between observer and performer models",
        "paper": "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text",
        "authors": "Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein",
        "params": ["observer_name_or_path", "performer_name_or_path", "device", "dtype"],
        "example": "mgteval-cli detect --detector binoculars --data test.jsonl --model1 gpt-neo-2.7B --model2 gpt-neo-2.7B-instruct",
    },
    "detectgpt": {
        "name": "DetectGPT",
        "type": "Metric-based",
        "description": "Probability curvature detection using perturbations",
        "paper": "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature",
        "authors": "Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn",
        "params": ["scoring_model_name", "mask_model", "device", "n_perturbations"],
        "example": "mgteval-cli detect --detector detectgpt --data test.jsonl --model1 gpt2-xl",
    },
    "fastdetectgpt": {
        "name": "Fast-DetectGPT",
        "type": "Metric-based",
        "description": "Optimized DetectGPT with conditional probability curvature",
        "paper": "Fast-DetectGPT: Efficient Detection of Machine-Generated Text via Sampling Discrepancy",
        "authors": "Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang",
        "params": ["scoring_model_name", "sampling_model_name", "device"],
        "example": "mgteval-cli detect --detector fastdetectgpt --data test.jsonl --model1 gpt2-xl --model2 gpt2",
    },
    "gltr": {
        "name": "GLTR",
        "type": "Metric-based",
        "description": "Giant Language Test Room - rank-based statistics",
        "paper": "GLTR: Statistical Detection and Visualization of Generated Text",
        "authors": "Sebastian Gehrmann, Hendrik Strobelt, Alexander M. Rush",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector gltr --data test.jsonl --model1 gpt2-large",
    },
    "likelihood": {
        "name": "Likelihood",
        "type": "Metric-based",
        "description": "Log-probability scoring under language model",
        "paper": "N/A",
        "authors": "Community Implementations",
        "params": ["score_model", "device"],
        "example": "mgteval-cli detect --detector likelihood --data test.jsonl --model1 gpt2",
    },
    "rank": {
        "name": "Rank",
        "type": "Metric-based",
        "description": "Token rank statistics",
        "paper": "N/A",
        "authors": "Community Implementations",
        "params": ["score_model", "device"],
        "example": "mgteval-cli detect --detector rank --data test.jsonl --model1 gpt2",
    },
    "logrank": {
        "name": "LogRank",
        "type": "Metric-based",
        "description": "Log-rank metric",
        "paper": "N/A",
        "authors": "Community Implementations",
        "params": ["score_model", "device"],
        "example": "mgteval-cli detect --detector logrank --data test.jsonl --model1 gpt2",
    },
    "entropy": {
        "name": "Entropy",
        "type": "Metric-based",
        "description": "Entropy-based scoring",
        "paper": "N/A",
        "authors": "Community Implementations",
        "params": ["score_model", "device"],
        "example": "mgteval-cli detect --detector entropy --data test.jsonl --model1 gpt2",
    },
    "lrr": {
        "name": "LRR",
        "type": "Metric-based",
        "description": "Likelihood Ratio with Rank",
        "paper": "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text",
        "authors": "Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov",
        "params": ["score_model", "device"],
        "example": "mgteval-cli detect --detector lrr --data test.jsonl --model1 gpt2",
    },
    "npr": {
        "name": "NPR",
        "type": "Metric-based",
        "description": "Nested Prediction Ratio",
        "paper": "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text",
        "authors": "Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector npr --data test.jsonl --model1 gpt2-large",
    },
    "raidar": {
        "name": "RAIDAR",
        "type": "Metric-based",
        "description": "Radar-based detection method",
        "paper": "Raidar: geneRative AI Detection viA Rewriting",
        "authors": "Chengzhi Mao, Carl Vondrick, Hao Wang, Junfeng Yang",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector raidar --data test.jsonl --model1 gpt2-xl",
    },
    "tocsin": {
        "name": "TOCSIN",
        "type": "Metric-based",
        "description": "Token-level OCR-based detection",
        "paper": "Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness",
        "authors": "Shixuan Ma, Quan Wang",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector tocsin --data test.jsonl --model1 roberta-large",
    },
    "dnadetectllm": {
        "name": "DNA-DetectLLM",
        "type": "Metric-based",
        "description": "DNA-based detection for large language models",
        "paper": "DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm",
        "authors": "Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector dnadetectllm --data test.jsonl --model1 gpt2-xl",
    },
    "dnagpt": {
        "name": "DNA-GPT",
        "type": "Metric-based",
        "description": "DNA-style detection for GPT models",
        "paper": "DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text",
        "authors": "Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector dnagpt --data test.jsonl --model1 gpt2-xl",
    },
    "lastde": {
        "name": "LASTDE",
        "type": "Metric-based",
        "description": "Last-layer hidden state detection",
        "paper": "Training-free LLM-generated Text Detection by Mining Token Probability Sequences",
        "authors": "Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector lastde --data test.jsonl --model1 roberta-base",
    },
    "lastdepp": {
        "name": "LASTDE++",
        "type": "Metric-based",
        "description": "Enhanced LASTDE variant",
        "paper": "Training-free LLM-generated Text Detection by Mining Token Probability Sequences",
        "authors": "Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu",
        "params": ["model_name", "device"],
        "example": "mgteval-cli detect --detector lastdepp --data test.jsonl --model1 roberta-base",
    },
    # Pretrained detectors
    "pretrained": {
        "name": "Pretrained Detector",
        "type": "Pretrained",
        "description": "Generic loader for HuggingFace classification models",
        "paper": "N/A - Generic wrapper",
        "authors": "N/A",
        "params": ["model_name_or_path", "device", "dtype", "max_length"],
        "example": "mgteval-cli detect --detector pretrained --data test.jsonl --model1 roberta-base-openai-detector",
    },
    # Fine-tuned detectors
    "greater": {
        "name": "GREATER",
        "type": "Fine-tuned",
        "description": "Adversarial training for robust MGT detection",
        "paper": "Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training",
        "authors": "Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu",
        "params": ["model_name", "graph_type", "device"],
        "example": "mgteval-cli train --detector greater --data train.jsonl --model1 roberta-base",
    },
    "detective": {
        "name": "DeTeCtive",
        "type": "Fine-tuned",
        "description": "Multi-level contrastive learning for AI-generated text detection",
        "paper": "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning",
        "authors": "Xun Guo, Shan Zhang, Yongxin He, Ting Zhang, Wanquan Feng, Haibin Huang, Chongyang Ma",
        "params": ["embedding_model_name", "num_neighbors", "device"],
        "example": "mgteval-cli train --detector detective --data train.jsonl --model1 simcse-roberta-base",
    },
    "coco": {
        "name": "CoCo",
        "type": "Fine-tuned",
        "description": "Coherence-enhanced contrastive learning for low-resource MGT detection",
        "paper": "CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning",
        "authors": "Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, Chao Shen",
        "params": ["model_name", "graph_config", "device"],
        "example": "mgteval-cli train --detector coco --data train.jsonl --model1 roberta-base",
    },
    "longformer": {
        "name": "Longformer",
        "type": "Fine-tuned",
        "description": "Long-document transformer classifier using Longformer backbone",
        "paper": "Longformer: Long-Document Transformer",
        "authors": "Beltagy et al.",
        "params": ["model_name", "max_length", "device"],
        "example": "mgteval-cli train --detector longformer --data train.jsonl --model1 allenai/longformer-base-4096",
    },
    "mpu": {
        "name": "MPU",
        "type": "Fine-tuned",
        "description": "Multiscale positive–unlabeled learning for short AI-text detection",
        "paper": "Multiscale Positive-Unlabeled Detection of AI-Generated Texts",
        "authors": "Yuchuan Tian, Hanting Chen, Xutao Wang, Zheyuan Bai, Qinghua Zhang, Ruifeng Li, Chao Xu, Yunhe Wang",
        "params": ["model_name", "device"],
        "example": "mgteval-cli train --detector mpu --data train.jsonl --model1 roberta-base",
    },
    "pecola": {
        "name": "PECOLA",
        "type": "Fine-tuned",
        "description": "Fine-tuned contrastive learning with selective perturbation",
        "paper": "Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better",
        "authors": "Shengchao Liu, Xiaoming Liu*, Yichen Wang, Zehua Cheng, Chengzhengxu Li, Yu Lan, Chao Shen",
        "params": ["model_name", "device"],
        "example": "mgteval-cli train --detector pecola --data train.jsonl --model1 roberta-base",
    },
}

# =======================
# Detector list display tweaks
# =======================

_DISPLAY_EXTRA_DETECTORS = {
    "coco",
    "greater",
    "detective",
    "longformer",
    "mpu",
}

_DISPLAY_EXCLUDE_DETECTORS = {
    "pretrained",
}

_DISPLAY_ALLOWED_TYPES = {"Metric-based", "Pretrained", "Fine-tuned"}

# =======================
# Beautiful Output Functions
# =======================

def print_banner():
    """Print a beautiful MGTEval banner."""
    if not _HAS_RICH or console is None:
        print("=" * 70)
        print(" " * 10 + "MGTEval - Machine-Generated Text Detection")
        print("=" * 70)
        return

    # ASCII art banner
    banner_art = """
    ███╗   ███╗ ██████╗ ████████╗███████╗██╗   ██╗ █████╗ ██╗
    ████╗ ████║██╔════╝ ╚══██╔══╝██╔════╝██║   ██║██╔══██╗██║
    ██╔████╔██║██║  ███╗   ██║   █████╗  ██║   ██║███████║██║
    ██║╚██╔╝██║██║   ██║   ██║   ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
    ██║ ╚═╝ ██║╚██████╔╝   ██║   ███████╗ ╚████╔╝ ██║  ██║███████╗
    ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
    """

    subtitle = Text()
    subtitle.append("Machine-Generated Text Detection Framework", style="bold cyan")
    subtitle.append("\n")
    subtitle.append("A unified toolkit for building, attacking, and evaluating MGT detectors", style="dim white")

    banner_content = Text()
    banner_content.append(banner_art, style="bold magenta")
    banner_content.append("\n")
    banner_content.append(subtitle)

    panel = Panel(
        banner_content,
        box=box.DOUBLE,
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print(panel)


def print_detector_list(detectors: List[str], verbose: bool = False):
    """
    Print a beautiful list of available detectors.

    Args:
        detectors: List of detector names
        verbose: If True, show additional metadata
    """
    det_set = {d for d in detectors if isinstance(d, str) and d.strip()}
    det_set.update(_DISPLAY_EXTRA_DETECTORS)
    det_set.difference_update(_DISPLAY_EXCLUDE_DETECTORS)

    def _det_type(det: str) -> str:
        return DETECTOR_METADATA.get(det, {}).get("type", "Unknown")

    detectors_sorted = sorted(d for d in det_set if _det_type(d) in _DISPLAY_ALLOWED_TYPES)

    if not _HAS_RICH or console is None:
        print("\n".join(detectors_sorted))
        return

    # Group detectors by type
    metric_based = []
    pretrained = []
    finetuned = []

    for det in detectors_sorted:
        meta = DETECTOR_METADATA.get(det, {})
        det_type = meta.get("type", "Unknown")

        if det_type == "Metric-based":
            metric_based.append(det)
        elif det_type == "Pretrained":
            pretrained.append(det)
        elif det_type == "Fine-tuned":
            finetuned.append(det)

    console.print()

    if verbose:
        # Title panel
        console.print(Panel.fit(
            "[bold magenta]🔍 Available Detectors - Detailed View[/bold magenta]",
            border_style="bright_blue",
            box=box.DOUBLE,
        ))
        console.print()

        # Create separate tables for each detector type
        types_to_show = [
            ("Metric-based", "📊", "bright_green"),
            ("Pretrained", "🤖", "bright_cyan"),
            ("Fine-tuned", "🎯", "bright_yellow"),
        ]

        for det_type, icon, color in types_to_show:
            # Filter detectors of this type
            type_detectors = [
                det for det in detectors_sorted
                if DETECTOR_METADATA.get(det, {}).get("type") == det_type
            ]

            if not type_detectors:
                continue

            # Create table for this type
            table = Table(
                show_header=True,
                header_style=f"bold {color}",
                border_style=color,
                box=box.ROUNDED,
                title=f"{icon} {det_type}",
                title_style=f"bold {color}",
            )
            table.add_column("#", style="cyan", justify="right", width=4)
            table.add_column("Detector", style="green", width=20)
            table.add_column("Description", style="white", width=60)

            for idx, det in enumerate(type_detectors, 1):
                meta = DETECTOR_METADATA.get(det, {})
                name = meta.get("name", det)
                desc = meta.get("description", "No description available")

                # Truncate long descriptions
                if len(desc) > 57:
                    desc = desc[:57] + "..."

                table.add_row(str(idx), name, desc)

            console.print(table)
            console.print()

        # Summary panel
        summary_lines = [
            f"[cyan]Total Detectors:[/cyan] [bold white]{len(detectors_sorted)}[/bold white]",
        ]
        if metric_based:
            summary_lines.append(f"[green]• Metric-based:[/green] {len(metric_based)}")
        if pretrained:
            summary_lines.append(f"[cyan]• Pretrained:[/cyan] {len(pretrained)}")
        if finetuned:
            summary_lines.append(f"[yellow]• Fine-tuned:[/yellow] {len(finetuned)}")
        summary_text = "\n".join(summary_lines)
        console.print(Panel(
            summary_text,
            title="[bold magenta]📊 Summary[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()
        console.print("[cyan]💡 Tip:[/cyan] Use [green]mgteval-cli info <detector>[/green] for detailed information")
    else:
        # Compact grouped display with panels
        console.print(Panel.fit(
            f"[bold cyan]Total: {len(detectors_sorted)} detectors[/bold cyan]",
            border_style="bright_blue",
            box=box.DOUBLE,
        ))
        console.print()

        # Metric-based detectors panel
        if metric_based:
            metric_lines = []
            # Create columns for better layout
            cols = 3
            for i in range(0, len(metric_based), cols):
                row_items = metric_based[i:i+cols]
                metric_lines.append("  ".join([f"[green]• {det:<20}[/green]" for det in row_items]))

            metric_content = "\n".join(metric_lines)
            console.print(Panel(
                metric_content,
                title="[bold yellow]📊 Metric-based (Training-Free)[/bold yellow]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
                box=box.ROUNDED,
            ))
            console.print()

        # Pretrained detectors panel
        if pretrained:
            pretrained_lines = [f"[green]• {det}[/green]" for det in pretrained]
            pretrained_content = "\n".join(pretrained_lines)
            console.print(Panel(
                pretrained_content,
                title="[bold yellow]🤖 Pretrained[/bold yellow]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                box=box.ROUNDED,
            ))
            console.print()

        # Fine-tuned detectors panel
        if finetuned:
            finetuned_lines = []
            # Create columns
            cols = 2
            for i in range(0, len(finetuned), cols):
                row_items = finetuned[i:i+cols]
                finetuned_lines.append("  ".join([f"[green]• {det:<20}[/green]" for det in row_items]))

            finetuned_content = "\n".join(finetuned_lines)
            console.print(Panel(
                finetuned_content,
                title="[bold yellow]🎯 Fine-tuned (Trainable)[/bold yellow]",
                title_align="left",
                border_style="yellow",
                padding=(1, 2),
                box=box.ROUNDED,
            ))
            console.print()

        # Tip panel
        console.print(Panel.fit(
            "[cyan]💡 Tip:[/cyan] Use [green]mgteval-cli list --verbose[/green] for detailed information\n"
            "[cyan]📖 Info:[/cyan] Use [green]mgteval-cli info <detector>[/green] to learn about a specific detector",
            border_style="cyan",
        ))


def print_detector_info(detector_name: str):
    """
    Print detailed information about a specific detector.

    Args:
        detector_name: Name of the detector
    """
    if not _HAS_RICH or console is None:
        meta = DETECTOR_METADATA.get(detector_name, {})
        print(f"\n=== {detector_name.upper()} ===")
        print(f"Name: {meta.get('name', 'N/A')}")
        print(f"Type: {meta.get('type', 'Unknown')}")
        print(f"Description: {meta.get('description', 'No description')}")
        print(f"Paper: {meta.get('paper', 'N/A')}")
        print(f"Authors: {meta.get('authors', 'N/A')}")
        print(f"\nKey Parameters:")
        for param in meta.get('params', []):
            print(f"  - {param}")
        print(f"\nExample:")
        print(f"  {meta.get('example', 'No example available')}")
        return

    meta = DETECTOR_METADATA.get(detector_name)

    if not meta:
        console.print(f"[red]❌ Detector '[bold]{detector_name}[/bold]' not found in metadata.[/red]")
        console.print("\n[yellow]💡 Tip:[/yellow] Run [green]mgteval-cli list[/green] to see available detectors")
        return

    # Header panel
    header = Text()
    header.append(meta.get('name', detector_name), style="bold magenta")
    header.append(f" ({meta.get('type', 'Unknown')})", style="cyan")

    console.print()
    console.print(Panel.fit(header, border_style="bright_blue", box=box.DOUBLE))
    console.print()

    # Description panel
    console.print(Panel(
        f"[white]{meta.get('description', 'No description available')}[/white]",
        title="[bold yellow]📝 Description[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Paper info panel
    paper_content = (
        f"[cyan bold]{meta.get('paper', 'N/A')}[/cyan bold]\n\n"
        f"[dim]Authors: {meta.get('authors', 'N/A')}[/dim]"
    )
    console.print(Panel(
        paper_content,
        title="[bold yellow]📄 Paper Reference[/bold yellow]",
        border_style="cyan",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    console.print()

    # Parameters panel
    params = meta.get('params', [])
    if params:
        param_lines = []
        for param in params:
            param_lines.append(f"• [green]{param}[/green]")
        param_content = "\n".join(param_lines)

        console.print(Panel(
            param_content,
            title="[bold yellow]⚙️  Key Parameters[/bold yellow]",
            border_style="green",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Example usage panel
    example = meta.get('example', None)
    if example:
        console.print(Panel(
            Syntax(example, "bash", theme="monokai", line_numbers=False, word_wrap=True),
            title="[bold yellow]💡 Example Usage[/bold yellow]",
            border_style="green",
            padding=(1, 1),
            box=box.ROUNDED,
        ))
        console.print()

    # Quick tips panel for this detector
    det_type = meta.get('type', 'Unknown')
    tips = []

    if det_type == "Metric-based":
        tips.extend([
            "• No training required - works out of the box",
            "• Requires access to language models (via --model1, --model2)",
            "• Consider calibration on dev set for better probability estimates",
        ])
    elif det_type == "Pretrained":
        tips.extend([
            "• Load directly from HuggingFace or local checkpoint",
            "• No additional training needed",
            "• May benefit from domain adaptation if your data differs from training",
        ])
    elif det_type == "Fine-tuned":
        tips.extend([
            "• Requires training on your specific dataset",
            "• Use 'mgteval-cli train' to fine-tune the model",
            "• Consider using validation set for early stopping",
        ])

    if tips:
        tips_content = "\n".join(tips)
        console.print(Panel(
            tips_content,
            title="[bold cyan]💡 Quick Tips[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()


def print_examples():
    """Print usage examples for common tasks."""
    if not _HAS_RICH or console is None:
        print("\n=== MGTEval Usage Examples ===\n")
        print("1. List all detectors:")
        print("   mgteval-cli list")
        print("\n2. Get detector info:")
        print("   mgteval-cli info binoculars")
        print("\n3. Build dataset:")
        print("   mgteval-cli build examples/build/build_dataset.yaml")
        print("\n4. Run detection:")
        print("   mgteval-cli detect examples/detect/binoculars.yaml")
        print("\n5. Calibrate detector:")
        print("   mgteval-cli calibrate --detector binoculars --data dev.jsonl --model1 gpt-neo-2.7B")
        return

    console.print()
    console.print(Panel.fit(
        "[bold magenta]MGTEval Usage Examples[/bold magenta]",
        border_style="bright_blue",
        box=box.DOUBLE,
    ))
    console.print()

    examples = [
        {
            "title": "📋 List Available Detectors",
            "icon": "📋",
            "description": "Discover all registered detection methods",
            "commands": [
                "mgteval-cli list",
                "mgteval-cli list --verbose  # Detailed view",
            ],
        },
        {
            "title": "ℹ️  Get Detector Information",
            "icon": "ℹ️",
            "description": "View comprehensive details about a specific detector",
            "commands": [
                "mgteval-cli info binoculars",
                "mgteval-cli info fastdetectgpt",
            ],
        },
        {
            "title": "🏗️  Build Dataset (Generate Machine Text)",
            "icon": "🏗️",
            "description": "Generate AI-written text from human samples",
            "commands": [
                "mgteval-cli build examples/build/build_dataset.yaml",
                "mgteval-cli build my_config.yaml --sample_k 100",
            ],
        },
        {
            "title": "⚔️  Apply Adversarial Attacks",
            "icon": "⚔️",
            "description": "Test detector robustness with text perturbations",
            "commands": [
                "mgteval-cli attack examples/attack/build_attack_dataset.yaml",
            ],
        },
        {
            "title": "🔍 Run Detection",
            "icon": "🔍",
            "description": "Evaluate detectors on your datasets",
            "commands": [
                "# Using YAML config (recommended)",
                "mgteval-cli detect examples/detect/binoculars.yaml",
                "",
                "# Using command-line arguments",
                "mgteval-cli detect --detector binoculars --data test.jsonl \\",
                "    --model1 gpt-neo-2.7B --model2 gpt-neo-2.7B-instruct \\",
                "    --batch_size 16 --out results/",
            ],
        },
        {
            "title": "🎯 Calibrate Detector",
            "icon": "🎯",
            "description": "Map raw scores to calibrated probabilities",
            "commands": [
                "mgteval-cli calibrate --detector binoculars \\",
                "    --data dev.jsonl --model1 gpt-neo-2.7B \\",
                "    --out calibration_results/binoculars.json",
            ],
        },
        {
            "title": "🏋️  Train Custom Detector",
            "icon": "🏋️",
            "description": "Fine-tune your own detection model",
            "commands": [
                "mgteval-cli train examples/train/seqcls.yaml",
                "",
                "# Or with command-line args",
                "mgteval-cli train --detector roberta-base \\",
                "    --data train.jsonl --epochs 3 --lr 2e-5 \\",
                "    --out models/my_detector",
            ],
        },
    ]

    # Print each example in its own panel
    for idx, example in enumerate(examples, 1):
        # Build command content
        cmd_lines = []
        for cmd in example['commands']:
            if cmd.startswith("#"):
                cmd_lines.append(f"[dim]{cmd}[/dim]")
            elif cmd == "":
                cmd_lines.append("")
            else:
                cmd_lines.append(f"[green]$ {cmd}[/green]")

        cmd_content = "\n".join(cmd_lines)

        # Create panel with description and commands
        panel_content = f"[dim]{example['description']}[/dim]\n\n{cmd_content}"

        console.print(Panel(
            panel_content,
            title=f"[bold cyan]{example['title']}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Tips section in a panel
    console.print(Panel(
        "[bold yellow]💡 Quick Tips[/bold yellow]\n\n"
        "• Use [cyan]--help[/cyan] with any command for more options\n"
        "• Set [cyan]HF_ENDPOINT=\"https://hf-mirror.com\"[/cyan] if HF downloads are slow\n"
        "• Use [cyan]--sample_k[/cyan] for quick testing on a subset of data\n"
        "• YAML configs are recommended for reproducible runs\n"
        "• Run [cyan]mgteval-cli troubleshoot[/cyan] for common issue solutions",
        title="[bold magenta]Helpful Tips[/bold magenta]",
        border_style="yellow",
        padding=(1, 2),
        box=box.ROUNDED,
    ))


def print_troubleshooting():
    """Print common troubleshooting tips."""
    if not _HAS_RICH or console is None:
        print("\n=== Troubleshooting ===\n")
        print("1. HuggingFace Download Issues:")
        print("   export HF_ENDPOINT=\"https://hf-mirror.com\"")
        print("\n2. GPU Out of Memory:")
        print("   Reduce --batch_size or --max_new_tokens")
        print("\n3. Tokenizer Errors:")
        print("   Try --tokenizer_strategy whitespace")
        return

    console.print()
    console.print(Panel.fit(
        "[bold red]🔧 Troubleshooting Guide[/bold red]",
        border_style="red",
        box=box.DOUBLE,
    ))
    console.print()

    issues = [
        {
            "problem": "🌐 HuggingFace Download Issues",
            "solution": "export HF_ENDPOINT=\"https://hf-mirror.com\"",
            "desc": "Use HF mirror for faster downloads (especially in China)",
            "explanation": "The official HuggingFace Hub may be slow or blocked in some regions. Using a mirror server can significantly speed up model downloads.",
        },
        {
            "problem": "💾 GPU Out of Memory",
            "solutions": [
                "Reduce batch size: --batch_size 4",
                "Reduce generation length: --max_new_tokens 128",
                "Use vLLM: --use_vllm --vllm_gpu_memory_utilization 0.7",
                "Enable FP16/BF16: --fp16 or --bf16",
            ],
            "desc": "Multiple strategies to reduce memory usage",
            "explanation": "GPU memory issues are common with large models. Try these solutions in order until the problem is resolved.",
        },
        {
            "problem": "🔤 Tokenizer Errors",
            "solution": "tokenizer_strategy: whitespace  # In YAML config",
            "desc": "Use whitespace tokenization as fallback",
            "explanation": "Some detectors may fail with complex tokenizers. Whitespace tokenization is a reliable fallback that works with any text.",
        },
        {
            "problem": "📦 Import Errors",
            "solutions": [
                "pip install torch-geometric  # For graph-based detectors (GREATER, CoCo)",
                "pip install spacy && python -m spacy download en_core_web_sm  # For NLP attacks",
                "pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"  # For synonym attacks",
            ],
            "desc": "Install optional dependencies as needed",
            "explanation": "Some detectors and attacks require additional packages. Install them only when you need those specific features.",
        },
        {
            "problem": "🐌 Slow Evaluation",
            "solutions": [
                "Use --sample_k 1000 for quick tests",
                "Increase --batch_size (if memory allows)",
                "Enable vLLM for model-based detectors: --use_vllm",
                "Disable expensive metrics: --no-save-curves",
            ],
            "desc": "Speed up evaluation runs",
            "explanation": "Evaluation can be time-consuming on large datasets. Use sampling and batching to speed things up during development.",
        },
        {
            "problem": "🔌 CUDA/PyTorch Version Mismatch",
            "solutions": [
                "Check CUDA version: nvidia-smi",
                "Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118",
                "Use CPU fallback: --device cpu",
            ],
            "desc": "Fix CUDA compatibility issues",
            "explanation": "Ensure your PyTorch version matches your CUDA installation. Visit pytorch.org for the correct installation command.",
        },
        {
            "problem": "📁 Permission Denied Errors",
            "solutions": [
                "Create output directory: mkdir -p results/",
                "Check write permissions: ls -la results/",
                "Use writable path: --out /tmp/mgteval_results",
            ],
            "desc": "Resolve file system permission issues",
            "explanation": "MGTEval needs write access to save results and checkpoints. Ensure the output directory exists and is writable.",
        },
    ]

    # Print each issue in its own panel
    for idx, issue in enumerate(issues, 1):
        # Build solution content
        if 'solution' in issue:
            solution_content = f"[green]Solution:[/green]\n[cyan]  {issue['solution']}[/cyan]"
        elif 'solutions' in issue:
            sol_lines = ["[green]Solutions:[/green]"]
            for sol in issue['solutions']:
                sol_lines.append(f"  • [cyan]{sol}[/cyan]")
            solution_content = "\n".join(sol_lines)
        else:
            solution_content = "[yellow]No specific solution available[/yellow]"

        # Build panel content
        panel_content = (
            f"[dim]{issue['desc']}[/dim]\n\n"
            f"{solution_content}\n\n"
            f"[dim italic]ℹ️  {issue['explanation']}[/dim italic]"
        )

        console.print(Panel(
            panel_content,
            title=f"[bold yellow]{issue['problem']}[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
            box=box.ROUNDED,
        ))
        console.print()

    # Additional resources panel
    console.print(Panel(
        "[bold cyan]📚 Additional Resources[/bold cyan]\n\n"
        "• [link]Documentation:[/link] Check the README and CLI_ENHANCEMENTS.md\n"
        "• [link]GitHub Issues:[/link] Report bugs at github.com/your-org/mgt_eval\n"
        "• [link]Examples:[/link] Run [cyan]mgteval-cli examples[/cyan] for usage patterns\n"
        "• [link]Detector Info:[/link] Run [cyan]mgteval-cli info <detector>[/cyan] for specific help",
        title="[bold magenta]Need More Help?[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
        box=box.ROUNDED,
    ))


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    if not os.path.exists(config_path):
        return False, [f"Config file not found: {config_path}"]

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to parse YAML: {e}"]

    if not isinstance(config, dict):
        return False, ["Config root must be a dictionary"]

    # Check for common issues
    if 'detector' in config:
        # Detect mode
        if 'data' not in config and 'dataset' not in config:
            warnings.append("Missing 'data' or 'dataset' field")

        if config.get('batch_size', 8) > 64:
            warnings.append("Large batch_size may cause OOM. Consider reducing.")

    if 'backend' in config:
        # Build/attack mode
        if config.get('backend') == 'hf' and config.get('use_vllm', False):
            warnings.append("vLLM enabled with HF backend - this is expected")

        if 'data' not in config:
            warnings.append("Missing 'data' field for input dataset")

        if 'out' not in config and 'output' not in config:
            warnings.append("Missing 'out' field for output path")

    return True, warnings


def print_config_summary(config: Dict[str, Any], config_type: str = "detect"):
    """
    Print a beautiful summary of the configuration.

    Args:
        config: Configuration dictionary
        config_type: Type of config (detect, build, train, etc.)
    """
    if not _HAS_RICH or console is None:
        print("\n=== Configuration Summary ===")
        for k, v in config.items():
            print(f"{k}: {v}")
        return

    console.print()
    console.print(Panel.fit(
        f"[bold magenta]Configuration Summary ({config_type})[/bold magenta]",
        border_style="bright_blue",
    ))
    console.print()

    # Group by categories
    if config_type == "detect":
        groups = {
            "Detector": ['detector', 'model1', 'model2', 'device', 'dtype'],
            "Dataset": ['data', 'dataset', 'sample_k', 'batch_size'],
            "Evaluation": ['threshold', 'out', 'save_curves', 'seed'],
        }
    elif config_type == "build":
        groups = {
            "Input/Output": ['data', 'out'],
            "Backend": ['backend', 'hf_model', 'use_vllm'],
            "Generation": ['max_new_tokens', 'temperature', 'top_p', 'gen_batch_size'],
            "Attacks": ['attacks_config', 'attack_types'],
        }
    elif config_type == "train":
        groups = {
            "Detector": ['detector', 'model1', 'model2'],
            "Datasets": ['dataset_train', 'dataset_val', 'dataset_test'],
            "Training": ['epochs', 'batch_size', 'lr', 'weight_decay'],
            "Output": ['out', 'output_dir'],
        }
    else:
        groups = {"Configuration": list(config.keys())}

    for group_name, keys in groups.items():
        items = []
        for key in keys:
            if key in config and config[key] is not None:
                value = config[key]
                # Truncate long values
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                items.append(f"[cyan]{key}[/cyan]: [green]{value}[/green]")

        if items:
            console.print(f"[bold yellow]{group_name}:[/bold yellow]")
            for item in items:
                console.print(f"  {item}")
            console.print()


# =======================
# Fallback for non-Rich environments
# =======================

def ensure_rich_or_warn():
    """Warn if rich is not available."""
    if not _HAS_RICH:
        print("[WARNING] 'rich' library not installed. Install with: pip install rich")
        print("[WARNING] CLI output will be plain text without colors/formatting.\n")
