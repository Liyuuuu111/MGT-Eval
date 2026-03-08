#!/usr/bin/env python3
"""
Demo script to showcase the enhanced CLI features.
Run this to see all the beautiful formatted outputs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli_helpers import (
    print_banner,
    print_detector_list,
    print_detector_info,
    print_examples,
    print_troubleshooting,
    console,
    _HAS_RICH,
)

def demo_all():
    """Run all demo functions."""
    if not _HAS_RICH:
        print("Warning: 'rich' library not installed. Install with: pip install rich")
        print("Falling back to plain text output.\n")

    # Demo 1: Banner
    print("\n" + "="*80)
    print("DEMO 1: Beautiful Banner")
    print("="*80 + "\n")
    print_banner()
    input("\nPress Enter to continue...")

    # Demo 2: List detectors (compact)
    print("\n" + "="*80)
    print("DEMO 2: List Detectors (Compact Mode)")
    print("="*80 + "\n")

    # Mock detector list
    mock_detectors = [
        "binoculars", "detectgpt", "fastdetectgpt", "gltr",
        "likelihood", "rank", "logrank", "entropy",
        "lrr", "npr", "raidar", "tocsin",
        "dnadetectllm", "dnagpt", "lastde", "lastdepp",
        "pretrained",
        "greater", "detective", "coco", "imbd", "longformer", "mpu", "pecola"
    ]

    print_detector_list(mock_detectors, verbose=False)
    input("\nPress Enter to continue...")

    # Demo 3: List detectors (verbose)
    print("\n" + "="*80)
    print("DEMO 3: List Detectors (Verbose Mode)")
    print("="*80 + "\n")
    print_detector_list(mock_detectors, verbose=True)
    input("\nPress Enter to continue...")

    # Demo 4: Detector info
    print("\n" + "="*80)
    print("DEMO 4: Detector Information (Binoculars)")
    print("="*80 + "\n")
    print_detector_info("binoculars")
    input("\nPress Enter to continue...")

    # Demo 5: Another detector info
    print("\n" + "="*80)
    print("DEMO 5: Detector Information (FastDetectGPT)")
    print("="*80 + "\n")
    print_detector_info("fastdetectgpt")
    input("\nPress Enter to continue...")

    # Demo 6: Examples
    print("\n" + "="*80)
    print("DEMO 6: Usage Examples")
    print("="*80 + "\n")
    print_examples()
    input("\nPress Enter to continue...")

    # Demo 7: Troubleshooting
    print("\n" + "="*80)
    print("DEMO 7: Troubleshooting Guide")
    print("="*80 + "\n")
    print_troubleshooting()

    # Final message
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    if console and _HAS_RICH:
        console.print("\n[green]✓[/green] All enhanced CLI features demonstrated!")
        console.print("[cyan]💡 Tip:[/cyan] Try these commands in the real CLI:")
        console.print("  • [green]mgteval-cli list[/green]")
        console.print("  • [green]mgteval-cli list --verbose[/green]")
        console.print("  • [green]mgteval-cli info binoculars[/green]")
        console.print("  • [green]mgteval-cli examples[/green]")
        console.print("  • [green]mgteval-cli troubleshoot[/green]")
    else:
        print("\n✓ All enhanced CLI features demonstrated!")
        print("💡 Tip: Try these commands in the real CLI:")
        print("  • mgteval-cli list")
        print("  • mgteval-cli list --verbose")
        print("  • mgteval-cli info binoculars")
        print("  • mgteval-cli examples")
        print("  • mgteval-cli troubleshoot")
    print()


def demo_quick():
    """Quick demo of key features."""
    print("\n" + "="*80)
    print("QUICK DEMO: MGTEval Enhanced CLI")
    print("="*80 + "\n")

    print_banner()
    print()

    print("1. List Command (Compact)")
    print("-" * 40)
    mock_detectors = [
        "binoculars", "detectgpt", "fastdetectgpt",
        "pretrained", "greater", "detective"
    ]
    print_detector_list(mock_detectors, verbose=False)

    print("\n2. Info Command")
    print("-" * 40)
    print_detector_info("binoculars")

    print("\n" + "="*80)
    print("Quick demo complete!")
    print("Run with 'full' argument for complete demo: python demo_enhanced_cli.py full")
    print("="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        demo_all()
    else:
        demo_quick()
