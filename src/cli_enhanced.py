# mgt_eval/cli_enhanced.py
"""
Enhanced CLI commands for MGTEval.
This module adds new commands (info, examples, troubleshoot) and improves existing ones.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from cli_helpers import (
    print_banner,
    print_detector_list,
    print_detector_info,
    print_examples,
    print_troubleshooting,
    validate_config_file,
    print_config_summary,
    ensure_rich_or_warn,
    console,
    _HAS_RICH,
)


def cmd_list_enhanced(args):
    """Enhanced list command with beautiful output."""
    # Import detector registry
    try:
        from detectors import ensure_all_detectors_registered, list_registered_detectors
        ensure_all_detectors_registered()
        detectors = list_registered_detectors()
    except Exception as e:
        if console and _HAS_RICH:
            console.print(f"[red]❌ Error loading detectors: {e}[/red]")
        else:
            print(f"Error loading detectors: {e}")
        sys.exit(1)

    if not detectors:
        if console and _HAS_RICH:
            console.print("[yellow]⚠️  No detectors found in registry[/yellow]")
        else:
            print("No detectors found in registry")
        return

    # Print beautiful list
    verbose = getattr(args, 'verbose', False)
    print_detector_list(detectors, verbose=verbose)


def cmd_info(args):
    """Show detailed information about a specific detector."""
    detector_name = args.detector

    if not detector_name:
        if console and _HAS_RICH:
            console.print("[red]❌ Error: Detector name is required[/red]")
            console.print("\n[yellow]Usage:[/yellow] [green]mgteval-cli info <detector_name>[/green]")
            console.print("\n[yellow]Example:[/yellow] [green]mgteval-cli info binoculars[/green]")
        else:
            print("Error: Detector name is required")
            print("Usage: mgteval-cli info <detector_name>")
        sys.exit(1)

    # Check if detector exists
    try:
        from detectors import ensure_all_detectors_registered, list_registered_detectors
        ensure_all_detectors_registered()
        available = list_registered_detectors()

        if detector_name not in available:
            if console and _HAS_RICH:
                console.print(f"[red]❌ Detector '[bold]{detector_name}[/bold]' not found[/red]")
                console.print("\n[yellow]Available detectors:[/yellow]")
                print_detector_list(available, verbose=False)
            else:
                print(f"Error: Detector '{detector_name}' not found")
                print(f"Available detectors: {', '.join(sorted(available))}")
            sys.exit(1)
    except Exception as e:
        if console and _HAS_RICH:
            console.print(f"[red]❌ Error checking detector: {e}[/red]")
        else:
            print(f"Error checking detector: {e}")
        sys.exit(1)

    # Print detailed info
    print_detector_info(detector_name)


def cmd_examples(args):
    """Show usage examples for common tasks."""
    print_examples()


def cmd_troubleshoot(args):
    """Show troubleshooting guide."""
    print_troubleshooting()


def cmd_validate(args):
    """Validate a configuration file."""
    config_path = args.config

    if not config_path:
        if console and _HAS_RICH:
            console.print("[red]❌ Error: Config path is required[/red]")
            console.print("\n[yellow]Usage:[/yellow] [green]mgteval-cli validate <config.yaml>[/green]")
        else:
            print("Error: Config path is required")
            print("Usage: mgteval-cli validate <config.yaml>")
        sys.exit(1)

    if console and _HAS_RICH:
        console.print(f"\n[cyan]🔍 Validating config:[/cyan] [green]{config_path}[/green]")
    else:
        print(f"\nValidating config: {config_path}")

    is_valid, warnings = validate_config_file(config_path)

    if not is_valid:
        if console and _HAS_RICH:
            console.print("[red]❌ Validation failed:[/red]")
            for warning in warnings:
                console.print(f"  • [red]{warning}[/red]")
        else:
            print("Validation failed:")
            for warning in warnings:
                print(f"  - {warning}")
        sys.exit(1)

    if warnings:
        if console and _HAS_RICH:
            console.print("[yellow]⚠️  Validation passed with warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • [yellow]{warning}[/yellow]")
        else:
            print("Validation passed with warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    else:
        if console and _HAS_RICH:
            console.print("[green]✅ Validation passed! No issues found.[/green]")
        else:
            print("Validation passed! No issues found.")

    # Show config summary if requested
    if getattr(args, 'show_summary', False):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Infer config type from content
            config_type = "unknown"
            if 'detector' in config:
                config_type = "detect"
            elif 'backend' in config:
                if config.get('attack_dataset_only'):
                    config_type = "attack"
                else:
                    config_type = "build"

            print_config_summary(config, config_type=config_type)
        except Exception as e:
            if console and _HAS_RICH:
                console.print(f"[yellow]⚠️  Could not load config for summary: {e}[/yellow]")
            else:
                print(f"Warning: Could not load config for summary: {e}")


def add_enhanced_commands(subparsers, yaml_cmd: Optional[str] = None):
    """
    Add enhanced commands to the CLI.

    Args:
        subparsers: argparse subparsers object
        yaml_cmd: Current YAML command (if any)
    """
    # Enhanced list command
    ap_list = subparsers.add_parser(
        "list",
        help="List available detectors with beautiful formatting",
        description="List all registered detectors in MGTEval",
    )
    ap_list.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information (type, description) for each detector"
    )
    ap_list.set_defaults(_fn=cmd_list_enhanced)

    # Info command
    ap_info = subparsers.add_parser(
        "info",
        help="Show detailed information about a detector",
        description="Display comprehensive information about a specific detector including parameters, paper reference, and usage examples",
    )
    ap_info.add_argument(
        "detector",
        help="Name of the detector (e.g., binoculars, fastdetectgpt)"
    )
    ap_info.set_defaults(_fn=cmd_info)

    # Examples command
    ap_examples = subparsers.add_parser(
        "examples",
        help="Show usage examples for common tasks",
        description="Display comprehensive usage examples for building datasets, running detection, training models, etc.",
    )
    ap_examples.set_defaults(_fn=cmd_examples)

    # Troubleshoot command
    ap_troubleshoot = subparsers.add_parser(
        "troubleshoot",
        help="Show troubleshooting guide",
        description="Display solutions for common issues like HF download problems, GPU OOM, tokenizer errors, etc.",
        aliases=['debug', 'help-me'],
    )
    ap_troubleshoot.set_defaults(_fn=cmd_troubleshoot)

    # Validate command
    ap_validate = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
        description="Check if a YAML configuration file is valid and show warnings for potential issues",
    )
    ap_validate.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    ap_validate.add_argument(
        "-s", "--show-summary",
        action="store_true",
        help="Show configuration summary after validation"
    )
    ap_validate.set_defaults(_fn=cmd_validate)


def create_enhanced_main(original_main_func):
    """
    Wrapper to enhance the original main function with banner and better UX.

    Args:
        original_main_func: The original main() function from cli.py

    Returns:
        Enhanced main function
    """
    def enhanced_main(argv=None):
        # Check for --version flag
        if argv and '--version' in argv:
            print("MGTEval v1.0.0")
            return

        # Check for --banner flag (show banner)
        show_banner = False
        if argv and '--banner' in argv:
            show_banner = True
            argv = [arg for arg in argv if arg != '--banner']

        # Ensure rich is available (or warn)
        ensure_rich_or_warn()

        # Show banner if requested or if it's the first time
        if show_banner:
            print_banner()

        # Call original main
        try:
            original_main_func(argv)
        except KeyboardInterrupt:
            if console and _HAS_RICH:
                console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
            else:
                print("\nOperation cancelled by user")
            sys.exit(130)
        except Exception as e:
            if console and _HAS_RICH:
                console.print(f"\n[red]❌ Error: {e}[/red]")
            else:
                print(f"\nError: {e}")
            sys.exit(1)

    return enhanced_main


def patch_cli_with_enhancements():
    """
    Patch the main CLI module with enhancements.
    This function should be called from cli.py to add new commands.
    """
    import sys
    import argparse

    # Get the current module
    cli_module = sys.modules.get('mgt_eval.cli') or sys.modules.get('cli')
    if not cli_module:
        return

    # Store original functions
    original_cmd_list = getattr(cli_module, 'cmd_list', None)
    original_main = getattr(cli_module, 'main', None)

    # Replace cmd_list with enhanced version
    if original_cmd_list:
        cli_module.cmd_list = cmd_list_enhanced

    # Enhance main if available
    if original_main:
        cli_module.main = create_enhanced_main(original_main)


# Standalone execution for testing
if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Import it from cli.py to add enhanced commands.")
