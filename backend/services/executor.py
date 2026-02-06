"""Command Executor for running CLI commands"""

import asyncio
import sys
import os
import re
from typing import Dict, Any, Callable, Awaitable, Optional
from pathlib import Path
from .yaml_service import YAMLService


LogCallback = Callable[[str, str, str], Awaitable[None]]


class CommandExecutor:
    """Executes CLI commands and streams output"""

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            # Auto-detect project root (backend is in project_root/backend/)
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.yaml_service = YAMLService(str(project_root))

    async def execute_command(
        self,
        job_id: str,
        command: str,
        config: Dict[str, Any],
        log_callback: LogCallback
    ) -> tuple[bool, int]:
        """
        Execute a CLI command

        Args:
            job_id: Job ID
            command: Command type ('build', 'attack', 'train', 'detect')
            config: Configuration dictionary
            log_callback: Async callback function(job_id, message, level)

        Returns:
            Tuple of (success, exit_code)
        """
        temp_yaml: Optional[Path] = None
        try:
            # Save config to temporary YAML file
            temp_yaml = self.yaml_service.save_temp_yaml(config, prefix=command)
            await log_callback(job_id, f"Configuration saved to {temp_yaml}", "info")

            # Build CLI command
            cmd_map = {
                "build": "build",
                "attack": "attack",
                "train": "train",
                "detect": "detect",
            }

            cli_command = cmd_map.get(command)
            if not cli_command:
                raise ValueError(f"Unknown command: {command}")

            # Construct command
            cmd = [
                sys.executable,
                "-m", "mgt_eval.cli",
                cli_command,
                str(temp_yaml)
            ]

            await log_callback(job_id, f"Executing: {' '.join(cmd)}", "info")

            # Set up environment with GPU selection
            env = os.environ.copy()
            hf_endpoint = config.get("hf_endpoint") if isinstance(config, dict) else None
            if hf_endpoint is not None:
                endpoint = str(hf_endpoint).strip()
                if endpoint:
                    env["HF_ENDPOINT"] = endpoint
                    await log_callback(job_id, f"Using HF endpoint: {endpoint}", "info")
                else:
                    env.pop("HF_ENDPOINT", None)

            # Set HF Token if provided
            hf_token = config.get("hf_token") if isinstance(config, dict) else None
            if hf_token is not None:
                token = str(hf_token).strip()
                if token:
                    env["HF_TOKEN"] = token
                    await log_callback(job_id, "HF Token configured (hidden for security)", "info")
                else:
                    env.pop("HF_TOKEN", None)

            if "gpu_ids" in config and config["gpu_ids"] is not None:
                gpu_ids = config["gpu_ids"]
                if isinstance(gpu_ids, list):
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
                else:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
                await log_callback(job_id, f"Using GPUs: {env['CUDA_VISIBLE_DEVICES']}", "info")

            # Execute command
            success, exit_code = await self._run_subprocess(job_id, cmd, log_callback, env)

            # Clean up temp file
            try:
                temp_yaml.unlink()
            except Exception:
                pass

            return success, exit_code

        except asyncio.CancelledError:
            if temp_yaml is not None:
                try:
                    temp_yaml.unlink()
                except Exception:
                    pass
            raise
        except Exception as e:
            await log_callback(job_id, f"Error: {str(e)}", "error")
            return False, 1

    async def _run_subprocess(
        self,
        job_id: str,
        cmd: list[str],
        log_callback: LogCallback,
        env: Optional[Dict[str, str]] = None
    ) -> tuple[bool, int]:
        """
        Run subprocess and stream output

        Args:
            job_id: Job ID
            cmd: Command list
            log_callback: Log callback function
            env: Environment variables

        Returns:
            Tuple of (success, exit_code)
        """
        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
                env=env
            )

            ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            control_chars = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

            def sanitize_message(raw: str) -> str:
                no_ansi = ansi_escape.sub("", raw)
                cleaned = control_chars.sub("", no_ansi)
                return cleaned.strip()

            def split_messages(buffer: str) -> tuple[list[str], str]:
                """
                Split stream text by both newline and carriage-return boundaries.
                This preserves tqdm/HF progress updates that are usually emitted with '\r'.
                """
                messages: list[str] = []
                start = 0
                idx = 0
                length = len(buffer)

                while idx < length:
                    ch = buffer[idx]
                    if ch == "\n" or ch == "\r":
                        if idx > start:
                            messages.append(buffer[start:idx])
                        if ch == "\r" and idx + 1 < length and buffer[idx + 1] == "\n":
                            idx += 1
                        start = idx + 1
                    idx += 1

                remainder = buffer[start:]
                return messages, remainder

            # Stream stdout and stderr concurrently
            async def stream_output(stream, level):
                pending = ""
                last_message = ""
                while True:
                    chunk = await stream.read(2048)
                    if not chunk:
                        break
                    pending += chunk.decode("utf-8", errors="replace")
                    raw_messages, pending = split_messages(pending)
                    for raw_message in raw_messages:
                        message = sanitize_message(raw_message)
                        if not message:
                            continue
                        if message == last_message:
                            continue
                        await log_callback(job_id, message, level)
                        last_message = message

                # Flush final non-delimited tail
                if pending:
                    message = sanitize_message(pending)
                    if message:
                        if message != last_message:
                            await log_callback(job_id, message, level)
                            last_message = message

            # Run both streams concurrently
            await asyncio.gather(
                stream_output(process.stdout, "info"),
                stream_output(process.stderr, "error")
            )

            # Wait for process to complete
            exit_code = await process.wait()

            success = exit_code == 0
            if success:
                await log_callback(job_id, "Process completed successfully", "info")
            else:
                await log_callback(job_id, f"Process failed with exit code {exit_code}", "error")

            return success, exit_code

        except asyncio.CancelledError:
            # Handle job cancellation
            await log_callback(job_id, "Job cancelled by user", "warning")
            if process is not None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
            raise

        except Exception as e:
            await log_callback(job_id, f"Subprocess error: {str(e)}", "error")
            return False, 1


# Global instance
command_executor = CommandExecutor()
