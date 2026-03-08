"""Attack API endpoints"""

from copy import deepcopy
import asyncio

from fastapi import APIRouter, HTTPException
from backend.models.requests import ExecuteRequest, ValidateRequest
from backend.models.responses import ExecuteResponse, ValidateResponse, TemplateResponse, AttacksResponse
from backend.services.yaml_service import YAMLService
from backend.services.file_lifecycle_service import file_lifecycle_service
from backend.services.process_manager import process_manager
from backend.services.executor import command_executor
from backend.api.websocket.logs import manager as ws_manager

router = APIRouter()
yaml_service = YAMLService()


@router.get("/template", response_model=TemplateResponse)
async def get_attack_template():
    """Get the attack dataset template"""
    try:
        template = yaml_service.load_template("attack")
        return TemplateResponse(template=template)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-attacks", response_model=AttacksResponse)
async def get_all_attacks():
    """Get all available attack types"""
    try:
        attacks = yaml_service.load_attacks_json()
        return AttacksResponse(attacks=attacks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidateResponse)
async def validate_attack_config(request: ValidateRequest):
    """Validate attack configuration"""
    valid, errors = yaml_service.validate_config("attack", request.config)
    return ValidateResponse(valid=valid, errors=errors)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_attack(request: ExecuteRequest):
    """Execute attack command"""
    try:
        config = deepcopy(request.config or {})
        config.setdefault("keep_attack_aux_files", False)

        # Validate config
        valid, errors = yaml_service.validate_config("attack", config)
        if not valid:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {', '.join(errors)}")

        # Create job
        job_id = process_manager.create_job("attack", config)
        effective_config = file_lifecycle_service.prepare_job_config("attack", job_id, config)
        process_manager.update_job_config(job_id, effective_config)

        # Define log callback
        async def log_callback(job_id: str, message: str, level: str):
            await ws_manager.send_log(job_id, message, level)

        # Create task
        async def run_job():
            success = False
            exit_code = 1
            cancelled = False
            finalized = False
            try:
                success, exit_code = await command_executor.execute_command(
                    job_id, "attack", effective_config, log_callback
                )
                file_lifecycle_service.finalize_job(
                    command="attack",
                    job_id=job_id,
                    config=effective_config,
                    success=bool(success),
                )
                finalized = True
                process_manager.complete_job(job_id, success, exit_code=exit_code)
                await ws_manager.send_completion(job_id, success, exit_code)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as e:
                process_manager.complete_job(job_id, False, error_message=str(e))
                await ws_manager.send_log(job_id, f"Job failed: {str(e)}", "error")
                await ws_manager.send_completion(job_id, False, 1)
            finally:
                if not finalized:
                    try:
                        file_lifecycle_service.finalize_job(
                            command="attack",
                            job_id=job_id,
                            config=effective_config,
                            success=bool(success and (not cancelled)),
                        )
                    except Exception as finalize_error:
                        await ws_manager.send_log(
                            job_id,
                            f"[lifecycle] finalize warning: {finalize_error}",
                            "warning",
                        )

        task = asyncio.create_task(run_job())
        process_manager.start_job(job_id, task)

        return ExecuteResponse(job_id=job_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
