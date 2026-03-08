"""Process Manager for handling long-running jobs"""

import uuid
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job information"""
    id: str
    command: str  # 'build', 'attack', 'train', 'detect'
    status: JobStatus
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None


class ProcessManager:
    """Manages long-running CLI jobs"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, command: str, config: Dict[str, Any]) -> str:
        """
        Create a new job

        Args:
            command: One of 'build', 'attack', 'train', 'detect'
            config: Job configuration

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            command=command,
            status=JobStatus.PENDING,
            config=config,
            created_at=datetime.utcnow().isoformat()
        )
        self.jobs[job_id] = job
        return job_id

    def update_job_config(self, job_id: str, config: Dict[str, Any]) -> None:
        """Update job config after runtime rewrites."""
        if job_id not in self.jobs:
            return
        self.jobs[job_id].config = config

    def start_job(self, job_id: str, task: asyncio.Task) -> None:
        """
        Mark job as started and store the task

        Args:
            job_id: Job ID
            task: Asyncio task running the job
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        self.jobs[job_id].status = JobStatus.RUNNING
        self.jobs[job_id].started_at = datetime.utcnow().isoformat()
        self.tasks[job_id] = task

    def complete_job(self, job_id: str, success: bool = True, error_message: Optional[str] = None, exit_code: int = 0) -> None:
        """
        Mark job as completed or failed

        Args:
            job_id: Job ID
            success: Whether job succeeded
            error_message: Error message if failed
            exit_code: Exit code
        """
        if job_id not in self.jobs:
            return

        self.jobs[job_id].status = JobStatus.COMPLETED if success else JobStatus.FAILED
        self.jobs[job_id].completed_at = datetime.utcnow().isoformat()
        self.jobs[job_id].error_message = error_message
        self.jobs[job_id].exit_code = exit_code

        # Clean up task
        if job_id in self.tasks:
            del self.tasks[job_id]

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a running job

        Args:
            job_id: Job ID
        """
        if job_id in self.tasks:
            self.tasks[job_id].cancel()
            del self.tasks[job_id]

        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.CANCELLED
            self.jobs[job_id].completed_at = datetime.utcnow().isoformat()
            self.jobs[job_id].exit_code = -1

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job information

        Args:
            job_id: Job ID

        Returns:
            Job object or None if not found
        """
        return self.jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get job status

        Args:
            job_id: Job ID

        Returns:
            JobStatus or None if not found
        """
        job = self.jobs.get(job_id)
        return job.status if job else None

    def list_jobs(self, command: Optional[str] = None) -> list[Job]:
        """
        List all jobs, optionally filtered by command

        Args:
            command: Filter by command type

        Returns:
            List of jobs
        """
        jobs = list(self.jobs.values())
        if command:
            jobs = [j for j in jobs if j.command == command]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed jobs

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs cleaned up
        """
        from datetime import timedelta

        now = datetime.utcnow()
        cutoff = now - timedelta(hours=max_age_hours)

        jobs_to_delete = []
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.completed_at:
                    completed_time = datetime.fromisoformat(job.completed_at)
                    if completed_time < cutoff:
                        jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del self.jobs[job_id]

        return len(jobs_to_delete)


# Global instance
process_manager = ProcessManager()
