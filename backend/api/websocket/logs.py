"""WebSocket log streaming"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Deque
import json
from datetime import datetime
from collections import deque


router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for log streaming"""

    def __init__(self, buffer_size: int = 500):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.message_buffer: Dict[str, Deque[dict]] = {}
        self.buffer_size = buffer_size

    async def connect(self, job_id: str, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

        # Send buffered messages (if any) to new connection
        if job_id in self.message_buffer:
            try:
                for entry in list(self.message_buffer[job_id]):
                    await websocket.send_json(entry)
            except Exception:
                self.disconnect(job_id, websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    def _buffer_message(self, job_id: str, message: dict) -> None:
        """Buffer messages so late connections still receive logs"""
        if job_id not in self.message_buffer:
            self.message_buffer[job_id] = deque(maxlen=self.buffer_size)
        self.message_buffer[job_id].append(message)

    def _log_to_console(self, job_id: str, message: dict) -> None:
        """Mirror logs to backend console for visibility"""
        msg_type = message.get("type", "log")
        if msg_type == "log":
            level = message.get("level", "info").upper()
            text = message.get("message", "")
            timestamp = message.get("timestamp", "")
            print(f"[{timestamp}] [{job_id}] {level}: {text}", flush=True)
        elif msg_type == "complete":
            status = message.get("status", "unknown").upper()
            exit_code = message.get("exit_code", 0)
            timestamp = message.get("timestamp", "")
            print(f"[{timestamp}] [{job_id}] COMPLETE: {status} (exit_code={exit_code})", flush=True)

    async def send_log(self, job_id: str, message: str, level: str = "info"):
        """
        Send log message to all connections for a job

        Args:
            job_id: Job ID
            message: Log message
            level: Log level ('info', 'warning', 'error')
        """
        log_entry = {
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._buffer_message(job_id, log_entry)
        self._log_to_console(job_id, log_entry)

        dead_connections = set()
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(log_entry)
                except Exception:
                    dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(job_id, conn)

    async def send_completion(self, job_id: str, success: bool, exit_code: int = 0):
        """
        Send completion message to all connections for a job

        Args:
            job_id: Job ID
            success: Whether job succeeded
            exit_code: Exit code
        """
        completion_message = {
            "type": "complete",
            "status": "success" if success else "error",
            "exit_code": exit_code,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._buffer_message(job_id, completion_message)
        self._log_to_console(job_id, completion_message)

        dead_connections = set()
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(completion_message)
                except Exception:
                    dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(job_id, conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/logs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for streaming logs

    Args:
        websocket: WebSocket connection
        job_id: Job ID to stream logs for
    """
    await manager.connect(job_id, websocket)

    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages (e.g., ping/pong, cancel requests)
            data = await websocket.receive_text()

            # Handle client messages if needed
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
    except Exception as e:
        print(f"WebSocket error for job {job_id}: {e}")
        manager.disconnect(job_id, websocket)
