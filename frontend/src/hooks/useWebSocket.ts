/**
 * WebSocket hook for log streaming
 */

import { useEffect, useRef } from 'react';
import { useStore } from '../store';
import { Section } from '../types';
import { getWebSocketLogsUrl } from '../services/api';

interface UseWebSocketOptions {
  jobId: string | null;
  section: Section;
  isRunning: boolean;
}

const resetSocketHandlers = (socket: WebSocket) => {
  socket.onopen = null;
  socket.onmessage = null;
  socket.onerror = null;
  socket.onclose = null;
};

const closeSocketIfActive = (socket: WebSocket | null) => {
  if (!socket) {
    return;
  }
  if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
    resetSocketHandlers(socket);
    socket.close();
  }
};

const inferLogLevelFromMessage = (message: unknown): 'info' | 'warning' | 'error' => {
  const text = String(message || '');
  if (/\berror\b/i.test(text)) {
    return 'error';
  }
  if (/\bwarning\b/i.test(text)) {
    return 'warning';
  }
  return 'info';
};

export const useWebSocket = ({ jobId, section, isRunning }: UseWebSocketOptions) => {
  const wsRef = useRef<WebSocket | null>(null);

  const addLog = useStore((state) => {
    switch (section) {
      case 'build':
        return state.addBuildLog;
      case 'attack':
        return state.addAttackLog;
      case 'train':
        return state.addTrainLog;
      case 'detect':
        return state.addDetectLog;
      case 'demo':
        return state.addDemoLog;
    }
  });

  const stopJob = useStore((state) => {
    switch (section) {
      case 'build':
        return state.stopBuild;
      case 'attack':
        return state.stopAttack;
      case 'train':
        return state.stopTrain;
      case 'detect':
        return state.stopDetect;
      case 'demo':
        return state.stopDemo;
    }
  });

  useEffect(() => {
    if (!jobId || !isRunning) {
      closeSocketIfActive(wsRef.current);
      wsRef.current = null;
      return;
    }

    closeSocketIfActive(wsRef.current);
    wsRef.current = null;

    const ws = new WebSocket(getWebSocketLogsUrl(jobId));
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connected for job ${jobId}`);
    };

    ws.onmessage = (event) => {
      let data: any;
      try {
        data = JSON.parse(event.data);
      } catch (error) {
        console.warn('Ignoring malformed WebSocket payload:', error);
        return;
      }

      if (data.type === 'log') {
        const normalizedLevel = inferLogLevelFromMessage(data.message);
        addLog({
          level: normalizedLevel,
          message: data.message,
          timestamp: data.timestamp,
        });
      } else if (data.type === 'complete') {
        stopJob();
        if (data.exit_code === -1 || data.status === 'cancelled') {
          const messageText = 'WARNING: Job cancelled by user';
          addLog({
            level: inferLogLevelFromMessage(messageText),
            message: messageText,
            timestamp: new Date().toISOString(),
          });
        } else if (data.status === 'error') {
          const messageText = `ERROR: Process failed with exit code ${data.exit_code}`;
          addLog({
            level: inferLogLevelFromMessage(messageText),
            message: messageText,
            timestamp: new Date().toISOString(),
          });
        } else {
          const messageText = 'Process completed successfully';
          addLog({
            level: inferLogLevelFromMessage(messageText),
            message: messageText,
            timestamp: new Date().toISOString(),
          });
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      const messageText = 'ERROR: WebSocket connection error';
      addLog({
        level: inferLogLevelFromMessage(messageText),
        message: messageText,
        timestamp: new Date().toISOString(),
      });
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
        resetSocketHandlers(ws);
        ws.close();
      }
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [jobId, section, isRunning, addLog, stopJob]);

  return wsRef;
};
