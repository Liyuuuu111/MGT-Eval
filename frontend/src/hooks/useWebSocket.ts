/**
 * WebSocket hook for log streaming
 */

import { useEffect, useRef } from 'react';
import { useStore } from '../store';
import { Section } from '../types';

interface UseWebSocketOptions {
  jobId: string | null;
  section: Section;
}

export const useWebSocket = ({ jobId, section }: UseWebSocketOptions) => {
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
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/logs/${jobId}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connected for job ${jobId}`);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'log') {
        addLog({
          level: data.level,
          message: data.message,
          timestamp: data.timestamp,
        });
      } else if (data.type === 'complete') {
        stopJob();
        if (data.exit_code === -1 || data.status === 'cancelled') {
          addLog({
            level: 'warning',
            message: 'Job cancelled by user',
            timestamp: new Date().toISOString(),
          });
        } else if (data.status === 'error') {
          addLog({
            level: 'error',
            message: `Process failed with exit code ${data.exit_code}`,
            timestamp: new Date().toISOString(),
          });
        } else {
          addLog({
            level: 'info',
            message: 'Process completed successfully',
            timestamp: new Date().toISOString(),
          });
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      addLog({
        level: 'error',
        message: 'WebSocket connection error',
        timestamp: new Date().toISOString(),
      });
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [jobId, addLog, stopJob]);

  return wsRef;
};
