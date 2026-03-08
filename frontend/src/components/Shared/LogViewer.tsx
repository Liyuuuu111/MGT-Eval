/**
 * Log Viewer Component
 */

import React, { useEffect, useRef } from 'react';
import { Tag } from 'antd';
import { LogEntry } from '../../types';

interface LogViewerProps {
  logs: LogEntry[];
  isRunning: boolean;
}

export const LogViewer: React.FC<LogViewerProps> = ({ logs, isRunning }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const inferLevel = (message: string): LogEntry['level'] => {
    const text = String(message || '');
    if (/\berror\b/i.test(text)) {
      return 'error';
    }
    if (/\bwarning\b/i.test(text)) {
      return 'warning';
    }
    return 'info';
  };

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error':
        return '#f48771';
      case 'warning':
        return '#dcdcaa';
      default:
        return '#d4d4d4';
    }
  };

  return (
    <div>
      {isRunning && (
        <Tag color="processing" style={{ marginBottom: 8 }}>
          Running...
        </Tag>
      )}

      <div
        ref={containerRef}
        style={{
          height: '600px',
          overflowY: 'auto',
          background: '#1e1e1e',
          color: '#d4d4d4',
          padding: '12px',
          fontFamily: "'Courier New', monospace",
          fontSize: '13px',
          borderRadius: '4px',
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: '#666' }}>No logs yet...</div>
        ) : (
          logs.map((log, idx) => {
            const normalizedLevel = inferLevel(log.message);
            return (
            <div
              key={idx}
              style={{
                padding: '4px 0',
                color: getLevelColor(normalizedLevel),
              }}
            >
              <span style={{ color: '#666' }}>
                [{new Date(log.timestamp).toLocaleTimeString()}]
              </span>
              {' '}
              <span style={{ color: '#4ec9b0' }}>
                {normalizedLevel.toUpperCase()}
              </span>
              {' '}
              {log.message}
            </div>
            );
          })
        )}
      </div>
    </div>
  );
};
