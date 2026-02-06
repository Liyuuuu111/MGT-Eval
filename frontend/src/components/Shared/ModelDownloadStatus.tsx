import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, Tag, Progress, Space } from 'antd';
import { CloudDownloadOutlined, LoadingOutlined } from '@ant-design/icons';
import { HFDownloadStatus, LogEntry } from '../../types';
import api from '../../services/api';

interface ModelDownloadStatusProps {
  logs: LogEntry[];
  isRunning: boolean;
}

interface DownloadDisplayStatus {
  active: boolean;
  percent: number | null;
  message: string | null;
  idleHint: boolean;
  indeterminate?: boolean;
  modelName?: string | null;
  downloadCount?: number;
  source?: 'cache' | 'logs';
}

export const ModelDownloadStatus: React.FC<ModelDownloadStatusProps> = ({ logs, isRunning }) => {
  const [nowTs, setNowTs] = useState<number>(Date.now());
  const [hfStatus, setHfStatus] = useState<HFDownloadStatus | null>(null);
  const [hfStatusError, setHfStatusError] = useState<boolean>(false);
  const [speedBytesPerSec, setSpeedBytesPerSec] = useState<number | null>(null);
  const previousSnapshotRef = useRef<{ tsMs: number; bytes: number } | null>(null);

  const formatBytes = (bytes: number): string => {
    if (!Number.isFinite(bytes)) {
      return '0 B';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = Math.max(0, bytes);
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
  };
  const formatDuration = (seconds: number): string => {
    if (!Number.isFinite(seconds) || seconds < 0) {
      return '-';
    }
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    if (h > 0) {
      return `${h}h ${m}m ${rem}s`;
    }
    if (m > 0) {
      return `${m}m ${rem}s`;
    }
    return `${rem}s`;
  };

  useEffect(() => {
    if (!isRunning) {
      return;
    }
    const intervalId = window.setInterval(() => {
      setNowTs(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [isRunning]);

  useEffect(() => {
    if (!hfStatus || !hfStatus.active) {
      previousSnapshotRef.current = null;
      setSpeedBytesPerSec(null);
      return;
    }
    const tsMs = Date.parse(hfStatus.timestamp || '');
    const currentTs = Number.isNaN(tsMs) ? Date.now() : tsMs;
    const currentBytes = Number(hfStatus.total_downloaded_bytes || 0);
    const previous = previousSnapshotRef.current;
    if (previous && currentTs > previous.tsMs) {
      const deltaBytes = currentBytes - previous.bytes;
      const deltaSeconds = (currentTs - previous.tsMs) / 1000;
      if (deltaSeconds > 0 && deltaBytes >= 0) {
        setSpeedBytesPerSec(deltaBytes / deltaSeconds);
      }
    }
    previousSnapshotRef.current = { tsMs: currentTs, bytes: currentBytes };
  }, [hfStatus]);

  useEffect(() => {
    if (!isRunning) {
      setHfStatus(null);
      return;
    }

    let isMounted = true;
    const fetchStatus = async () => {
      try {
        const data = await api.getHfDownloads();
        if (!isMounted) return;
        setHfStatus(data);
        setHfStatusError(false);
      } catch (error) {
        if (!isMounted) return;
        setHfStatusError(true);
      }
    };

    fetchStatus();
    const intervalId = window.setInterval(fetchStatus, 2000);
    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [isRunning]);

  const logStatus = useMemo<DownloadDisplayStatus>(() => {
    let message: string | null = null;
    let percent: number | null = null;
    let active = false;
    let downloadCount = 0;
    let modelName: string | null = null;

    // Enhanced detection keywords - more comprehensive
    const downloadKeywords = [
      'download',
      'downloading',
      'hf_hub',
      'huggingface',
      'snapshot',
      'fetch',
      'fetching',
      'loading model',
      'load model',
      'model loading',
      'checkpoint',
      'tokenizer',
      'config.json',
      'pytorch_model',
      'model.safetensors',
      '.bin',
      'resolving',
      'from cache',
      'cache-miss',
      'hub.download',
      'transformers',
      'pretrained',
      'safetensors',
      'sharded',
    ];

    // Model initialization keywords
    const initKeywords = [
      'initializing',
      'loading weights',
      'loading checkpoint',
      'device map',
      'quantization',
      'from_pretrained',
      'loading model',
    ];

    // Scan recent logs for download activity (check last 100 logs)
    const recentLogs = logs.slice(Math.max(0, logs.length - 100));

    for (let i = recentLogs.length - 1; i >= 0; i -= 1) {
      const entry = recentLogs[i];
      if (!entry) continue;

      const msg = entry.message || '';
      const lower = msg.toLowerCase();

      // Check if this is a download-related line
      const isDownloadLine = downloadKeywords.some(keyword => lower.includes(keyword));
      const isInitLine = initKeywords.some(keyword => lower.includes(keyword));

      if (isDownloadLine || isInitLine) {
        downloadCount++;

        // Capture the most relevant message
        if (!message || isDownloadLine) {
          message = msg;
        }

        // Try to extract model name
        if (!modelName) {
          // Look for model names like "gpt2", "roberta-base", etc.
          const modelMatch = msg.match(/([a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+|(?:gpt|roberta|bert|llama|mistral|qwen)[a-zA-Z0-9_-]*)/i);
          if (modelMatch) {
            modelName = modelMatch[0];
          }
        }

        // Try to extract percentage
        const percentMatch = msg.match(/(\d{1,3})%/);
        if (percentMatch && percent === null) {
          const parsed = Number(percentMatch[1]);
          if (!Number.isNaN(parsed) && parsed <= 100) {
            percent = Math.min(100, Math.max(0, parsed));
          }
        }

        // Try to extract progress from patterns like "5/10", "3 of 5"
        const fractionMatch = msg.match(/(\d+)\s*(?:\/|of)\s*(\d+)/);
        if (fractionMatch && !percent) {
          const current = Number(fractionMatch[1]);
          const total = Number(fractionMatch[2]);
          if (total > 0 && current <= total) {
            percent = Math.round((current / total) * 100);
          }
        }

        // Try to extract bytes information (e.g., "50MB/100MB")
        const bytesMatch = msg.match(/(\d+(?:\.\d+)?)\s*(KB|MB|GB)?\s*\/\s*(\d+(?:\.\d+)?)\s*(KB|MB|GB)?/i);
        if (bytesMatch && !percent) {
          const downloaded = parseFloat(bytesMatch[1]);
          const total = parseFloat(bytesMatch[3]);
          if (total > 0 && downloaded <= total) {
            percent = Math.round((downloaded / total) * 100);
          }
        }

        // Check if download is still active (not completed)
        const completedKeywords = ['downloaded', 'complete', 'finished', 'done', 'successfully loaded'];
        const isCompleted = completedKeywords.some(keyword => lower.includes(keyword));

        if (!isCompleted && (isDownloadLine || isInitLine)) {
          active = true;
        }

        // Found enough evidence
        if (downloadCount >= 5) break;
      }
    }

    // Check for idle state
    const lastLog = logs.length > 0 ? logs[logs.length - 1] : null;
    const lastLogTime = lastLog ? Date.parse(lastLog.timestamp) : null;
    const idleSeconds = lastLogTime ? Math.max(0, (nowTs - lastLogTime) / 1000) : null;

    // Show hint if running and idle for more than 3 seconds with recent download activity
    const idleHint = isRunning && idleSeconds !== null && idleSeconds >= 3 && idleSeconds < 30 && downloadCount > 0;

    // If we detected download activity and still running, show as active
    const shouldShowActive = (active || (isRunning && downloadCount > 0 && idleSeconds !== null && idleSeconds < 30));

    // If no explicit percent but activity detected, show indeterminate progress
    if (shouldShowActive && percent === null) {
      percent = 0; // Will show as indeterminate/active progress
    }

    return {
      active: shouldShowActive,
      percent,
      message,
      idleHint,
      downloadCount,
      modelName,
    };
  }, [logs, isRunning, nowTs]);

  const backendStatus = useMemo<DownloadDisplayStatus | null>(() => {
    if (!hfStatus || !hfStatus.active) {
      return null;
    }

    let percent: number | null = null;
    if (hfStatus.total_expected_bytes && hfStatus.total_expected_bytes > 0) {
      percent = Math.round(
        (hfStatus.total_downloaded_bytes / hfStatus.total_expected_bytes) * 100
      );
      percent = Math.min(100, Math.max(0, percent));
    } else {
      const knownTotals = hfStatus.downloads.filter(
        (d) => typeof d.total_bytes === 'number' && (d.total_bytes || 0) > 0
      );
      if (knownTotals.length > 0) {
        const knownDownloaded = knownTotals.reduce(
          (sum, d) => sum + Math.min(Number(d.size_bytes || 0), Number(d.total_bytes || 0)),
          0
        );
        const knownExpected = knownTotals.reduce((sum, d) => sum + Number(d.total_bytes || 0), 0);
        percent = knownExpected > 0 ? Math.round((knownDownloaded / knownExpected) * 100) : null;
      } else {
        const percents = hfStatus.downloads
          .map((d) => d.percent)
          .filter((p): p is number => typeof p === 'number' && p >= 0);
        if (percents.length > 0) {
        percent = Math.round(
          percents.reduce((sum, p) => sum + p, 0) / percents.length
        );
          percent = Math.min(100, Math.max(0, percent));
        }
      }
    }

    const models = Array.from(
      new Set(
        hfStatus.downloads
          .map((d) => d.model)
          .filter((m): m is string => !!m)
      )
    );

    let message = '';
    if (models.length > 0) {
      const shown = models.slice(0, 3);
      message = `Downloading ${shown.join(', ')}`;
      if (models.length > shown.length) {
        message += ` +${models.length - shown.length} more`;
      }
    } else {
      const count = hfStatus.downloads.length;
      message = `Downloading ${count} file${count === 1 ? '' : 's'} (cache scan)`;
    }

    const downloadedText = formatBytes(hfStatus.total_downloaded_bytes);
    const expectedText = hfStatus.total_expected_bytes
      ? ` / ${formatBytes(hfStatus.total_expected_bytes)}`
      : '';
    const speedText = speedBytesPerSec && speedBytesPerSec > 0
      ? `${formatBytes(speedBytesPerSec)}/s`
      : null;
    const etaText = (
      speedBytesPerSec &&
      speedBytesPerSec > 0 &&
      hfStatus.total_expected_bytes &&
      hfStatus.total_expected_bytes > hfStatus.total_downloaded_bytes
    )
      ? formatDuration(
        (hfStatus.total_expected_bytes - hfStatus.total_downloaded_bytes) / speedBytesPerSec!
      )
      : null;

    message = `${message} — ${downloadedText}${expectedText}`;
    if (speedText) {
      message += ` — ${speedText}`;
    }
    if (etaText) {
      message += ` — ETA ${etaText}`;
    }

    return {
      active: true,
      percent,
      message,
      idleHint: false,
      indeterminate: percent === null,
      source: 'cache' as const,
    };
  }, [hfStatus, speedBytesPerSec]);

  const downloadStatus = useMemo<DownloadDisplayStatus>(() => {
    if (backendStatus?.active) {
      return backendStatus;
    }
    return logStatus;
  }, [backendStatus, logStatus]);

  if (!downloadStatus.active && !downloadStatus.idleHint) {
    return null;
  }

  // Format message for display
  const displayMessage = downloadStatus.message ||
    (logStatus.modelName ? `Loading ${logStatus.modelName}...` : 'Downloading model or initializing...') ||
    (downloadStatus.idleHint ? 'Model download or initialization in progress...' : 'Downloading model...');

  const animatedPercent = ((nowTs / 1000) * 12) % 100;
  const progressPercent = downloadStatus.percent ?? animatedPercent;

  return (
    <Card
      size="small"
      style={{
        marginBottom: 16,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        border: 'none',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {downloadStatus.active ? (
            <LoadingOutlined style={{ fontSize: 16, color: '#fff' }} spin />
          ) : (
            <CloudDownloadOutlined style={{ fontSize: 16, color: '#fff' }} />
          )}
          <Tag color="blue" style={{ margin: 0 }}>Model Download</Tag>
          <span style={{ color: '#fff', fontSize: 13, flex: 1, whiteSpace: 'normal', lineHeight: 1.4 }}>
            {displayMessage}
          </span>
        </div>
        {downloadStatus.active && (
          <Progress
            percent={progressPercent}
            status="active"
            showInfo={downloadStatus.percent !== null}
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
            trailColor="rgba(255, 255, 255, 0.3)"
          />
        )}
        {hfStatusError && (
          <span style={{ color: '#fff', fontSize: 11 }}>
            Unable to query cache status. Falling back to log detection.
          </span>
        )}
      </Space>
    </Card>
  );
};
