import React, { useEffect, useMemo, useState } from 'react';
import { Alert, Card, Col, Progress, Row, Tag, Typography } from 'antd';
import api from '../../services/api';
import { SystemMonitorResponse } from '../../types';

const formatGb = (value: number): string => `${value.toFixed(2)} GB`;

const gpuMemoryPercent = (usedMb: number, totalMb: number): number => {
  if (!Number.isFinite(totalMb) || totalMb <= 0) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round((usedMb / totalMb) * 100)));
};

const temperatureColor = (temp: number): string => {
  if (temp >= 85) return 'red';
  if (temp >= 70) return 'orange';
  return 'green';
};

export const SystemMonitorPanel: React.FC = () => {
  const [monitor, setMonitor] = useState<SystemMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchMonitor = async () => {
      try {
        const data = await api.getSystemMonitor();
        if (!mounted) {
          return;
        }
        setMonitor(data);
        setError(null);
      } catch (e: any) {
        if (!mounted) {
          return;
        }
        setError(e?.response?.data?.detail || 'Failed to load system monitor.');
      }
    };

    fetchMonitor();
    const intervalId = window.setInterval(fetchMonitor, 2000);
    return () => {
      mounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const gpuCards = useMemo(() => {
    if (!monitor?.gpus || monitor.gpus.length === 0) {
      return (
        <Alert
          type="info"
          showIcon
          message="No GPU detected"
          description="nvidia-smi returned no active GPUs."
        />
      );
    }

    return (
      <Row gutter={[12, 12]}>
        {monitor.gpus.map((gpu) => (
          <Col key={`${gpu.index}-${gpu.name}`} span={12}>
            <Card size="small" bodyStyle={{ padding: 10 }}>
              <Typography.Text strong>{`GPU ${gpu.index}: ${gpu.name}`}</Typography.Text>
              <div style={{ marginTop: 8 }}>
                <Typography.Text style={{ fontSize: 12 }}>
                  Utilization: {gpu.utilization.toFixed(1)}%
                </Typography.Text>
                <Progress percent={Math.round(gpu.utilization)} size="small" />
              </div>
              <div style={{ marginTop: 6 }}>
                <Typography.Text style={{ fontSize: 12 }}>
                  Memory: {gpu.memory_used_mb.toFixed(0)} / {gpu.memory_total_mb.toFixed(0)} MB
                </Typography.Text>
                <Progress percent={gpuMemoryPercent(gpu.memory_used_mb, gpu.memory_total_mb)} size="small" />
              </div>
              <div style={{ marginTop: 6 }}>
                <Tag color={temperatureColor(gpu.temperature)}>
                  Temp: {gpu.temperature.toFixed(0)}°C
                </Tag>
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    );
  }, [monitor]);

  return (
    <Card title="System Monitor" size="small" style={{ marginBottom: 16 }}>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Typography.Text strong>CPU</Typography.Text>
          <Typography.Text style={{ float: 'right' }}>
            {monitor ? `${monitor.cpu_percent.toFixed(1)}%` : '-'}
          </Typography.Text>
          <Progress percent={monitor ? Math.round(monitor.cpu_percent) : 0} status="active" />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            Cores: {monitor?.cpu_count ?? '-'}
          </Typography.Text>
        </Col>
        <Col span={12}>
          <Typography.Text strong>Memory</Typography.Text>
          <Typography.Text style={{ float: 'right' }}>
            {monitor ? `${monitor.memory_percent.toFixed(1)}%` : '-'}
          </Typography.Text>
          <Progress percent={monitor ? Math.round(monitor.memory_percent) : 0} status="active" />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {monitor ? `${formatGb(monitor.memory_used_gb)} / ${formatGb(monitor.memory_total_gb)}` : '-'}
          </Typography.Text>
        </Col>
      </Row>

      {gpuCards}

      {error && (
        <Alert
          style={{ marginTop: 12 }}
          type="warning"
          showIcon
          message="Monitor degraded"
          description={error}
        />
      )}
    </Card>
  );
};
