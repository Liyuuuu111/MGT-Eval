/**
 * GPU Selector Component
 */

import React, { useEffect, useState } from 'react';
import { Select, Alert, Spin, Tag } from 'antd';
import { ThunderboltOutlined } from '@ant-design/icons';
import api from '../../services/api';

interface GPU {
  id: number;
  name: string;
  memory_total: string;
  memory_free: string;
  utilization: string;
  available: boolean;
}

interface GPUSelectorProps {
  value?: number | number[];
  onChange?: (value: number[] | number | undefined) => void;
  mode?: 'single' | 'multiple';
}

export const GPUSelector: React.FC<GPUSelectorProps> = ({
  value,
  onChange,
  mode = 'multiple'
}) => {
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [recommended, setRecommended] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadGPUs();
  }, []);

  const normalizeToArray = (input: number | number[] | undefined): number[] => {
    if (Array.isArray(input)) {
      return input.filter((v) => Number.isFinite(v));
    }
    if (typeof input === 'number' && Number.isFinite(input)) {
      return [input];
    }
    return [];
  };

  const handleChange = (next: number | number[]) => {
    if (mode === 'multiple') {
      onChange?.(normalizeToArray(next as number[]));
      return;
    }
    if (Array.isArray(next)) {
      onChange?.(next.length > 0 ? next[0] : undefined);
      return;
    }
    onChange?.(next);
  };

  const normalizedMultipleValue = normalizeToArray(value);
  const normalizedSingleValue =
    typeof value === 'number' ? value : (Array.isArray(value) && value.length > 0 ? value[0] : undefined);

  const loadGPUs = async () => {
    try {
      setLoading(true);
      const result = await api.getGPUs();
      setGpus(result.gpus);
      setRecommended(result.recommended_gpu);

      // Auto-select recommended GPU if nothing is selected
      if ((value === undefined || value === null || (Array.isArray(value) && value.length === 0)) && result.recommended_gpu !== null) {
        if (mode === 'multiple') {
          onChange?.([result.recommended_gpu]);
        } else {
          onChange?.(result.recommended_gpu);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to detect GPUs');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Spin spinning tip="Detecting GPUs...">
        <div style={{ minHeight: 28 }} />
      </Spin>
    );
  }

  if (error) {
    return (
      <Alert
        type="warning"
        message="GPU Detection Failed"
        description={error}
        showIcon
      />
    );
  }

  if (gpus.length === 0) {
    return (
      <Alert
        type="info"
        message="No GPUs Detected"
        description="No NVIDIA GPUs found. Operations will run on CPU."
        showIcon
      />
    );
  }

  const renderGPUOption = (gpu: GPU) => (
    <Select.Option key={gpu.id} value={gpu.id}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>
          <ThunderboltOutlined /> GPU {gpu.id}: {gpu.name}
        </span>
        <div>
          {recommended === gpu.id && (
            <Tag color="green" style={{ marginRight: 8 }}>Recommended</Tag>
          )}
          <Tag color={gpu.available ? 'success' : 'warning'}>
            {gpu.utilization}
          </Tag>
          <Tag>{gpu.memory_free} free</Tag>
        </div>
      </div>
    </Select.Option>
  );

  return (
    <div>
      <Select
        mode={mode === 'multiple' ? 'multiple' : undefined}
        value={mode === 'multiple' ? normalizedMultipleValue : normalizedSingleValue}
        onChange={handleChange}
        placeholder="Select GPU(s)"
        style={{ width: '100%' }}
      >
        {gpus.map((gpu) => renderGPUOption(gpu))}
      </Select>
      <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
        💡 Tip: Select multiple GPUs for parallel processing or leave empty for CPU
      </div>
    </div>
  );
};
