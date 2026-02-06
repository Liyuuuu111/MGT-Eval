/**
 * Local Model Selector Component
 */

import React, { useEffect, useState } from 'react';
import { Select, Alert, Spin, Tag, Input } from 'antd';
import { RobotOutlined, ReloadOutlined } from '@ant-design/icons';
import api from '../../services/api';

interface Model {
  name: string;
  path: string;
  size: string;
}

interface ModelSelectorProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  allowManual?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  value,
  onChange,
  placeholder = "Select a local model or enter model name",
  allowManual = true
}) => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [manualInput, setManualInput] = useState(false);
  const marqueeThreshold = 26;
  const marqueeDurationSec = 10;

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.getLocalModels();
      setModels(result.models);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to detect models');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Spin tip="Scanning local models..." />;
  }

  if (error) {
    return (
      <Alert
        type="warning"
        message="Model Detection Failed"
        description={
          <div>
            {error}
            <br />
            <a onClick={loadModels} style={{ cursor: 'pointer' }}>
              <ReloadOutlined /> Retry
            </a>
          </div>
        }
        showIcon
      />
    );
  }

  if (manualInput || models.length === 0) {
    return (
      <div>
        <style>
          {`
            @keyframes model-name-marquee {
              0% { transform: translateX(0); }
              100% { transform: translateX(-50%); }
            }
            .model-option-row {
              display: grid;
              grid-template-columns: minmax(0, 1fr) 90px;
              align-items: center;
              gap: 8px;
              width: 100%;
            }
            .model-name-wrap {
              overflow: hidden;
              white-space: nowrap;
              position: relative;
            }
            .model-name-text {
              display: inline-block;
              white-space: nowrap;
            }
            .model-name-scroll {
              display: inline-flex;
              gap: 24px;
              white-space: nowrap;
              animation: model-name-marquee ${marqueeDurationSec}s linear infinite;
              will-change: transform;
            }
          `}
        </style>
        <Input
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder}
          prefix={<RobotOutlined />}
        />
        {models.length > 0 && (
          <div style={{ marginTop: 4 }}>
            <a onClick={() => setManualInput(false)} style={{ fontSize: '12px' }}>
              ← Back to local models
            </a>
          </div>
        )}
        {models.length === 0 && (
          <Alert
            type="info"
            message="No Local Models Found"
            description="No locally cached models detected. Enter a model name manually or download models first."
            showIcon
            style={{ marginTop: 8 }}
          />
        )}
      </div>
    );
  }

  return (
    <div>
      <style>
        {`
          @keyframes model-name-marquee {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
          }
          .model-option-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 90px;
            align-items: center;
            gap: 8px;
            width: 100%;
          }
          .model-name-wrap {
            overflow: hidden;
            white-space: nowrap;
            position: relative;
          }
          .model-name-text {
            display: inline-block;
            white-space: nowrap;
          }
          .model-name-scroll {
            display: inline-flex;
            gap: 24px;
            white-space: nowrap;
            animation: model-name-marquee ${marqueeDurationSec}s linear infinite;
            will-change: transform;
          }
        `}
      </style>
      <Select
        showSearch
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        style={{ width: '100%' }}
        filterOption={(input, option) =>
          (option?.label?.toString() || '').toLowerCase().includes(input.toLowerCase())
        }
        dropdownRender={(menu) => (
          <>
            {menu}
            {allowManual && (
              <div
                style={{
                  borderTop: '1px solid #f0f0f0',
                  padding: '8px',
                  cursor: 'pointer',
                }}
                onClick={() => setManualInput(true)}
              >
                ✏️ Enter model name manually
              </div>
            )}
          </>
        )}
      >
        {models.map((model) => (
          <Select.Option key={model.name} value={model.name} label={model.name}>
            <div className="model-option-row">
              <div className="model-name-wrap" title={model.name}>
                {model.name.length > marqueeThreshold ? (
                  <div className="model-name-scroll">
                    <span><RobotOutlined /> {model.name}</span>
                    <span><RobotOutlined /> {model.name}</span>
                  </div>
                ) : (
                  <span className="model-name-text"><RobotOutlined /> {model.name}</span>
                )}
              </div>
              <div style={{ textAlign: 'right' }}>
                <Tag color="blue" style={{ margin: 0 }}>{model.size}</Tag>
              </div>
            </div>
          </Select.Option>
        ))}
      </Select>
      <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
        💡 Found {models.length} local model(s) in cache
      </div>
    </div>
  );
};
