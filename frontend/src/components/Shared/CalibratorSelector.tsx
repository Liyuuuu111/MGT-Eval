/**
 * Calibrator path selector for metric detectors.
 * Supports dropdown search and manual path input.
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Input,
  Select,
  Space,
  Spin,
  Tag,
  Typography,
} from 'antd';
import {
  FileTextOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import api from '../../services/api';

interface CalibratorInfo {
  name: string;
  path: string;
  size: string;
}

interface CalibratorSelectorProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  allowManual?: boolean;
}

export const CalibratorSelector: React.FC<CalibratorSelectorProps> = ({
  value,
  onChange,
  placeholder = 'Select calibrator path or enter manually',
  allowManual = true,
}) => {
  const [calibrators, setCalibrators] = useState<CalibratorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [manualInput, setManualInput] = useState(false);
  const [customDirs, setCustomDirs] = useState('');

  const loadCalibrators = async (dirs?: string) => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.getCalibrators(dirs);
      const rows = Array.isArray(result?.calibrators) ? result.calibrators : [];
      setCalibrators(rows);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to detect calibrator files');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCalibrators();
  }, []);

  if (loading) {
    return <Spin tip="Scanning calibrator paths..." />;
  }

  if (error) {
    return (
      <Alert
        type="warning"
        message="Calibrator Detection Failed"
        description={
          <div>
            {error}
            <br />
            <a onClick={() => loadCalibrators()} style={{ cursor: 'pointer' }}>
              <ReloadOutlined /> Retry
            </a>
          </div>
        }
        showIcon
      />
    );
  }

  if (manualInput) {
    return (
      <div>
        <Input
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder}
          prefix={<FileTextOutlined />}
        />
        <div style={{ marginTop: 6 }}>
          <a onClick={() => setManualInput(false)} style={{ fontSize: 12 }}>
            Back to detected calibrators
          </a>
        </div>
      </div>
    );
  }

  return (
    <div>
      <Select
        showSearch
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        style={{ width: '100%' }}
        optionFilterProp="label"
        filterOption={(input, option) =>
          String(option?.label || '').toLowerCase().includes(input.toLowerCase())
        }
      >
        {calibrators.map((item) => {
          const isDir = item.path.endsWith('/');
          return (
            <Select.Option key={item.path} value={item.path} label={`${item.path} ${item.size}`}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={item.path}>
                  {isDir ? <FolderOpenOutlined /> : <FileTextOutlined />} {item.path}
                </div>
                <Tag color={isDir ? 'gold' : 'blue'} style={{ margin: 0 }}>
                  {item.size}
                </Tag>
              </div>
            </Select.Option>
          );
        })}
      </Select>

      <Space style={{ marginTop: 8, width: '100%' }}>
        <Input
          value={customDirs}
          onChange={(e) => setCustomDirs(e.target.value)}
          placeholder="Additional scan dirs (comma-separated)"
        />
        <Button
          icon={<SearchOutlined />}
          onClick={() => loadCalibrators(customDirs.trim() || undefined)}
        >
          Search Path
        </Button>
        <Button
          icon={<ReloadOutlined />}
          onClick={() => {
            setCustomDirs('');
            loadCalibrators();
          }}
        >
          Reset
        </Button>
      </Space>

      <div style={{ marginTop: 6 }}>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          Auto-scan default path: calibration_results
        </Typography.Text>
        {allowManual && (
          <>
            {' '}
            <a onClick={() => setManualInput(true)} style={{ fontSize: 12 }}>
              Enter path manually
            </a>
          </>
        )}
      </div>
    </div>
  );
};
