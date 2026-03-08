import React, { useState } from 'react';
import { Button, Input, Space, Tag, Typography, Upload, message } from 'antd';
import type { UploadProps } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

import api from '../../services/api';

const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_UPLOAD_SIZE_LABEL = '10MB';

interface DatasetUploadInputProps {
  value?: string;
  onChange?: (value: string) => void;
  phase: 'build' | 'attack' | 'train' | 'detect';
  disabled?: boolean;
  placeholder?: string;
}

export const DatasetUploadInput: React.FC<DatasetUploadInputProps> = ({
  value,
  onChange,
  phase,
  disabled = false,
  placeholder,
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadedName, setUploadedName] = useState<string>('');
  const [uploadedSize, setUploadedSize] = useState<number>(0);

  const beforeUpload: UploadProps['beforeUpload'] = (file) => {
    if ((file as File).size > MAX_UPLOAD_SIZE_BYTES) {
      message.error(`File is too large. Maximum allowed size is ${MAX_UPLOAD_SIZE_LABEL}.`);
      return Upload.LIST_IGNORE;
    }

    const doUpload = async () => {
      setUploading(true);
      try {
        const response = await api.uploadDataset(file as File, phase);
        const storedPath = String(response?.stored_path || '');
        if (storedPath) {
          onChange?.(storedPath);
          setUploadedName(String(response?.file_name || file.name || 'dataset'));
          setUploadedSize(Number(response?.file_size || 0));
          message.success(`Uploaded to managed storage: ${storedPath}`);
        } else {
          message.error('Upload succeeded, but no stored path was returned.');
        }
      } catch (error: any) {
        message.error(error?.response?.data?.detail || 'Dataset upload failed');
      } finally {
        setUploading(false);
      }
    };
    void doUpload();
    return Upload.LIST_IGNORE;
  };

  const formatSize = (bytes: number): string => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={8}>
      <Space.Compact style={{ width: '100%' }}>
        <Input
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder || 'Enter file path or upload a dataset file'}
          disabled={disabled || uploading}
        />
        <Upload
          disabled={disabled || uploading}
          showUploadList={false}
          beforeUpload={beforeUpload}
        >
          <Button icon={<UploadOutlined />} loading={uploading} disabled={disabled || uploading}>
            Upload
          </Button>
        </Upload>
      </Space.Compact>
      {uploadedName ? (
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          <Tag color="blue" style={{ marginRight: 8 }}>Managed Upload</Tag>
          {uploadedName} ({formatSize(uploadedSize)})
        </Typography.Text>
      ) : null}
    </Space>
  );
};

export default DatasetUploadInput;
