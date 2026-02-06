import React from 'react';
import { Button, Space, Tag, Typography } from 'antd';
import { GlobalOutlined, ThunderboltOutlined } from '@ant-design/icons';

interface HFMirrorSuggestionProps {
  show: boolean;
  onUseMirror: () => void;
}

export const HFMirrorSuggestion: React.FC<HFMirrorSuggestionProps> = ({
  show,
  onUseMirror,
}) => {
  if (!show) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: 8,
        padding: 12,
        borderRadius: 10,
        border: '1px solid #d6e4ff',
        background: 'linear-gradient(135deg, #f7fbff 0%, #edf5ff 100%)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 10,
          marginBottom: 6,
        }}
      >
        <Space size={8}>
          <GlobalOutlined style={{ color: '#1677ff' }} />
          <Typography.Text strong style={{ color: '#1d39c4' }}>
            建议使用 HF Mirror 源
          </Typography.Text>
          <Tag color="blue" style={{ margin: 0 }}>
            CN Friendly
          </Tag>
        </Space>
        <Button
          size="small"
          type="primary"
          icon={<ThunderboltOutlined />}
          onClick={onUseMirror}
        >
          一键切换
        </Button>
      </div>
      <Typography.Text style={{ fontSize: 12, color: '#4b5563' }}>
        检测到您可能位于中国地区，切换到 `https://hf-mirror.com` 通常可显著提升模型下载成功率与速度。
      </Typography.Text>
    </div>
  );
};
