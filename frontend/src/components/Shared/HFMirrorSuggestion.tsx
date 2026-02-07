import React from 'react';
import { Button, Space, Tag, Typography } from 'antd';
import { GlobalOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { UILanguage } from '../../types';
import { getCoreText } from '../../i18n/coreText';

interface HFMirrorSuggestionProps {
  language: UILanguage;
  show: boolean;
  onUseMirror: () => void;
}

export const HFMirrorSuggestion: React.FC<HFMirrorSuggestionProps> = ({
  language,
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
            {getCoreText(language, 'hfMirrorTitle')}
          </Typography.Text>
          <Tag color="blue" style={{ margin: 0 }}>
            {getCoreText(language, 'hfMirrorTag')}
          </Tag>
        </Space>
        <Button
          size="small"
          type="primary"
          icon={<ThunderboltOutlined />}
          onClick={onUseMirror}
        >
          {getCoreText(language, 'hfMirrorAction')}
        </Button>
      </div>
      <Typography.Text style={{ fontSize: 12, color: '#4b5563' }}>
        {getCoreText(language, 'hfMirrorBody')}
      </Typography.Text>
    </div>
  );
};
