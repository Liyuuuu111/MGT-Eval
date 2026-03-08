import React from 'react';
import { Segmented, Space } from 'antd';
import { GlobalOutlined } from '@ant-design/icons';
import { useUILanguage } from '../../hooks/useUILanguage';
import type { UILanguage } from '../../types';

export const LanguageSwitcher: React.FC = () => {
  const { language, setLanguage } = useUILanguage();

  return (
    <Space size={8} align="center">
      <GlobalOutlined style={{ color: '#fff', fontSize: 16 }} />
      <style>
        {`
          .language-switcher .ant-segmented-item {
            color: #595959 !important;
            font-weight: 500;
          }
          .language-switcher .ant-segmented-item-selected {
            color: #1890ff !important;
            font-weight: 600;
            background: #fff !important;
          }
          .language-switcher .ant-segmented-thumb {
            background: #fff !important;
          }
        `}
      </style>
      <Segmented
        className="language-switcher"
        value={language}
        onChange={(value) => setLanguage(value as UILanguage)}
        options={[
          { label: '🇬🇧 English', value: 'en' },
          { label: '🇨🇳 中文', value: 'zh' },
        ]}
        style={{
          backgroundColor: 'rgba(240, 242, 245, 0.95)',
          padding: '3px',
        }}
      />
    </Space>
  );
};

