import React from 'react';
import { Card, Empty, Space, Tag, Typography } from 'antd';

import { JobDownloadItem } from '../../types';
import { resolveApiUrl } from '../../services/api';

interface DownloadLinksCardProps {
  title: string;
  downloads: JobDownloadItem[];
  loading?: boolean;
  emptyText?: string;
}

const formatSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const formatRemaining = (expiresAt: string): string => {
  const ts = Date.parse(expiresAt);
  if (!Number.isFinite(ts)) return 'unknown';
  const remainSec = Math.floor((ts - Date.now()) / 1000);
  if (remainSec <= 0) return 'expired';
  const days = Math.floor(remainSec / 86400);
  const hours = Math.floor((remainSec % 86400) / 3600);
  const mins = Math.floor((remainSec % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
};

export const DownloadLinksCard: React.FC<DownloadLinksCardProps> = ({
  title,
  downloads,
  loading = false,
  emptyText = 'No downloadable outputs were generated for this run.',
}) => {
  return (
    <Card title={title} loading={loading} style={{ marginBottom: 16 }}>
      {downloads.length <= 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
      ) : (
        <Space direction="vertical" style={{ width: '100%' }} size={10}>
          {downloads.map((item) => (
            <div key={`${item.url}__${item.name}`} style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
              <div style={{ minWidth: 0 }}>
                <Typography.Link href={resolveApiUrl(item.url)} target="_blank" rel="noopener noreferrer">
                  {item.name}
                </Typography.Link>
                <div style={{ fontSize: 12, color: '#8c8c8c' }}>
                  expires: {new Date(item.expires_at).toLocaleString()}
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <Tag color="geekblue">{formatSize(item.size_bytes)}</Tag>
                <Tag color="orange">TTL {formatRemaining(item.expires_at)}</Tag>
              </div>
            </div>
          ))}
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            Download links are valid for 3 days. Expired files are removed automatically.
          </Typography.Text>
        </Space>
      )}
    </Card>
  );
};

export default DownloadLinksCard;
