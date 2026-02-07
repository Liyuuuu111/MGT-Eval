/**
 * Attack Example Viewer Component
 * Displays curated before/after text plus diff highlights.
 */

import React, { useMemo } from 'react';
import { Card, Typography, Divider, Tag, Space } from 'antd';
import { DiffOutlined } from '@ant-design/icons';
import { ATTACK_EXAMPLES } from './attackExamples';
import { buildDiffSegments } from '../../utils/textDiff';
import { getCoreText } from '../../i18n/coreText';
import type { UILanguage } from '../../types';

interface AttackExampleViewerProps {
  attackType: string;
  language: UILanguage;
}

const boxStyle: React.CSSProperties = {
  fontSize: 13,
  color: '#262626',
  display: 'block',
  marginTop: 6,
  padding: '10px',
  background: '#fff',
  borderRadius: '6px',
  border: '1px solid #d9d9d9',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  lineHeight: 1.6,
};

const renderDiffSegment = (text: string, type: 'equal' | 'add' | 'del', key: number) => {
  if (type === 'add') {
    return (
      <span key={key} style={{ background: '#f6ffed', color: '#237804', borderRadius: 2 }}>
        {text}
      </span>
    );
  }
  if (type === 'del') {
    return (
      <span
        key={key}
        style={{
          background: '#fff1f0',
          color: '#a8071a',
          textDecoration: 'line-through',
          borderRadius: 2,
        }}
      >
        {text}
      </span>
    );
  }
  return <span key={key}>{text}</span>;
};

export const AttackExampleViewer: React.FC<AttackExampleViewerProps> = ({ attackType, language }) => {
  const example = useMemo(() => {
    const exact = ATTACK_EXAMPLES.find(
      (ex) => ex.attackType === attackType && ex.language === language,
    );
    if (exact) {
      return exact;
    }
    return ATTACK_EXAMPLES.find((ex) => ex.attackType === attackType && ex.language === 'en') ?? null;
  }, [attackType, language]);

  if (!example) {
    return (
      <Card size="small" style={{ marginTop: 12 }}>
        <Typography.Text type="secondary">{getCoreText(language, 'attackExampleNoData')}</Typography.Text>
      </Card>
    );
  }

  const diffSegments = buildDiffSegments(example.original, example.attacked, example.diffMode ?? 'auto');

  return (
    <Card
      size="small"
      title={
        <Space>
          <DiffOutlined />
          <span>{example.title || getCoreText(language, 'attackExampleTitle')}</span>
        </Space>
      }
      style={{
        marginTop: 12,
        background: language === 'zh' ? '#f6ffed' : '#e6f7ff',
        borderColor: language === 'zh' ? '#b7eb8f' : '#91d5ff',
      }}
    >
      <Typography.Paragraph style={{ marginBottom: 10 }}>
        <Tag color="blue">{getCoreText(language, 'attackExampleOriginal')}</Tag>
        <Typography.Text style={boxStyle}>{example.original}</Typography.Text>
      </Typography.Paragraph>

      <Typography.Paragraph style={{ marginBottom: 10 }}>
        <Tag color="orange">{getCoreText(language, 'attackExampleAttacked')}</Tag>
        <Typography.Text style={{ ...boxStyle, border: '1px solid #ffa940' }}>{example.attacked}</Typography.Text>
      </Typography.Paragraph>

      <Divider style={{ margin: '10px 0' }} />

      <Typography.Paragraph style={{ marginBottom: 8 }}>
        <Tag color="purple">{getCoreText(language, 'attackExampleDiff')}</Tag>
        <Typography.Text style={boxStyle}>
          {diffSegments.map((segment, idx) => renderDiffSegment(segment.text, segment.type, idx))}
        </Typography.Text>
      </Typography.Paragraph>

      {example.notes && (
        <Typography.Paragraph style={{ marginBottom: 8 }}>
          <Tag color="geekblue">{getCoreText(language, 'attackExampleNotes')}</Tag>
          <Typography.Text style={{ ...boxStyle, border: '1px solid #adc6ff' }}>{example.notes}</Typography.Text>
        </Typography.Paragraph>
      )}

      {example.parameters && Object.keys(example.parameters).length > 0 && (
        <Space size={[6, 6]} wrap>
          {Object.entries(example.parameters).map(([key, value]) => (
            <Tag key={key} color="default">
              {key}: {String(value)}
            </Tag>
          ))}
        </Space>
      )}
    </Card>
  );
};

