import React, { useMemo } from 'react';
import { Alert, Button, Card, Collapse, Descriptions, Divider, Select, Space, Tag, Tooltip, Typography } from 'antd';
import { CheckOutlined, ExperimentOutlined } from '@ant-design/icons';
import { CalibratorThresholdPreset, UILanguage } from '../../types';

interface ThresholdPresetSelectorProps {
  language: UILanguage;
  presets: CalibratorThresholdPreset[];
  loading?: boolean;
  selectedPreset?: string;
  onSelectedPresetChange: (value: string | undefined) => void;
  onApplySelectedPreset: () => void;
  calibratorPath?: string;
  defaultThreshold?: number;
  onApplyDefaultThreshold?: () => void;
  selectPlaceholder: string;
  applyPresetLabel: string;
  noPresetLabel: string;
  noCalibratorLabel: string;
}

const getFileName = (path?: string): string => {
  if (!path) {
    return '';
  }
  const trimmed = path.replace(/\/+$/, '');
  const parts = trimmed.split('/');
  return parts[parts.length - 1] || trimmed;
};

const toPresetChipLabel = (key: string, threshold: number): string => {
  const cleanKey = String(key || '').replace(/\s+/g, ' ').trim();
  return `${cleanKey} · ${threshold.toFixed(4)}`;
};

const formatRatePercent = (value: number): string => {
  const pct = value * 100;
  if (!Number.isFinite(pct)) {
    return 'N/A';
  }
  if (Math.abs(pct) < 0.01) {
    return `${pct.toExponential(2)}%`;
  }
  if (Math.abs(pct) < 1) {
    return `${pct.toFixed(3)}%`;
  }
  return `${pct.toFixed(2)}%`;
};

export const ThresholdPresetSelector: React.FC<ThresholdPresetSelectorProps> = ({
  language,
  presets,
  loading = false,
  selectedPreset,
  onSelectedPresetChange,
  onApplySelectedPreset,
  calibratorPath,
  defaultThreshold,
  onApplyDefaultThreshold,
  selectPlaceholder,
  applyPresetLabel,
  noPresetLabel,
  noCalibratorLabel,
}) => {
  const selectedItem = useMemo(
    () => presets.find((item) => `${item.key}@@${item.threshold}` === selectedPreset),
    [presets, selectedPreset],
  );

  const topPresets = useMemo(() => presets.slice(0, 8), [presets]);
  const compactCalibrator = getFileName(calibratorPath);
  const selectedMetricTags = useMemo(() => {
    if (!selectedItem) {
      return [];
    }
    const rows: Array<{ label: string; value: string; isRate?: boolean }> = [];
    const labels = language === 'zh'
      ? {
          tpr: 'TPR',
          fpr: 'FPR',
          targetFpr: '目标 FPR',
          acc: '准确率',
          precision: '精确率',
          recall: '召回率',
          f1: 'F1',
          tp: 'TP',
          tn: 'TN',
          fp: 'FP',
          fn: 'FN',
        }
      : {
          tpr: 'TPR',
          fpr: 'FPR',
          targetFpr: 'Target FPR',
          acc: 'Accuracy',
          precision: 'Precision',
          recall: 'Recall',
          f1: 'F1',
          tp: 'TP',
          tn: 'TN',
          fp: 'FP',
          fn: 'FN',
        };

    if (typeof selectedItem.tpr === 'number') rows.push({ label: labels.tpr, value: formatRatePercent(selectedItem.tpr), isRate: true });
    if (typeof selectedItem.fpr === 'number') rows.push({ label: labels.fpr, value: formatRatePercent(selectedItem.fpr), isRate: true });
    if (typeof selectedItem.target_fpr === 'number') rows.push({ label: labels.targetFpr, value: formatRatePercent(selectedItem.target_fpr), isRate: true });
    if (typeof selectedItem.acc === 'number') rows.push({ label: labels.acc, value: formatRatePercent(selectedItem.acc), isRate: true });
    if (typeof selectedItem.precision === 'number') rows.push({ label: labels.precision, value: formatRatePercent(selectedItem.precision), isRate: true });
    if (typeof selectedItem.recall === 'number') rows.push({ label: labels.recall, value: formatRatePercent(selectedItem.recall), isRate: true });
    if (typeof selectedItem.f1 === 'number') rows.push({ label: labels.f1, value: formatRatePercent(selectedItem.f1), isRate: true });
    if (typeof selectedItem.tp === 'number') rows.push({ label: labels.tp, value: String(selectedItem.tp) });
    if (typeof selectedItem.tn === 'number') rows.push({ label: labels.tn, value: String(selectedItem.tn) });
    if (typeof selectedItem.fp === 'number') rows.push({ label: labels.fp, value: String(selectedItem.fp) });
    if (typeof selectedItem.fn === 'number') rows.push({ label: labels.fn, value: String(selectedItem.fn) });
    return rows;
  }, [language, selectedItem]);

  return (
    <Card
      size="small"
      style={{
        background: 'linear-gradient(180deg, #fcfdff 0%, #f5f8ff 100%)',
        border: '1px solid #d6e4ff',
        borderRadius: 10,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <Space wrap size={[6, 6]}>
            <Tag color={presets.length > 0 ? 'blue' : 'default'} style={{ margin: 0 }}>
              {presets.length} {language === 'zh' ? '个预设' : 'presets'}
            </Tag>
            {!!compactCalibrator && (
              <Tag
                color="geekblue"
                style={{
                  margin: 0,
                  maxWidth: 320,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                }}
                title={calibratorPath}
              >
                <span>{language === 'zh' ? '校准器' : 'Calibrator'}:</span>
                <span
                  style={{
                    maxWidth: 220,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    display: 'inline-block',
                    verticalAlign: 'bottom',
                  }}
                >
                  {compactCalibrator}
                </span>
              </Tag>
            )}
          </Space>
        </div>
        {!!calibratorPath && (
          <Typography.Text type="secondary" style={{ fontSize: 12, whiteSpace: 'normal', wordBreak: 'break-all' }}>
            {language === 'zh' ? '当前校准器路径：' : 'Active calibrator path: '}
            {calibratorPath}
          </Typography.Text>
        )}

        <div style={{ display: 'flex', gap: 8, width: '100%' }}>
          <Select
            value={selectedPreset}
            onChange={(v) => onSelectedPresetChange(v || undefined)}
            placeholder={selectPlaceholder}
            loading={loading}
            disabled={loading || presets.length === 0}
            style={{ flex: 1, minWidth: 0 }}
            optionLabelProp="label"
            popupMatchSelectWidth={false}
            dropdownStyle={{ minWidth: 560, maxWidth: 860 }}
          >
            {presets.map((item) => (
              <Select.Option
                key={`${item.key}@@${item.threshold}`}
                value={`${item.key}@@${item.threshold}`}
                label={toPresetChipLabel(item.key, item.threshold)}
              >
                <div style={{ display: 'grid', gap: 4, whiteSpace: 'normal' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                    <Typography.Text strong style={{ color: '#1d39c4' }}>
                      {item.key}
                    </Typography.Text>
                    <Tag color="processing" style={{ margin: 0 }}>
                      {item.threshold.toFixed(6)}
                    </Tag>
                  </div>
                  <Typography.Text type="secondary" style={{ fontSize: 12, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                    {item.label}
                  </Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 11, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                    {item.source}
                  </Typography.Text>
                </div>
              </Select.Option>
            ))}
          </Select>
          <Button
            type="primary"
            icon={<CheckOutlined />}
            onClick={onApplySelectedPreset}
            disabled={!selectedPreset}
            style={{ flexShrink: 0, borderRadius: 8 }}
          >
            {applyPresetLabel}
          </Button>
        </div>

        {topPresets.length > 0 && (
          <Space wrap size={[6, 6]}>
            {topPresets.map((item) => {
              const key = `${item.key}@@${item.threshold}`;
              const isActive = selectedPreset === key;
              return (
                <Tooltip key={key} title={`${item.label}\n${item.source}`}>
                  <Button
                    size="small"
                    type={isActive ? 'primary' : 'default'}
                    onClick={() => onSelectedPresetChange(key)}
                  >
                    {item.key} · {item.threshold.toFixed(4)}
                  </Button>
                </Tooltip>
              );
            })}
          </Space>
        )}

        {selectedItem && (
          <Descriptions
            size="small"
            column={1}
            bordered
            styles={{
              label: { width: 138 },
              content: { whiteSpace: 'normal', wordBreak: 'break-word' },
            }}
          >
            <Descriptions.Item label={language === 'zh' ? '预设名称' : 'Preset'}>
              <Typography.Text style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>
                {selectedItem.label}
              </Typography.Text>
            </Descriptions.Item>
            <Descriptions.Item label={language === 'zh' ? '阈值' : 'Threshold'}>
              <Typography.Text strong>{selectedItem.threshold.toFixed(6)}</Typography.Text>
            </Descriptions.Item>
            <Descriptions.Item label={language === 'zh' ? '来源' : 'Source'}>
              <Typography.Text type="secondary" style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>
                {selectedItem.source}
              </Typography.Text>
            </Descriptions.Item>
            {selectedMetricTags.length > 0 && (
              <Descriptions.Item label={language === 'zh' ? '训练集指标' : 'Dev-set Metrics'}>
                <Space wrap size={[6, 6]}>
                  {selectedMetricTags.map((entry) => (
                    <Tag
                      key={`${entry.label}:${entry.value}`}
                      color={entry.isRate ? 'processing' : 'default'}
                      style={{ margin: 0 }}
                    >
                      {entry.label}: {entry.value}
                    </Tag>
                  ))}
                </Space>
              </Descriptions.Item>
            )}
          </Descriptions>
        )}

        {presets.length > 8 && (
          <>
            <Divider style={{ margin: '4px 0 0 0' }} />
            <Collapse
              ghost
              size="small"
              items={[
                {
                  key: 'all-presets',
                  label: language === 'zh' ? `查看全部预设（${presets.length}）` : `View all presets (${presets.length})`,
                  children: (
                    <div style={{ maxHeight: 220, overflowY: 'auto', display: 'grid', gap: 8 }}>
                      {presets.map((item) => {
                        const key = `${item.key}@@${item.threshold}`;
                        return (
                          <Button
                            key={key}
                            type={selectedPreset === key ? 'primary' : 'default'}
                            style={{ textAlign: 'left', height: 'auto', whiteSpace: 'normal' }}
                            onClick={() => onSelectedPresetChange(key)}
                          >
                            <div style={{ display: 'grid', gap: 2 }}>
                              <Typography.Text strong>{item.key} · {item.threshold.toFixed(6)}</Typography.Text>
                              <Typography.Text type="secondary" style={{ fontSize: 12 }}>{item.label}</Typography.Text>
                            </div>
                          </Button>
                        );
                      })}
                    </div>
                  ),
                },
              ]}
            />
          </>
        )}

        {typeof defaultThreshold === 'number' && onApplyDefaultThreshold && (
          <Button
            block
            onClick={onApplyDefaultThreshold}
            icon={<ExperimentOutlined />}
          >
            {language === 'zh' ? '使用默认阈值' : 'Use Default Threshold'}: {defaultThreshold.toFixed(6)}
          </Button>
        )}

        {!!calibratorPath && !loading && presets.length === 0 && (
          <Alert type="info" showIcon message={noPresetLabel} />
        )}
        {!calibratorPath && (
          <Alert type="warning" showIcon message={noCalibratorLabel} />
        )}
      </Space>
    </Card>
  );
};
