/**
 * Results Display Component
 * Displays training and detection results
 */

import React, { useMemo } from 'react';
import { Card, Descriptions, Statistic, Row, Col, Table, Tag, Space, Typography } from 'antd';
import { CheckCircleOutlined } from '@ant-design/icons';
import { useUILanguage } from '../../hooks/useUILanguage';

interface ResultsDisplayProps {
  results: any;
  type: 'train' | 'detect';
}

const formatPercent = (value: any) => `${(Number(value) * 100).toFixed(2)}`;

const asNumber = (value: any): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  return null;
};

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, type }) => {
  const { t } = useUILanguage();
  if (!results) {
    return null;
  }

  const evalSummary = (type === 'train'
    ? (results.eval_summary || results.summary)
    : results.summary) || {};
  const trainSummary = results.train_summary || {};

  const metrics = evalSummary?.metrics || evalSummary?.dev_eval || {};
  const counts = evalSummary?.counts || {};
  const confusion = counts?.confusion || evalSummary?.confusion;

  const accuracy = counts?.acc ?? metrics?.acc;
  const precision = counts?.precision ?? metrics?.precision;
  const recall = counts?.recall ?? metrics?.recall;

  const f1FromConfusion = (() => {
    const tp = confusion?.tp;
    const fp = confusion?.fp;
    const fn = confusion?.fn;
    if ([tp, fp, fn].every((v) => typeof v === 'number')) {
      const denom = (2 * tp) + fp + fn;
      return denom > 0 ? (2 * tp) / denom : null;
    }
    return null;
  })();
  const f1 = metrics?.f1 ?? counts?.f1 ?? f1FromConfusion;

  const trainMetrics = trainSummary?.train || trainSummary || {};
  const auroc = metrics?.auroc || metrics?.auroc_on_probs;
  const aupr = metrics?.aupr || metrics?.aupr_on_probs;
  const predictionsPreview = Array.isArray(results.predictions_preview)
    ? results.predictions_preview.slice(0, 8)
    : [];

  const asrPayload = type === 'detect' ? evalSummary?.asr : null;
  const asrSummary = asrPayload?.summary || {};
  const asrDefinition = asrPayload?.definition;
  const asrAttacks = (asrPayload && typeof asrPayload.attacks === 'object')
    ? asrPayload.attacks
    : {};

  const asrRows = useMemo(() => {
    if (!asrAttacks || typeof asrAttacks !== 'object') {
      return [];
    }
    return Object.entries(asrAttacks).map(([attackKey, rec]: [string, any]) => {
      const row = rec || {};
      const rowAsr = asNumber(row.asr) ?? asNumber(row?.summary?.asr_mean);
      const rowAttackAcc = asNumber(row.attack_acc);
      const rowAttackEvalN = typeof row.attack_eval_n === 'number' ? row.attack_eval_n : null;
      const rowBaseCorrectN = typeof row.base_correct_n === 'number' ? row.base_correct_n : null;
      const rowMatchMode = typeof row.match_mode === 'string' ? row.match_mode : '-';
      return {
        key: attackKey,
        attack: attackKey,
        asr: rowAsr,
        attack_acc: rowAttackAcc,
        attack_eval_n: rowAttackEvalN,
        base_correct_n: rowBaseCorrectN,
        match_mode: rowMatchMode,
      };
    });
  }, [asrAttacks]);

  const hasAsr = Boolean(asrPayload && (asrRows.length > 0 || asrSummary));

  return (
    <Card
      title={
        <Space>
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
          <span>{type === 'train' ? t('resultsTrainingTitle') : t('resultsDetectionTitle')}</span>
        </Space>
      }
      style={{
        marginTop: 16,
        marginBottom: 16,
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        border: 'none',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
      }}
    >
      <Row gutter={16} style={{ marginBottom: 16 }}>
        {accuracy !== undefined && accuracy !== null && (
          <Col span={6}>
            <Statistic
              title={t('resultsAccuracy')}
              value={accuracy}
              precision={4}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
              formatter={(value) => formatPercent(value)}
            />
          </Col>
        )}
        {precision !== undefined && precision !== null && (
          <Col span={6}>
            <Statistic
              title={t('resultsPrecision')}
              value={precision}
              precision={4}
              valueStyle={{ color: '#1890ff' }}
              suffix="%"
              formatter={(value) => formatPercent(value)}
            />
          </Col>
        )}
        {recall !== undefined && recall !== null && (
          <Col span={6}>
            <Statistic
              title={t('resultsRecall')}
              value={recall}
              precision={4}
              valueStyle={{ color: '#fa8c16' }}
              suffix="%"
              formatter={(value) => formatPercent(value)}
            />
          </Col>
        )}
        {f1 !== undefined && f1 !== null && (
          <Col span={6}>
            <Statistic
              title={t('resultsF1')}
              value={f1}
              precision={4}
              valueStyle={{ color: '#7c3aed' }}
              suffix="%"
              formatter={(value) => formatPercent(value)}
            />
          </Col>
        )}
      </Row>

      {(auroc || aupr) && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          {auroc && (
            <Col span={12}>
              <Statistic
                title={t('resultsAuroc')}
                value={auroc}
                precision={4}
                valueStyle={{ color: '#13c2c2' }}
              />
            </Col>
          )}
          {aupr && (
            <Col span={12}>
              <Statistic
                title={t('resultsAupr')}
                value={aupr}
                precision={4}
                valueStyle={{ color: '#eb2f96' }}
              />
            </Col>
          )}
        </Row>
      )}

      {type === 'train' && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          {trainMetrics?.best_val_acc !== undefined && (
            <Col span={12}>
              <Statistic
                title={t('resultsBestValAcc')}
                value={trainMetrics.best_val_acc}
                precision={4}
                valueStyle={{ color: '#2f54eb' }}
                suffix="%"
                formatter={(value) => formatPercent(value)}
              />
            </Col>
          )}
          {trainMetrics?.test_acc !== undefined && (
            <Col span={12}>
              <Statistic
                title={t('resultsTrainTestAcc')}
                value={trainMetrics.test_acc}
                precision={4}
                valueStyle={{ color: '#08979c' }}
                suffix="%"
                formatter={(value) => formatPercent(value)}
              />
            </Col>
          )}
        </Row>
      )}

      {hasAsr && (
        <Card title={t('resultsAsrOverview')} size="small" style={{ marginTop: 16, background: '#fff' }}>
          <Row gutter={16} style={{ marginBottom: 12 }}>
            {asNumber(asrSummary?.asr_mean) !== null && (
              <Col span={6}>
                <Statistic
                  title={t('resultsAsrMean')}
                  value={asNumber(asrSummary?.asr_mean) || 0}
                  precision={4}
                  valueStyle={{ color: '#cf1322' }}
                  suffix="%"
                  formatter={(value) => formatPercent(value)}
                />
              </Col>
            )}
            {asNumber(asrSummary?.asr_weighted_mean) !== null && (
              <Col span={6}>
                <Statistic
                  title={t('resultsAsrWeightedMean')}
                  value={asNumber(asrSummary?.asr_weighted_mean) || 0}
                  precision={4}
                  valueStyle={{ color: '#d4380d' }}
                  suffix="%"
                  formatter={(value) => formatPercent(value)}
                />
              </Col>
            )}
            {typeof asrSummary?.n_attacks === 'number' && (
              <Col span={6}>
                <Statistic title={t('resultsAsrAttackCount')} value={asrSummary.n_attacks} />
              </Col>
            )}
            {typeof asrSummary?.n_valid_asr === 'number' && (
              <Col span={6}>
                <Statistic title={t('resultsAsrValidCount')} value={asrSummary.n_valid_asr} />
              </Col>
            )}
          </Row>
          {typeof asrDefinition === 'string' && asrDefinition && (
            <Typography.Text type="secondary">
              {t('resultsAsrDefinition')}: {asrDefinition}
            </Typography.Text>
          )}

          {asrRows.length > 0 && (
            <Table
              style={{ marginTop: 12 }}
              dataSource={asrRows}
              columns={[
                { title: t('resultsAsrByAttack'), dataIndex: 'attack', key: 'attack' },
                {
                  title: t('resultsAsr'),
                  dataIndex: 'asr',
                  key: 'asr',
                  width: 120,
                  render: (value: any) => (
                    typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '-'
                  ),
                },
                {
                  title: t('resultsAttackAcc'),
                  dataIndex: 'attack_acc',
                  key: 'attack_acc',
                  width: 120,
                  render: (value: any) => (
                    typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '-'
                  ),
                },
                {
                  title: t('resultsAttackEvalN'),
                  dataIndex: 'attack_eval_n',
                  key: 'attack_eval_n',
                  width: 120,
                  render: (value: any) => (typeof value === 'number' ? value : '-'),
                },
                {
                  title: t('resultsBaseCorrectN'),
                  dataIndex: 'base_correct_n',
                  key: 'base_correct_n',
                  width: 120,
                  render: (value: any) => (typeof value === 'number' ? value : '-'),
                },
                {
                  title: t('resultsMatchMode'),
                  dataIndex: 'match_mode',
                  key: 'match_mode',
                  width: 140,
                },
              ]}
              pagination={false}
              size="small"
            />
          )}
        </Card>
      )}

      {confusion && (
        <Card
          title={t('resultsConfusionMatrix')}
          size="small"
          style={{ marginTop: 16, background: '#fff' }}
        >
          <Table
            dataSource={[
              {
                key: 'positive',
                actual: t('resultsPositive'),
                predictedPositive: confusion.tp,
                predictedNegative: confusion.fn,
              },
              {
                key: 'negative',
                actual: t('resultsNegative'),
                predictedPositive: confusion.fp,
                predictedNegative: confusion.tn,
              },
            ]}
            columns={[
              {
                title: t('resultsActualPredicted'),
                dataIndex: 'actual',
                key: 'actual',
                width: 150,
              },
              {
                title: t('resultsPositive'),
                dataIndex: 'predictedPositive',
                key: 'predictedPositive',
                align: 'center',
                render: (value) => <Tag color="green">{value}</Tag>,
              },
              {
                title: t('resultsNegative'),
                dataIndex: 'predictedNegative',
                key: 'predictedNegative',
                align: 'center',
                render: (value) => <Tag color="red">{value}</Tag>,
              },
            ]}
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {predictionsPreview.length > 0 && (
        <Card title={t('resultsPredictionPreview')} size="small" style={{ marginTop: 16, background: '#fff' }}>
          <Table
            dataSource={predictionsPreview.map((row: any, idx: number) => ({
              key: String(row.id ?? idx),
              text: String(row.text ?? ''),
              pred: row.pred,
              prob: row.prob,
            }))}
            columns={[
              {
                title: t('resultsText'),
                dataIndex: 'text',
                key: 'text',
                render: (value: string) => value || t('resultsEmptyText'),
              },
              {
                title: t('resultsPred'),
                dataIndex: 'pred',
                key: 'pred',
                width: 90,
                render: (value: any) => (
                  <Tag color={Number(value) === 1 ? 'red' : 'green'}>
                    {Number(value) === 1 ? t('resultsMachine') : t('resultsHuman')}
                  </Tag>
                ),
              },
              {
                title: t('resultsProb'),
                dataIndex: 'prob',
                key: 'prob',
                width: 120,
                render: (value: any) => (
                  typeof value === 'number' ? value.toFixed(4) : '-'
                ),
              },
            ]}
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {(evalSummary?.meta || results?.manifest) && (
        <Descriptions
          title={t('resultsDetectorInformation')}
          bordered
          size="small"
          style={{ marginTop: 16 }}
        >
          {evalSummary?.detector && (
            <Descriptions.Item label={t('resultsDetector')}>{evalSummary.detector}</Descriptions.Item>
          )}
          {evalSummary?.meta?.detector_type && (
            <Descriptions.Item label={t('resultsType')}>{evalSummary.meta.detector_type}</Descriptions.Item>
          )}
          {evalSummary?.meta?.dev?.num_samples && (
            <Descriptions.Item label={t('resultsSamples')}>{evalSummary.meta.dev.num_samples}</Descriptions.Item>
          )}
          {results?.manifest?.timing?.evaluate_sec && (
            <Descriptions.Item label={t('resultsEvalTimeSec')}>
              {Number(results.manifest.timing.evaluate_sec).toFixed(3)}
            </Descriptions.Item>
          )}
        </Descriptions>
      )}
    </Card>
  );
};
