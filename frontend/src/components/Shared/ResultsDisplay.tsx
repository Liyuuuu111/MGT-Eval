/**
 * Results Display Component
 * Displays training and detection results
 */

import React from 'react';
import { Card, Descriptions, Statistic, Row, Col, Table, Tag, Space } from 'antd';
import { CheckCircleOutlined } from '@ant-design/icons';

interface ResultsDisplayProps {
  results: any;
  type: 'train' | 'detect';
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, type }) => {
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

  return (
    <Card
      title={
        <Space>
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
          <span>{type === 'train' ? 'Training Results' : 'Detection Results'}</span>
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
      {/* Performance Metrics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        {accuracy !== undefined && accuracy !== null && (
          <Col span={6}>
            <Statistic
              title="Accuracy"
              value={accuracy}
              precision={4}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
              formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
            />
          </Col>
        )}
        {precision !== undefined && precision !== null && (
          <Col span={6}>
            <Statistic
              title="Precision"
              value={precision}
              precision={4}
              valueStyle={{ color: '#1890ff' }}
              suffix="%"
              formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
            />
          </Col>
        )}
        {recall !== undefined && recall !== null && (
          <Col span={6}>
            <Statistic
              title="Recall"
              value={recall}
              precision={4}
              valueStyle={{ color: '#fa8c16' }}
              suffix="%"
              formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
            />
          </Col>
        )}
        {f1 !== undefined && f1 !== null && (
          <Col span={6}>
            <Statistic
              title="F1 Score"
              value={f1}
              precision={4}
              valueStyle={{ color: '#722ed1' }}
              suffix="%"
              formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
            />
          </Col>
        )}
      </Row>

      {/* AUROC and AUPR */}
      {(auroc || aupr) && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          {auroc && (
            <Col span={12}>
              <Statistic
                title="AUROC"
                value={auroc}
                precision={4}
                valueStyle={{ color: '#13c2c2' }}
              />
            </Col>
          )}
          {aupr && (
            <Col span={12}>
              <Statistic
                title="AUPR"
                value={aupr}
                precision={4}
                valueStyle={{ color: '#eb2f96' }}
              />
            </Col>
          )}
        </Row>
      )}

      {/* Train-specific highlights */}
      {type === 'train' && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          {trainMetrics?.best_val_acc !== undefined && (
            <Col span={12}>
              <Statistic
                title="Best Val Acc"
                value={trainMetrics.best_val_acc}
                precision={4}
                valueStyle={{ color: '#2f54eb' }}
                suffix="%"
                formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
              />
            </Col>
          )}
          {trainMetrics?.test_acc !== undefined && (
            <Col span={12}>
              <Statistic
                title="Train Test Acc"
                value={trainMetrics.test_acc}
                precision={4}
                valueStyle={{ color: '#08979c' }}
                suffix="%"
                formatter={(value) => `${(Number(value) * 100).toFixed(2)}`}
              />
            </Col>
          )}
        </Row>
      )}

      {/* Confusion Matrix */}
      {confusion && (
        <Card
          title="Confusion Matrix"
          size="small"
          style={{ marginTop: 16, background: '#fff' }}
        >
          <Table
            dataSource={[
              {
                key: 'positive',
                actual: 'Positive',
                predictedPositive: confusion.tp,
                predictedNegative: confusion.fn,
              },
              {
                key: 'negative',
                actual: 'Negative',
                predictedPositive: confusion.fp,
                predictedNegative: confusion.tn,
              },
            ]}
            columns={[
              {
                title: 'Actual / Predicted',
                dataIndex: 'actual',
                key: 'actual',
                width: 150,
              },
              {
                title: 'Positive',
                dataIndex: 'predictedPositive',
                key: 'predictedPositive',
                align: 'center',
                render: (value) => <Tag color="green">{value}</Tag>,
              },
              {
                title: 'Negative',
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
        <Card title="Prediction Preview" size="small" style={{ marginTop: 16, background: '#fff' }}>
          <Table
            dataSource={predictionsPreview.map((row: any, idx: number) => ({
              key: String(row.id ?? idx),
              text: String(row.text ?? ''),
              pred: row.pred,
              prob: row.prob,
            }))}
            columns={[
              {
                title: 'Text',
                dataIndex: 'text',
                key: 'text',
                render: (value: string) => value || '(empty)',
              },
              {
                title: 'Pred',
                dataIndex: 'pred',
                key: 'pred',
                width: 90,
                render: (value: any) => (
                  <Tag color={Number(value) === 1 ? 'red' : 'green'}>
                    {Number(value) === 1 ? 'Machine' : 'Human'}
                  </Tag>
                ),
              },
              {
                title: 'Prob',
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

      {/* Additional Info */}
      {(evalSummary?.meta || results?.manifest) && (
        <Descriptions
          title="Detector Information"
          bordered
          size="small"
          style={{ marginTop: 16 }}
        >
          {evalSummary?.detector && (
            <Descriptions.Item label="Detector">{evalSummary.detector}</Descriptions.Item>
          )}
          {evalSummary?.meta?.detector_type && (
            <Descriptions.Item label="Type">{evalSummary.meta.detector_type}</Descriptions.Item>
          )}
          {evalSummary?.meta?.dev?.num_samples && (
            <Descriptions.Item label="Samples">{evalSummary.meta.dev.num_samples}</Descriptions.Item>
          )}
          {results?.manifest?.timing?.evaluate_sec && (
            <Descriptions.Item label="Eval Time (s)">
              {Number(results.manifest.timing.evaluate_sec).toFixed(3)}
            </Descriptions.Item>
          )}
        </Descriptions>
      )}
    </Card>
  );
};
