/**
 * Train Section Component - Fully Dynamic
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Form, Button, Select, message, Divider, Typography } from 'antd';
import { useStore } from '../../store';
import { LogViewer } from '../Shared/LogViewer';
import { GPUSelector } from '../Shared/GPUSelector';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { ResultsDisplay } from '../Shared/ResultsDisplay';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { HFMirrorSuggestion } from '../Shared/HFMirrorSuggestion';
import { FieldHelpText } from '../Shared/FieldHelpText';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useUILanguage } from '../../hooks/useUILanguage';
import api from '../../services/api';
import { DETECTOR_INFO, DetectorInfo, formatDetectorLabel, mergeDetectorInfo } from '../Detect/detectorInfo';

// Helper function to split config keys for balanced layout
const splitConfigKeys = (config: any, mainKeys: string[]): { leftKeys: string[], rightKeys: string[] } => {
  if (!config) return { leftKeys: [], rightKeys: [] };

  const allKeys = Object.keys(config);
  const otherKeys = allKeys.filter(key => !mainKeys.includes(key));

  // Split remaining keys roughly in half
  const midpoint = Math.ceil(otherKeys.length / 2);
  const leftKeys = otherKeys.slice(0, midpoint);
  const rightKeys = otherKeys.slice(midpoint);

  return { leftKeys, rightKeys };
};

export const TrainSection: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [detectors, setDetectors] = useState<string[]>([]);
  const [selectedDetector, setSelectedDetector] = useState<string | null>(null);
  const [templateConfig, setTemplateConfig] = useState<any>(null);
  const [hfEndpoint, setHfEndpoint] = useState<string>('');
  const [detectorInfoMap, setDetectorInfoMap] = useState<Record<string, DetectorInfo>>(DETECTOR_INFO);
  const [resultLoading, setResultLoading] = useState(false);
  const resultFetchedRef = React.useRef<string | null>(null);
  const { language } = useUILanguage();

  // Detect if user is likely in China (based on browser language)
  const isLikelyInChina = useMemo(() => {
    const language = navigator.language || '';
    return language.toLowerCase().startsWith('zh');
  }, []);

  const {
    trainLogs,
    trainJobId,
    isTrainRunning,
    trainResult,
    startTrain,
    clearTrainLogs,
    stopTrain,
    addTrainLog,
    setTrainResult,
    hfToken,
  } = useStore();

  useWebSocket({ jobId: trainJobId, section: 'train', isRunning: isTrainRunning });

  // Dynamically split config keys for balanced layout
  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['gpu_ids'];
    return splitConfigKeys(templateConfig, mainKeys);
  }, [templateConfig]);

  // Load detectors
  useEffect(() => {
    const loadDetectors = async () => {
      try {
        const [list, metadata] = await Promise.all([
          api.getTrainDetectors(),
          api.getDetectorMetadata(),
        ]);
        setDetectors(list);
        setDetectorInfoMap(mergeDetectorInfo(metadata?.detectors || []));
      } catch (error) {
        message.error('Failed to load detectors');
      }
    };
    loadDetectors();
  }, []);

  // Load template when detector changes
  const handleDetectorChange = async (detector: string) => {
    setSelectedDetector(detector);
    clearTrainLogs();
    setTrainResult(null);
    resultFetchedRef.current = null;
    try {
      const template = await api.getTrainTemplate(detector);
      setTemplateConfig(template);
      form.setFieldsValue(template);
    } catch (error) {
      message.error('Failed to load detector template');
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      const config = {
        ...values,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };
      // Validate
      const validation = await api.validateTrainConfig(config);
      if (!validation.valid) {
        message.error(`Validation failed: ${validation.errors.join(', ')}`);
        setLoading(false);
        return;
      }

      // Clear logs
      clearTrainLogs();
      setTrainResult(null);
      resultFetchedRef.current = null;

      // Execute
      const result = await api.executeTrain(config);
      startTrain(result.job_id);
      message.success('Train job started');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!trainJobId) {
      return;
    }
    try {
      await api.cancelJob(trainJobId);
      addTrainLog({
        level: 'warning',
        message: 'Cancellation requested',
        timestamp: new Date().toISOString(),
      });
      stopTrain();
      message.info('Cancellation requested');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to cancel training');
    }
  };

  const detectorInfo = useMemo(() => {
    if (!selectedDetector) {
      return null;
    }
    const key = selectedDetector.toLowerCase();
    return detectorInfoMap[key] || {
      name: formatDetectorLabel(selectedDetector, detectorInfoMap),
      description: 'No description available.',
    };
  }, [selectedDetector, detectorInfoMap]);

  useEffect(() => {
    const fetchResult = async () => {
      if (!trainJobId || isTrainRunning) {
        return;
      }
      if (resultFetchedRef.current === trainJobId) {
        return;
      }
      resultFetchedRef.current = trainJobId;
      setResultLoading(true);
      try {
        const response = await api.getJobResult(trainJobId);
        setTrainResult(response);
      } catch (error) {
        message.warning('Train finished, but no result artifact was found.');
      } finally {
        setResultLoading(false);
      }
    };
    fetchResult();
  }, [trainJobId, isTrainRunning, setTrainResult]);

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        {/* Left Column: Main Configuration + First Half of Parameters */}
        <Col span={12}>
          <Card title="Train Configuration">
            <Form.Item
              label="Select Detector"
              extra={<FieldHelpText path="detector" value={selectedDetector} />}
            >
              <Select
                value={selectedDetector}
                onChange={handleDetectorChange}
                placeholder="Choose a detector..."
                disabled={isTrainRunning}
              >
                {detectors.map((d) => (
                  <Select.Option key={d} value={d}>
                    {formatDetectorLabel(d, detectorInfoMap)}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            {selectedDetector && templateConfig && (
              <>
                {detectorInfo && (
                  <Card size="small" style={{ marginBottom: 16, background: '#f0f7ff', border: '1px solid #91d5ff' }}>
                    <Typography.Title level={5} style={{ marginBottom: 8, color: '#1890ff' }}>
                      {detectorInfo.name}
                    </Typography.Title>
                    <Typography.Paragraph style={{ marginBottom: 0, color: '#595959' }}>
                      {detectorInfo.description}
                    </Typography.Paragraph>
                    {detectorInfo.paper && (
                      <Typography.Paragraph style={{ marginBottom: 0, marginTop: 8, color: '#595959' }}>
                        <strong>Paper:</strong> {detectorInfo.paper}
                      </Typography.Paragraph>
                    )}
                    {detectorInfo.authors && (
                      <Typography.Paragraph style={{ marginBottom: 0, color: '#595959' }}>
                        <strong>Authors:</strong> {detectorInfo.authors}
                      </Typography.Paragraph>
                    )}
                    {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                      <Typography.Paragraph style={{ marginBottom: 0, color: '#595959' }}>
                        <strong>Link:</strong>{' '}
                        <a href={detectorInfo.link} target="_blank" rel="noreferrer">
                          {detectorInfo.link}
                        </a>
                      </Typography.Paragraph>
                    )}
                  </Card>
                )}
                <Divider orientation="left">System Resources</Divider>

                <Form.Item
                  name="gpu_ids"
                  label="GPU Selection"
                  extra={<FieldHelpText path="gpu_ids" value={form.getFieldValue('gpu_ids')} />}
                >
                  <GPUSelector mode="multiple" />
                </Form.Item>

                <Form.Item
                  label="HF Download Source"
                  extra={<FieldHelpText path="hf_endpoint" value={hfEndpoint} />}
                >
                  <Select value={hfEndpoint} onChange={setHfEndpoint}>
                    <Select.Option value="">Official (huggingface.co)</Select.Option>
                    <Select.Option value="https://hf-mirror.com">HF Mirror (hf-mirror.com)</Select.Option>
                  </Select>
                  <HFMirrorSuggestion
                    language={language}
                    show={isLikelyInChina && !hfEndpoint}
                    onUseMirror={() => setHfEndpoint('https://hf-mirror.com')}
                  />
                </Form.Item>
                <HFTokenInput disabled={isTrainRunning} />

                {leftKeys.length > 0 && (
                  <>
                    <Divider orientation="left">Configuration (Part 1)</Divider>
                    <DynamicFormFields
                      data={templateConfig}
                      includeKeys={leftKeys}
                    />
                  </>
                )}

                <Form.Item style={{ marginTop: 24 }}>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading || isTrainRunning}
                      size="large"
                      style={{ flex: 1 }}
                    >
                      {isTrainRunning ? 'Training...' : 'Execute Training'}
                    </Button>
                    <Button
                      danger
                      onClick={handleStop}
                      disabled={!isTrainRunning}
                      size="large"
                      style={{ flex: 1 }}
                    >
                      Stop
                    </Button>
                  </div>
                </Form.Item>
              </>
            )}
          </Card>
        </Col>

        {/* Right Column: Logs + Second Half of Parameters */}
        <Col span={12}>
          <ModelDownloadStatus logs={trainLogs} isRunning={isTrainRunning} />
          {resultLoading && !trainResult && (
            <Card title="Training Results" loading style={{ marginBottom: 16 }} />
          )}
          {trainResult?.result && (
            <ResultsDisplay results={trainResult.result} type="train" />
          )}
          <Card title="Train Logs" style={{ marginBottom: 16 }}>
            <LogViewer logs={trainLogs} isRunning={isTrainRunning} />
          </Card>

          {selectedDetector && templateConfig && rightKeys.length > 0 && (
            <Card title="Configuration (Part 2)">
              <DynamicFormFields
                data={templateConfig}
                includeKeys={rightKeys}
              />
            </Card>
          )}
        </Col>
      </Row>
    </Form>
  );
};
