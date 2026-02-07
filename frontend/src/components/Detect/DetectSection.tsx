/**
 * Detect Section Component - Fully Dynamic
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
import { DETECTOR_INFO, DetectorInfo, formatDetectorLabel, mergeDetectorInfo } from './detectorInfo';

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

export const DetectSection: React.FC = () => {
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
    detectLogs,
    detectJobId,
    isDetectRunning,
    detectResult,
    startDetect,
    clearDetectLogs,
    stopDetect,
    addDetectLog,
    setDetectResult,
    hfToken,
  } = useStore();

  useWebSocket({ jobId: detectJobId, section: 'detect', isRunning: isDetectRunning });

  // Dynamically split config keys for balanced layout
  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['detector', 'gpu_ids'];
    return splitConfigKeys(templateConfig, mainKeys);
  }, [templateConfig]);

  // Load detectors
  useEffect(() => {
    const loadDetectors = async () => {
      try {
        const [list, metadata] = await Promise.all([
          api.getDetectDetectors(),
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
    // Clear logs and results when switching detectors
    clearDetectLogs();
    setDetectResult(null);
    try {
      const template = await api.getDetectTemplate(detector);
      setTemplateConfig(template);
      form.setFieldsValue(template);
    } catch (error) {
      message.error('Failed to load detector template');
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      const detectorValue = templateConfig?.detector ?? values.detector;
      if (!selectedDetector || !detectorValue) {
        message.error('Please select a detector');
        setLoading(false);
        return;
      }

      const config = {
        ...values,
        detector: detectorValue,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };

      // Validate
      const validation = await api.validateDetectConfig(config);
      if (!validation.valid) {
        message.error(`Validation failed: ${validation.errors.join(', ')}`);
        setLoading(false);
        return;
      }

      // Clear logs
      clearDetectLogs();
      setDetectResult(null);
      resultFetchedRef.current = null;

      // Execute
      const result = await api.executeDetect(config);
      startDetect(result.job_id);
      message.success('Detect job started');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to start detection');
    } finally {
      setLoading(false);
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
      if (!detectJobId || isDetectRunning) {
        return;
      }
      if (resultFetchedRef.current === detectJobId) {
        return;
      }
      resultFetchedRef.current = detectJobId;
      setResultLoading(true);
      try {
        const response = await api.getJobResult(detectJobId);
        setDetectResult(response);
      } catch (error) {
        message.warning('Detect finished, but no result artifact was found.');
      } finally {
        setResultLoading(false);
      }
    };
    fetchResult();
  }, [detectJobId, isDetectRunning, setDetectResult]);

  const handleStop = async () => {
    if (!detectJobId) {
      return;
    }
    try {
      await api.cancelJob(detectJobId);
      addDetectLog({
        level: 'warning',
        message: 'Cancellation requested',
        timestamp: new Date().toISOString(),
      });
      stopDetect();
      message.info('Cancellation requested');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to cancel detection');
    }
  };

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        {/* Left Column: Main Configuration + First Half of Parameters */}
        <Col span={12}>
          <Card title="Detect Configuration">
            <Form.Item
              label="Select Detector"
              extra={<FieldHelpText path="detector" value={selectedDetector} />}
            >
              <Select
                value={selectedDetector}
                onChange={handleDetectorChange}
                placeholder="Choose a detector..."
                disabled={isDetectRunning}
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
                  <Card size="small" style={{ marginBottom: 16 }}>
                    <Typography.Title level={5} style={{ marginBottom: 8 }}>
                      {detectorInfo.name}
                    </Typography.Title>
                    <Typography.Paragraph style={{ marginBottom: 0 }}>
                      {detectorInfo.description}
                    </Typography.Paragraph>
                    {detectorInfo.paper && (
                      <Typography.Paragraph style={{ marginBottom: 0, marginTop: 8 }}>
                        <strong>Paper:</strong> {detectorInfo.paper}
                      </Typography.Paragraph>
                    )}
                    {detectorInfo.authors && (
                      <Typography.Paragraph style={{ marginBottom: 0 }}>
                        <strong>Authors:</strong> {detectorInfo.authors}
                      </Typography.Paragraph>
                    )}
                    {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                      <Typography.Paragraph style={{ marginBottom: 0 }}>
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
                <HFTokenInput disabled={isDetectRunning} />

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
                      loading={loading || isDetectRunning}
                      size="large"
                      style={{ flex: 1 }}
                    >
                      {isDetectRunning ? 'Detecting...' : 'Execute Detection'}
                    </Button>
                    <Button
                      danger
                      onClick={handleStop}
                      disabled={!isDetectRunning}
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
          <ModelDownloadStatus logs={detectLogs} isRunning={isDetectRunning} />
          {resultLoading && !detectResult && (
            <Card title="Detection Results" loading style={{ marginBottom: 16 }} />
          )}
          {detectResult?.result && (
            <ResultsDisplay results={detectResult.result} type="detect" />
          )}
          <Card title="Detect Logs" style={{ marginBottom: 16 }}>
            <LogViewer logs={detectLogs} isRunning={isDetectRunning} />
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
