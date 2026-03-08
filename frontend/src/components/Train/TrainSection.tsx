/**
 * Train Section Component - Fully Dynamic
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Form, Button, Select, Input, message, Divider, Space, Tag, Typography } from 'antd';
import { FileTextOutlined, TeamOutlined, TrophyOutlined, LinkOutlined } from '@ant-design/icons';
import { useStore } from '../../store';
import { LogViewer } from '../Shared/LogViewer';
import { GPUSelector } from '../Shared/GPUSelector';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { ResultsDisplay } from '../Shared/ResultsDisplay';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { HFMirrorSuggestion } from '../Shared/HFMirrorSuggestion';
import { FieldHelpText } from '../Shared/FieldHelpText';
import { DatasetUploadInput } from '../Shared/DatasetUploadInput';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useUILanguage } from '../../hooks/useUILanguage';
import { getCoreText } from '../../i18n/coreText';
import api from '../../services/api';
import {
  DETECTOR_INFO,
  DetectorInfo,
  formatDetectorLabel,
  formatDetectorVenue,
  getDetectorVenueTagColor,
  hasDetectorVenue,
  mergeDetectorInfo,
} from '../Detect/detectorInfo';
import { START_SECTION_EVENT, StartSectionEventDetail } from '../../constants/jobControls';

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

  useEffect(() => {
    const handleStartRequest = (event: Event) => {
      const detail = (event as CustomEvent<StartSectionEventDetail>).detail;
      if (detail?.section !== 'train') {
        return;
      }
      if (isTrainRunning || loading) {
        return;
      }
      form.submit();
    };

    window.addEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    return () => {
      window.removeEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    };
  }, [form, isTrainRunning, loading]);

  // Dynamically split config keys for balanced layout
  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['gpu_ids', 'dataset_train', 'dataset_test', 'dataset_val'];
    const result = splitConfigKeys(templateConfig, mainKeys);
    // model1 is unused/irrelevant in the finetuned training template
    if (selectedDetector === 'finetuned') {
      return {
        leftKeys: result.leftKeys.filter((k) => k !== 'model1'),
        rightKeys: result.rightKeys.filter((k) => k !== 'model1'),
      };
    }
    return result;
  }, [templateConfig, selectedDetector]);

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
      // Important: clear stale values from previous detector templates.
      // Otherwise hidden legacy keys (e.g. sample_k) may leak into submit payload.
      form.resetFields();
      form.setFieldsValue(template);
    } catch (error) {
      message.error('Failed to load detector template');
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      const config: any = {
        ...values,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };
      // Prefer train-specific sampling keys; ignore legacy generic sample_k if both exist.
      if (config.sample_k_train !== undefined) {
        delete config.sample_k;
      }
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
                  <Select.Option key={d} value={d} label={formatDetectorLabel(d, detectorInfoMap)}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                      <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {formatDetectorLabel(d, detectorInfoMap)}
                      </span>
                      <Tag
                        color={hasDetectorVenue(d, detectorInfoMap) ? getDetectorVenueTagColor(d, detectorInfoMap) : 'default'}
                        style={{
                          margin: 0,
                          borderRadius: 999,
                          fontSize: 12,
                          lineHeight: '18px',
                          paddingInline: 9,
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {hasDetectorVenue(d, detectorInfoMap) ? formatDetectorVenue(d, detectorInfoMap) : 'Baseline'}
                      </Tag>
                    </div>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            {selectedDetector && templateConfig && (
              <>
                {detectorInfo && (
                  <Card
                    size="small"
                    style={{
                      marginBottom: 16,
                      borderRadius: 12,
                      background: 'linear-gradient(135deg, #f5f0ff 0%, #e8f4fd 100%)',
                      border: '1px solid #d3adf7',
                      boxShadow: '0 2px 8px rgba(114, 46, 209, 0.08)',
                    }}
                  >
                    <Space direction="vertical" size={6} style={{ width: '100%' }}>
                      <Typography.Title level={5} style={{ margin: 0, color: '#5b21b6' }}>
                        {detectorInfo.name}
                      </Typography.Title>
                      <Typography.Text style={{ color: '#595959', fontSize: 13 }}>
                        {detectorInfo.description}
                      </Typography.Text>
                      <Divider style={{ margin: '8px 0' }} />
                      {detectorInfo.paper && (
                        <Typography.Text style={{ fontSize: 13, color: '#595959' }}>
                          <FileTextOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong>{getCoreText(language, 'detectPaper')}:</strong> {detectorInfo.paper}
                        </Typography.Text>
                      )}
                      {detectorInfo.authors && (
                        <Typography.Text style={{ fontSize: 13, color: '#595959' }}>
                          <TeamOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong>{getCoreText(language, 'detectAuthors')}:</strong> {detectorInfo.authors}
                        </Typography.Text>
                      )}
                      {detectorInfo.venue && (
                        <div>
                          <TrophyOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong style={{ fontSize: 13 }}>{getCoreText(language, 'detectVenue')}:</strong>{' '}
                          <Tag color="purple" style={{ fontSize: 12 }}>{detectorInfo.venue}</Tag>
                        </div>
                      )}
                      {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                        <div>
                          <LinkOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong style={{ fontSize: 13 }}>{getCoreText(language, 'detectLink')}:</strong>{' '}
                          <Typography.Link href={detectorInfo.link} target="_blank" style={{ fontSize: 13 }}>
                            {detectorInfo.link}
                          </Typography.Link>
                        </div>
                      )}
                    </Space>
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
                    <Select.Option value="modelscope">ModelScope (modelscope.cn)</Select.Option>
                  </Select>
                  <HFMirrorSuggestion
                    language={language}
                    show={isLikelyInChina && !hfEndpoint}
                    onUseMirror={() => setHfEndpoint('https://hf-mirror.com')}
                  />
                </Form.Item>
                <HFTokenInput disabled={isTrainRunning} />

                <Divider orientation="left">Dataset Configuration</Divider>
                <Form.Item
                  name="dataset_train"
                  label="Training Dataset"
                  extra={<FieldHelpText path="dataset_train" value={form.getFieldValue('dataset_train')} />}
                  rules={[{ required: true, message: 'Training dataset is required' }]}
                >
                  <DatasetUploadInput
                    phase="train"
                    disabled={isTrainRunning || loading}
                    placeholder="Upload training dataset or enter existing path"
                  />
                </Form.Item>

                <Form.Item
                  name="dataset_test"
                  label="Evaluation Dataset (Optional)"
                  extra={<FieldHelpText path="dataset_test" value={form.getFieldValue('dataset_test')} />}
                >
                  <DatasetUploadInput
                    phase="train"
                    disabled={isTrainRunning || loading}
                    placeholder="Upload evaluation dataset or enter existing path"
                  />
                </Form.Item>

                <Form.Item
                  name="dataset_val"
                  label="Validation Dataset (Optional)"
                  extra={<FieldHelpText path="dataset_val" value={form.getFieldValue('dataset_val')} />}
                >
                  <Input placeholder="Optional validation dataset path" />
                </Form.Item>

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
            <ResultsDisplay results={trainResult.result} type="train" logs={trainLogs} />
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
