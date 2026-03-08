/**
 * Detect Section Component - Fully Dynamic
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Button, Card, Col, Divider, Form, InputNumber, message, Row, Select, Space, Tag, Typography } from 'antd';
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
import { ThresholdPresetSelector } from '../Shared/ThresholdPresetSelector';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useUILanguage } from '../../hooks/useUILanguage';
import { CalibratorThresholdPreset } from '../../types';
import api from '../../services/api';
import {
  DETECTOR_INFO,
  DetectorInfo,
  formatDetectorLabel,
  formatDetectorVenue,
  getDetectorVenueTagColor,
  hasDetectorVenue,
  mergeDetectorInfo,
} from './detectorInfo';
import { START_SECTION_EVENT, StartSectionEventDetail } from '../../constants/jobControls';

// Helper function to split config keys for balanced layout
const splitConfigKeys = (config: any, mainKeys: string[]): { leftKeys: string[]; rightKeys: string[] } => {
  if (!config) return { leftKeys: [], rightKeys: [] };

  const allKeys = Object.keys(config);
  const otherKeys = allKeys.filter(key => !mainKeys.includes(key));

  const midpoint = Math.ceil(otherKeys.length / 2);
  const leftKeys = otherKeys.slice(0, midpoint);
  const rightKeys = otherKeys.slice(midpoint);

  return { leftKeys, rightKeys };
};

const filterThresholdPresets = (presets: CalibratorThresholdPreset[]): CalibratorThresholdPreset[] => {
  return presets.filter((item) => {
    const key = String(item?.key || '').toLowerCase();
    const source = String(item?.source || '').toLowerCase();
    const label = String(item?.label || '').toLowerCase();
    if (key.includes('raw_p05')) return false;
    if (source.includes('threshold_raw_p05')) return false;
    if (label.includes('raw_p05')) return false;
    if (key.includes('decision_boundary_raw_at_prob_0.5')) return false;
    if (source.includes('decision_boundary_raw_at_prob_0.5')) return false;
    if (label.includes('decision_boundary_raw_at_prob_0.5')) return false;
    return true;
  });
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
  const [thresholdPresetsByPath, setThresholdPresetsByPath] = useState<Record<string, CalibratorThresholdPreset[]>>({});
  const [defaultThresholdByPath, setDefaultThresholdByPath] = useState<Record<string, number>>({});
  const [thresholdPresetLoading, setThresholdPresetLoading] = useState(false);
  const [selectedThresholdPreset, setSelectedThresholdPreset] = useState<string>();
  const resultFetchedRef = React.useRef<string | null>(null);
  const { language, t } = useUILanguage();

  const isLikelyInChina = useMemo(() => {
    const browserLanguage = navigator.language || '';
    return browserLanguage.toLowerCase().startsWith('zh');
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

  useEffect(() => {
    const handleStartRequest = (event: Event) => {
      const detail = (event as CustomEvent<StartSectionEventDetail>).detail;
      if (detail?.section !== 'detect') {
        return;
      }
      if (isDetectRunning || loading) {
        return;
      }
      form.submit();
    };

    window.addEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    return () => {
      window.removeEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    };
  }, [form, isDetectRunning, loading]);

  const detectorCalibratorPath = Form.useWatch(['detector_kwargs', 'calibrator_path'], form);
  const rootCalibratorPath = Form.useWatch(['calibrator_path'], form);
  const calibratorPath = (detectorCalibratorPath || rootCalibratorPath || '').toString().trim();

  const thresholdValue = Form.useWatch(['threshold'], form);

  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['detector', 'gpu_ids', 'threshold', 'mode', 'data'];
    return splitConfigKeys(templateConfig, mainKeys);
  }, [templateConfig]);

  const thresholdPresets = useMemo(() => {
    if (!calibratorPath) {
      return [];
    }
    return thresholdPresetsByPath[calibratorPath] || [];
  }, [calibratorPath, thresholdPresetsByPath]);

  useEffect(() => {
    const loadDetectors = async () => {
      try {
        const [list, metadata] = await Promise.all([
          api.getDetectDetectors(),
          api.getDetectorMetadata(),
        ]);
        setDetectors(list);
        setDetectorInfoMap(mergeDetectorInfo(metadata?.detectors || []));
      } catch (_error) {
        message.error(t('detectLoadDetectorsFailed'));
      }
    };
    loadDetectors();
  }, [t]);

  useEffect(() => {
    const loadThresholdPresets = async () => {
      if (!calibratorPath) {
        setSelectedThresholdPreset(undefined);
        return;
      }
      if (thresholdPresetsByPath[calibratorPath]) {
        return;
      }
      setThresholdPresetLoading(true);
      try {
        const response = await api.getCalibratorThresholds(calibratorPath);
        const presets = filterThresholdPresets(
          Array.isArray(response?.presets) ? response.presets : [],
        );
        setThresholdPresetsByPath((prev) => ({
          ...prev,
          [calibratorPath]: presets,
        }));
        if (typeof response?.default_threshold === 'number') {
          setDefaultThresholdByPath((prev) => ({
            ...prev,
            [calibratorPath]: response.default_threshold,
          }));
          if (thresholdValue === undefined || thresholdValue === null || thresholdValue === '') {
            form.setFieldValue('threshold', response.default_threshold);
          }
        }
      } catch (_error) {
        setThresholdPresetsByPath((prev) => ({
          ...prev,
          [calibratorPath]: [],
        }));
      } finally {
        setThresholdPresetLoading(false);
      }
    };
    loadThresholdPresets();
  }, [calibratorPath, thresholdPresetsByPath, form, thresholdValue]);

  const handleDetectorChange = async (detector: string) => {
    setSelectedDetector(detector);
    clearDetectLogs();
    setDetectResult(null);
    setSelectedThresholdPreset(undefined);
    try {
      const template = await api.getDetectTemplate(detector);
      setTemplateConfig(template);
      form.setFieldsValue(template);
    } catch (_error) {
      message.error(t('detectLoadTemplateFailed'));
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      const detectorValue = templateConfig?.detector ?? values.detector;
      if (!selectedDetector || !detectorValue) {
        message.error(t('detectSelectRequired'));
        setLoading(false);
        return;
      }

      const config = {
        ...values,
        detector: detectorValue,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };

      const validation = await api.validateDetectConfig(config);
      if (!validation.valid) {
        message.error(`${t('detectValidationFailed')}: ${validation.errors.join(', ')}`);
        setLoading(false);
        return;
      }

      clearDetectLogs();
      setDetectResult(null);
      resultFetchedRef.current = null;

      const result = await api.executeDetect(config);
      startDetect(result.job_id);
      message.success(t('detectStarted'));
    } catch (error: any) {
      message.error(error.response?.data?.detail || t('detectStartFailed'));
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
      description: t('detectNoDescription'),
    };
  }, [selectedDetector, detectorInfoMap, t]);

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
      } catch (_error) {
        message.warning(t('detectResultMissing'));
      } finally {
        setResultLoading(false);
      }
    };
    fetchResult();
  }, [detectJobId, isDetectRunning, setDetectResult, t]);

  const handleApplyThresholdPreset = () => {
    if (!selectedThresholdPreset) {
      return;
    }
    const preset = thresholdPresets.find((item) => `${item.key}@@${item.threshold}` === selectedThresholdPreset);
    if (!preset) {
      return;
    }
    form.setFieldValue('threshold', preset.threshold);
  };

  const handleUseDefaultThreshold = () => {
    if (!calibratorPath) {
      return;
    }
    const threshold = defaultThresholdByPath[calibratorPath];
    if (typeof threshold === 'number') {
      form.setFieldValue('threshold', threshold);
    }
  };

  const handleStop = async () => {
    if (!detectJobId) {
      return;
    }
    try {
      await api.cancelJob(detectJobId);
      addDetectLog({
        level: 'warning',
        message: t('detectCancelRequested'),
        timestamp: new Date().toISOString(),
      });
      stopDetect();
      message.info(t('detectCancelRequested'));
    } catch (error: any) {
      message.error(error.response?.data?.detail || t('detectCancelFailed'));
    }
  };

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        <Col span={12}>
          <Card title={t('detectConfigTitle')}>
            <Form.Item
              label={t('detectSelectDetector')}
              extra={<FieldHelpText path="detector" value={selectedDetector} />}
            >
              <Select
                value={selectedDetector}
                onChange={handleDetectorChange}
                placeholder={t('detectChooseDetector')}
                disabled={isDetectRunning}
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
                          <strong>{t('detectPaper')}:</strong> {detectorInfo.paper}
                        </Typography.Text>
                      )}
                      {detectorInfo.authors && (
                        <Typography.Text style={{ fontSize: 13, color: '#595959' }}>
                          <TeamOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong>{t('detectAuthors')}:</strong> {detectorInfo.authors}
                        </Typography.Text>
                      )}
                      <div>
                        <TrophyOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                        <strong style={{ fontSize: 13 }}>{t('detectVenue')}:</strong>{' '}
                        <Tag color="purple" style={{ fontSize: 12 }}>{detectorInfo.venue || 'N/A'}</Tag>
                      </div>
                      {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                        <div>
                          <LinkOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                          <strong style={{ fontSize: 13 }}>{t('detectLink')}:</strong>{' '}
                          <Typography.Link href={detectorInfo.link} target="_blank" style={{ fontSize: 13 }}>
                            {detectorInfo.link}
                          </Typography.Link>
                        </div>
                      )}
                    </Space>
                  </Card>
                )}
                <Divider orientation="left">{t('detectSystemResources')}</Divider>

                <Form.Item
                  name="gpu_ids"
                  label={t('detectGpuSelection')}
                  extra={<FieldHelpText path="gpu_ids" value={form.getFieldValue('gpu_ids')} />}
                >
                  <GPUSelector mode="multiple" />
                </Form.Item>

                <Form.Item
                  label={t('detectHfSource')}
                  extra={<FieldHelpText path="hf_endpoint" value={hfEndpoint} />}
                >
                  <Select value={hfEndpoint} onChange={setHfEndpoint}>
                    <Select.Option value="">{t('detectHfSourceOfficial')}</Select.Option>
                    <Select.Option value="https://hf-mirror.com">{t('detectHfSourceMirror')}</Select.Option>
                    <Select.Option value="modelscope">{t('detectHfSourceModelScope')}</Select.Option>
                  </Select>
                  <HFMirrorSuggestion
                    language={language}
                    show={isLikelyInChina && !hfEndpoint}
                    onUseMirror={() => setHfEndpoint('https://hf-mirror.com')}
                  />
                </Form.Item>

                <HFTokenInput disabled={isDetectRunning} />

                <Divider orientation="left">Dataset Configuration</Divider>

                <Form.Item
                  name="data"
                  label="Evaluation Dataset"
                  extra={<FieldHelpText path="data" value={form.getFieldValue('data')} />}
                  rules={[{ required: true, message: 'Evaluation dataset is required' }]}
                >
                  <DatasetUploadInput
                    phase="detect"
                    disabled={isDetectRunning || loading}
                    placeholder="Upload evaluation dataset or enter existing path"
                  />
                </Form.Item>

                <Form.Item
                  name="threshold"
                  label={t('detectThreshold')}
                  extra={<FieldHelpText path="threshold" value={form.getFieldValue('threshold')} />}
                >
                  <InputNumber
                    style={{ width: '100%' }}
                    min={-1e9}
                    max={1e9}
                    step={0.0001}
                    precision={6}
                    placeholder={t('detectThresholdManualPlaceholder')}
                  />
                </Form.Item>

                <Form.Item
                  label={t('detectThresholdPreset')}
                  extra={<FieldHelpText path="threshold_preset" value={selectedThresholdPreset || ''} />}
                >
                  <ThresholdPresetSelector
                    language={language}
                    presets={thresholdPresets}
                    loading={thresholdPresetLoading}
                    selectedPreset={selectedThresholdPreset}
                    onSelectedPresetChange={setSelectedThresholdPreset}
                    onApplySelectedPreset={handleApplyThresholdPreset}
                    calibratorPath={calibratorPath}
                    defaultThreshold={calibratorPath ? defaultThresholdByPath[calibratorPath] : undefined}
                    onApplyDefaultThreshold={handleUseDefaultThreshold}
                    selectPlaceholder={t('detectThresholdPresetPlaceholder')}
                    applyPresetLabel={t('detectApplyPreset')}
                    noPresetLabel={t('detectNoPreset')}
                    noCalibratorLabel={language === 'zh' ? '请先在高级配置中选择校准器路径。' : 'Please select a calibrator path in advanced configuration first.'}
                  />
                </Form.Item>

                {leftKeys.length > 0 && (
                  <>
                    <Divider orientation="left">{t('detectConfigPart1')}</Divider>
                    <DynamicFormFields
                      data={templateConfig}
                      includeKeys={leftKeys}
                      excludeKeys={['threshold', 'mode']}
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
                      {isDetectRunning ? t('detectRunning') : t('detectExecute')}
                    </Button>
                    <Button
                      danger
                      onClick={handleStop}
                      disabled={!isDetectRunning}
                      size="large"
                      style={{ flex: 1 }}
                    >
                      {t('detectStop')}
                    </Button>
                  </div>
                </Form.Item>
              </>
            )}
          </Card>
        </Col>

        <Col span={12}>
          <ModelDownloadStatus logs={detectLogs} isRunning={isDetectRunning} />
          {resultLoading && !detectResult && (
            <Card title={t('detectResultsTitle')} loading style={{ marginBottom: 16 }} />
          )}
          {detectResult?.result && (
            <ResultsDisplay results={detectResult.result} type="detect" />
          )}
          <Card title={t('detectLogsTitle')} style={{ marginBottom: 16 }}>
            <LogViewer logs={detectLogs} isRunning={isDetectRunning} />
          </Card>

          {selectedDetector && templateConfig && rightKeys.length > 0 && (
            <Card title={t('detectConfigPart2')}>
              <DynamicFormFields
                data={templateConfig}
                includeKeys={rightKeys}
                excludeKeys={['threshold', 'mode']}
              />
            </Card>
          )}
        </Col>
      </Row>
    </Form>
  );
};
