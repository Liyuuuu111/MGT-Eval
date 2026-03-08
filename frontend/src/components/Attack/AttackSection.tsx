/**
 * Attack Section Component - Fully Dynamic with Unique Keys
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Button, Card, Checkbox, Col, Divider, Form, Input, Row, Select, message } from 'antd';
import { useStore } from '../../store';
import { LogViewer } from '../Shared/LogViewer';
import { GPUSelector } from '../Shared/GPUSelector';
import { AttackConfigEditor } from './AttackConfigEditor';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { DownloadLinksCard } from '../Shared/DownloadLinksCard';
import { DatasetUploadInput } from '../Shared/DatasetUploadInput';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { HFMirrorSuggestion } from '../Shared/HFMirrorSuggestion';
import { FieldHelpText } from '../Shared/FieldHelpText';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useUILanguage } from '../../hooks/useUILanguage';
import { JobDownloadItem } from '../../types';
import api from '../../services/api';
import { formatAttackLabel } from './attackLabels';
import { AttackMethodIntroPanel } from './AttackMethodIntroPanel';
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

export const AttackSection: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [allAttacks, setAllAttacks] = useState<any>(null);
  const [selectedAttacks, setSelectedAttacks] = useState<string[]>([]);
  const [templateConfig, setTemplateConfig] = useState<any>(null);
  const [attackDatasetOnlyValue, setAttackDatasetOnlyValue] = useState<any>(1);
  const [attackMap, setAttackMap] = useState<Map<string, any>>(new Map());
  const [attackOptions, setAttackOptions] = useState<Array<{ key: string; attack: any }>>([]);
  const [hfEndpoint, setHfEndpoint] = useState<string>('');
  const [downloads, setDownloads] = useState<JobDownloadItem[]>([]);
  const [downloadsLoading, setDownloadsLoading] = useState(false);
  const resultFetchedRef = React.useRef<string | null>(null);
  const { language } = useUILanguage();
  const watchedGpuIds = Form.useWatch('gpu_ids', form);

  const isLikelyInChina = useMemo(() => {
    const nav = window.navigator;
    const langs = [nav.language, ...(nav.languages || [])]
      .filter(Boolean)
      .map((s) => String(s).toLowerCase());
    const hasZhLocale = langs.some((l) => l.startsWith('zh') || l.includes('zh-'));
    let tz = '';
    try {
      tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    } catch {
      tz = '';
    }
    const tzLower = tz.toLowerCase();
    const chinaFriendlyTz = new Set([
      'asia/shanghai',
      'asia/chongqing',
      'asia/harbin',
      'asia/urumqi',
      'asia/hong_kong',
      'asia/macau',
      'asia/taipei',
    ]);
    return hasZhLocale || chinaFriendlyTz.has(tzLower);
  }, []);

  const {
    attackLogs,
    attackJobId,
    isAttackRunning,
    startAttack,
    clearAttackLogs,
    stopAttack,
    addAttackLog,
    hfToken,
  } = useStore();

  useWebSocket({ jobId: attackJobId, section: 'attack', isRunning: isAttackRunning });

  useEffect(() => {
    const handleStartRequest = (event: Event) => {
      const detail = (event as CustomEvent<StartSectionEventDetail>).detail;
      if (detail?.section !== 'attack') {
        return;
      }
      if (isAttackRunning || loading) {
        return;
      }
      form.submit();
    };

    window.addEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    return () => {
      window.removeEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    };
  }, [form, isAttackRunning, loading]);

  // Dynamically split config keys for balanced layout
  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['data', 'out', 'attacks_config', 'gpu_ids', 'attack_dataset_only'];
    return splitConfigKeys(templateConfig, mainKeys);
  }, [templateConfig]);

  // Load template and attacks
  useEffect(() => {
    const loadData = async () => {
      try {
        const [template, attacks] = await Promise.all([
          api.getAttackTemplate(),
          api.getAllAttacks(),
        ]);

        // Store template config (excluding attacks_config)
        const { attacks_config, attack_dataset_only, ...baseConfig } = template;
        setTemplateConfig(baseConfig);
        setAttackDatasetOnlyValue(attack_dataset_only ?? 1);

        // Set form values for base config
        form.setFieldsValue(baseConfig);

        setAllAttacks(attacks);

        // Process attacks to create unique keys
        if (attacks?.text_attacks) {
          const map = new Map();
          const options: Array<{ key: string; attack: any }> = [];

          attacks.text_attacks.forEach((attack: any) => {
            // Create unique key: type or type-backend
            const key = attack.backend ? `${attack.type}-${attack.backend}` : attack.type;

            map.set(key, attack);
            options.push({ key, attack });
          });

          setAttackMap(map);
          setAttackOptions(options);
        }
      } catch (error) {
        message.error('Failed to load data');
      }
    };
    loadData();
  }, [form]);

  useEffect(() => {
    if (!templateConfig || !Object.prototype.hasOwnProperty.call(templateConfig, 'vllm_tensor_parallel_size')) {
      return;
    }
    if (form.isFieldTouched('vllm_tensor_parallel_size')) {
      return;
    }

    const gpuCount = Array.isArray(watchedGpuIds)
      ? watchedGpuIds.filter((id) => Number.isFinite(id)).length
      : (typeof watchedGpuIds === 'number' && Number.isFinite(watchedGpuIds) ? 1 : 0);
    if (gpuCount <= 0) {
      return;
    }

    const current = Number(form.getFieldValue('vllm_tensor_parallel_size'));
    if (!Number.isFinite(current) || current !== gpuCount) {
      form.setFieldValue('vllm_tensor_parallel_size', gpuCount);
    }
  }, [templateConfig, watchedGpuIds, form]);

  // Initialize attack configs when attacks are selected
  useEffect(() => {
    if (selectedAttacks.length > 0 && attackMap.size > 0) {
      const attackConfigs: any = {};

      selectedAttacks.forEach((attackKey) => {
        const attack = attackMap.get(attackKey);
        if (attack) {
          const { type, backend, ...config } = attack;
          attackConfigs[attackKey] = config;
        }
      });

      form.setFieldsValue({ attack_configs: attackConfigs });
    }
  }, [selectedAttacks, attackMap, form]);

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      // Build attacks_config from selected attacks and their configs
      const attacks_config: any = { text_attacks: [] };

      if (selectedAttacks.length > 0) {
        selectedAttacks.forEach((attackKey) => {
          const attack = attackMap.get(attackKey);
          if (!attack) return;

          const attackConfig = values.attack_configs?.[attackKey] || {};

          // Reconstruct attack with type, backend (if exists), and config
          const reconstructed: any = {
            type: attack.type,
            ...attackConfig
          };

          // Add backend if it exists
          if (attack.backend) {
            reconstructed.backend = attack.backend;
          }

          attacks_config.text_attacks.push(reconstructed);
        });
      }

      // Build final config
      const { attack_configs, gpu_ids, ...baseConfig } = values;
      const config = {
        ...baseConfig,
        attacks_config,
        gpu_ids,
        attack_dataset_only: attackDatasetOnlyValue,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };

      // Validate
      const validation = await api.validateAttackConfig(config);
      if (!validation.valid) {
        message.error(`Validation failed: ${validation.errors.join(', ')}`);
        setLoading(false);
        return;
      }

      // Clear logs
      clearAttackLogs();
      setDownloads([]);
      resultFetchedRef.current = null;

      // Execute
      const result = await api.executeAttack(config);
      startAttack(result.job_id);
      message.success('Attack job started');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to start attack');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!attackJobId) {
      return;
    }
    try {
      await api.cancelJob(attackJobId);
      addAttackLog({
        level: 'warning',
        message: 'Cancellation requested',
        timestamp: new Date().toISOString(),
      });
      stopAttack();
      message.info('Cancellation requested');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to cancel attack');
    }
  };

  useEffect(() => {
    const fetchResult = async () => {
      if (!attackJobId || isAttackRunning) {
        return;
      }
      if (resultFetchedRef.current === attackJobId) {
        return;
      }
      resultFetchedRef.current = attackJobId;
      setDownloadsLoading(true);
      try {
        const response = await api.getJobResult(attackJobId);
        const items = Array.isArray(response?.downloads) ? response.downloads : [];
        setDownloads(items);
      } catch (_error) {
        setDownloads([]);
      } finally {
        setDownloadsLoading(false);
      }
    };
    fetchResult();
  }, [attackJobId, isAttackRunning]);

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        {/* Left Column: Main Configuration + First Half of Parameters */}
        <Col span={12}>
          <Card title="Attack Configuration">
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
            <HFTokenInput disabled={isAttackRunning} />

            <Divider orientation="left">Dataset Configuration</Divider>

            <Form.Item
              name="data"
              label="Input Data"
              extra={<FieldHelpText path="data" value={form.getFieldValue('data')} />}
              rules={[{ required: true, message: 'Input data is required' }]}
            >
              <DatasetUploadInput
                phase="attack"
                disabled={isAttackRunning || loading}
                placeholder="Upload dataset or enter existing path"
              />
            </Form.Item>

            <Form.Item
              name="out"
              label="Output File"
              extra={<FieldHelpText path="out" value={form.getFieldValue('out')} />}
              rules={[{ required: true, message: 'Output file is required' }]}
            >
              <Input placeholder="data/output.attack.jsonl" />
            </Form.Item>

            <Divider orientation="left">Attack Selection</Divider>

            <Form.Item
              label="Select Attack Types"
              extra={<FieldHelpText path="attacks_config.text_attacks" value={selectedAttacks} />}
            >
              <Checkbox.Group
                value={selectedAttacks}
                onChange={(values) => setSelectedAttacks(values as string[])}
              >
                <Row>
                  {attackOptions.map(({ key, attack }) => (
                    <Col span={12} key={key} style={{ marginBottom: 8 }}>
                      <Checkbox value={key}>
                        {formatAttackLabel(attack?.type, attack?.backend, language)}
                      </Checkbox>
                    </Col>
                  ))}
                </Row>
              </Checkbox.Group>
            </Form.Item>

            <AttackMethodIntroPanel
              selectedAttacks={selectedAttacks}
              attackMap={attackMap}
              language={language}
            />

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
                  loading={loading || isAttackRunning}
                  size="large"
                  style={{ flex: 1 }}
                >
                  {isAttackRunning ? 'Attacking...' : 'Execute Attack'}
                </Button>
                <Button
                  danger
                  onClick={handleStop}
                  disabled={!isAttackRunning}
                  size="large"
                  style={{ flex: 1 }}
                >
                  Stop
                </Button>
              </div>
            </Form.Item>
          </Card>
        </Col>

        {/* Right Column: Logs + Second Half of Parameters */}
        <Col span={12}>
          <ModelDownloadStatus logs={attackLogs} isRunning={isAttackRunning} />
          {!isAttackRunning && (attackJobId || downloadsLoading || downloads.length > 0) && (
            <DownloadLinksCard
              title="Generated Attack Dataset Downloads"
              downloads={downloads}
              loading={downloadsLoading}
              emptyText="No downloadable attack output was generated for this run."
            />
          )}
          <Card title="Attack Logs" style={{ marginBottom: 16 }}>
            <LogViewer logs={attackLogs} isRunning={isAttackRunning} />
          </Card>

          <Card title="Configuration (Part 2)">
            {templateConfig && rightKeys.length > 0 && (
              <DynamicFormFields
                data={templateConfig}
                includeKeys={rightKeys}
              />
            )}

            {selectedAttacks.length > 0 && (
              <>
                <Divider orientation="left">Attack Configuration</Divider>
                <AttackConfigEditor
                  selectedAttacks={selectedAttacks}
                  allAttacks={allAttacks}
                  attackMap={attackMap}
                  language={language}
                />
              </>
            )}
          </Card>
        </Col>
      </Row>
    </Form>
  );
};
