/**
 * Attack Section Component - Fully Dynamic with Unique Keys
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Form, Input, Button, Checkbox, message, Divider, Select } from 'antd';
import { useStore } from '../../store';
import { LogViewer } from '../Shared/LogViewer';
import { GPUSelector } from '../Shared/GPUSelector';
import { AttackConfigEditor } from './AttackConfigEditor';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { useWebSocket } from '../../hooks/useWebSocket';
import api from '../../services/api';
import { formatAttackLabel } from './attackLabels';

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
  const [attackOptions, setAttackOptions] = useState<Array<{ key: string; label: string; attack: any }>>([]);
  const [hfEndpoint, setHfEndpoint] = useState<string>('');

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

  useWebSocket({ jobId: attackJobId, section: 'attack' });

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
          const options: Array<{ key: string; label: string; attack: any }> = [];

          attacks.text_attacks.forEach((attack: any) => {
            // Create unique key: type or type-backend
            const key = attack.backend ? `${attack.type}-${attack.backend}` : attack.type;

            const label = formatAttackLabel(attack.type, attack.backend);

            map.set(key, attack);
            options.push({ key, label, attack });
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

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        {/* Left Column: Main Configuration + First Half of Parameters */}
        <Col span={12}>
          <Card title="Attack Configuration">
            <Divider orientation="left">System Resources</Divider>

            <Form.Item name="gpu_ids" label="GPU Selection">
              <GPUSelector mode="multiple" />
            </Form.Item>

            <Form.Item label="HF Download Source">
              <Select value={hfEndpoint} onChange={setHfEndpoint}>
                <Select.Option value="">Official (huggingface.co)</Select.Option>
                <Select.Option value="https://hf-mirror.com">HF Mirror (hf-mirror.com)</Select.Option>
              </Select>
            </Form.Item>
            <HFTokenInput disabled={isAttackRunning} />

            <Divider orientation="left">Dataset Configuration</Divider>

            <Form.Item
              name="data"
              label="Input Data"
              rules={[{ required: true, message: 'Input data is required' }]}
            >
              <Input placeholder="data/input.jsonl" />
            </Form.Item>

            <Form.Item
              name="out"
              label="Output File"
              rules={[{ required: true, message: 'Output file is required' }]}
            >
              <Input placeholder="data/output.attack.jsonl" />
            </Form.Item>

            <Divider orientation="left">Attack Selection</Divider>

            <Form.Item label="Select Attack Types">
              <Checkbox.Group
                value={selectedAttacks}
                onChange={(values) => setSelectedAttacks(values as string[])}
              >
                <Row>
                  {attackOptions.map(({ key, label }) => (
                    <Col span={12} key={key} style={{ marginBottom: 8 }}>
                      <Checkbox value={key}>{label}</Checkbox>
                    </Col>
                  ))}
                </Row>
              </Checkbox.Group>
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
                />
              </>
            )}
          </Card>
        </Col>
      </Row>
    </Form>
  );
};
