/**
 * Build Section Component - Full Dynamic Form
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Form, Input, Button, message, Divider, Select } from 'antd';
import { useStore } from '../../store';
import { LogViewer } from '../Shared/LogViewer';
import { GPUSelector } from '../Shared/GPUSelector';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { useWebSocket } from '../../hooks/useWebSocket';
import api from '../../services/api';

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

export const BuildSection: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [templateConfig, setTemplateConfig] = useState<any>(null);
  const [hfEndpoint, setHfEndpoint] = useState<string>('');

  const {
    buildLogs,
    buildJobId,
    isBuildRunning,
    startBuild,
    clearBuildLogs,
    stopBuild,
    addBuildLog,
    hfToken,
  } = useStore();

  useWebSocket({ jobId: buildJobId, section: 'build' });

  // Dynamically split config keys for balanced layout
  const { leftKeys, rightKeys } = useMemo(() => {
    const mainKeys = ['data', 'out', 'gpu_ids'];
    return splitConfigKeys(templateConfig, mainKeys);
  }, [templateConfig]);

  // Load template on mount
  useEffect(() => {
    const loadTemplate = async () => {
      try {
        const template = await api.getBuildTemplate();
        setTemplateConfig(template);
        form.setFieldsValue(template);
      } catch (error) {
        message.error('Failed to load template');
      }
    };
    loadTemplate();
  }, [form]);

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      // Validate
      const config = {
        ...values,
        hf_endpoint: hfEndpoint ? hfEndpoint : '',
        ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      };
      const validation = await api.validateBuildConfig(config);
      if (!validation.valid) {
        message.error(`Validation failed: ${validation.errors.join(', ')}`);
        setLoading(false);
        return;
      }

      // Clear logs
      clearBuildLogs();

      // Execute
      const result = await api.executeBuild(config);
      startBuild(result.job_id);
      message.success('Build job started');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to start build');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!buildJobId) {
      return;
    }
    try {
      await api.cancelJob(buildJobId);
      addBuildLog({
        level: 'warning',
        message: 'Cancellation requested',
        timestamp: new Date().toISOString(),
      });
      stopBuild();
      message.info('Cancellation requested');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to cancel build');
    }
  };

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={16}>
        {/* Left Column: Main Configuration + First Half of Parameters */}
        <Col span={12}>
          <Card title="Build Configuration">
            {templateConfig && (
              <>
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
                <HFTokenInput disabled={isBuildRunning} />

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
                  <Input placeholder="data/output.jsonl" />
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
                      loading={loading || isBuildRunning}
                      size="large"
                      style={{ flex: 1 }}
                    >
                      {isBuildRunning ? 'Building...' : 'Execute Build'}
                    </Button>
                    <Button
                      danger
                      onClick={handleStop}
                      disabled={!isBuildRunning}
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
          <ModelDownloadStatus logs={buildLogs} isRunning={isBuildRunning} />
          <Card title="Build Logs" style={{ marginBottom: 16 }}>
            <LogViewer logs={buildLogs} isRunning={isBuildRunning} />
          </Card>

          {templateConfig && rightKeys.length > 0 && (
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
