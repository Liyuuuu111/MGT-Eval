/**
 * Dynamic Form Fields Component
 * Automatically generates form fields based on YAML/JSON structure
 */

import React from 'react';
import { Form, Input, InputNumber, Switch, Select, Collapse } from 'antd';
import { ModelSelector } from './ModelSelector';
import { CalibratorSelector } from './CalibratorSelector';

const { Panel } = Collapse;

interface DynamicFormFieldsProps {
  data: any;
  prefix?: string[];
  excludeKeys?: string[];
  includeKeys?: string[];  // If provided, only render these keys
  modelFields?: string[];
}

export const DynamicFormFields: React.FC<DynamicFormFieldsProps> = ({
  data,
  prefix = [],
  excludeKeys = [],
  includeKeys,
  modelFields = ['model', 'model1', 'model2', 'model3', 'hf_model', 'api_model', 'ppl_model', 'bertscore_model']
}) => {
  if (!data || typeof data !== 'object') {
    return null;
  }

  const binaryFieldKeys = new Set([
    'prompt_from_label',
    'only_human_prompts',
    'do_sample',
    'return_full_text',
    'vllm_enforce_eager',
    'vllm_trust_remote_code',
    'vllm_disable_log_stats',
    'metric_ppl',
    'metric_readability',
    'metric_bertscore',
    'only_attack_machine',
  ]);

  const renderField = (key: string, value: any, currentPrefix: string[]) => {
    const fieldName = [...currentPrefix, key];
    const fieldPath = fieldName.join('.');
    const keyLower = key.toLowerCase();
    const isBinaryField = binaryFieldKeys.has(keyLower);

    // If includeKeys is provided, only render those keys
    if (includeKeys && !includeKeys.includes(key)) {
      return null;
    }

    // Skip excluded keys
    if (excludeKeys.includes(key)) {
      return null;
    }

    // Handle null values
    if (value === null) {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={formatLabel(key)}
        >
          <Input placeholder="null" />
        </Form.Item>
      );
    }

    // Render binary (0/1) fields
    if (isBinaryField) {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={formatLabel(key)}
          tooltip="0 = False, 1 = True"
          normalize={(inputValue: any) => {
            if (inputValue === null || inputValue === undefined || inputValue === '') {
              return inputValue;
            }
            if (inputValue === true) return 1;
            if (inputValue === false) return 0;
            return Number(inputValue);
          }}
          getValueProps={(inputValue: any) => ({
            value: typeof inputValue === 'boolean' ? (inputValue ? 1 : 0) : inputValue,
          })}
        >
          <InputNumber min={0} max={1} step={1} style={{ width: '100%' }} />
        </Form.Item>
      );
    }

    // Handle boolean
    if (typeof value === 'boolean') {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={formatLabel(key)}
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>
      );
    }

    // Handle number
    if (typeof value === 'number') {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={formatLabel(key)}
        >
          <InputNumber style={{ width: '100%' }} step={value < 1 ? 0.01 : 1} />
        </Form.Item>
      );
    }

    // Handle string - check if it's a model field
    if (typeof value === 'string') {
      if (keyLower === 'calibrator_path' || keyLower === 'calibrator') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <CalibratorSelector allowManual />
          </Form.Item>
        );
      }

      // Model fields use ModelSelector
      if (modelFields.includes(key)) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <ModelSelector allowManual />
          </Form.Item>
        );
      }

      // Device fields
      if (key === 'device' || key.includes('device')) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <Select>
              <Select.Option value="cuda">cuda</Select.Option>
              <Select.Option value="cuda:0">cuda:0</Select.Option>
              <Select.Option value="cuda:1">cuda:1</Select.Option>
              <Select.Option value="cpu">cpu</Select.Option>
              <Select.Option value="auto">auto</Select.Option>
            </Select>
          </Form.Item>
        );
      }

      // Dtype fields
      if (key === 'dtype' || key.includes('dtype')) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <Select>
              <Select.Option value="auto">auto</Select.Option>
              <Select.Option value="float32">float32</Select.Option>
              <Select.Option value="float16">float16</Select.Option>
              <Select.Option value="bfloat16">bfloat16</Select.Option>
              <Select.Option value="bf16">bf16</Select.Option>
            </Select>
          </Form.Item>
        );
      }

      // Backend fields
      if (key === 'backend') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <Select>
              <Select.Option value="hf">hf</Select.Option>
              <Select.Option value="api">api</Select.Option>
              <Select.Option value="chatgpt">chatgpt</Select.Option>
              <Select.Option value="pegasus">pegasus</Select.Option>
              <Select.Option value="dipper">dipper</Select.Option>
              <Select.Option value="modelfree">modelfree</Select.Option>
              <Select.Option value="modelbase">modelbase</Select.Option>
            </Select>
          </Form.Item>
        );
      }

      // Mode fields
      if (key === 'mode') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <Input />
          </Form.Item>
        );
      }

      // Long text fields (prompts, etc)
      if (key.includes('prompt') || key.includes('system') || value.length > 100) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
          >
            <Input.TextArea rows={4} />
          </Form.Item>
        );
      }

      // Regular string input
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={formatLabel(key)}
        >
          <Input />
        </Form.Item>
      );
    }

    // Handle array
    if (Array.isArray(value)) {
      // Simple array of primitives
      if (value.length === 0 || typeof value[0] !== 'object') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={formatLabel(key)}
            tooltip="Enter comma-separated values"
          >
            <Input placeholder="value1, value2, value3" />
          </Form.Item>
        );
      }
      // Array of objects - render as nested forms
      return null; // Skip complex arrays for now
    }

    // Handle nested object
    if (typeof value === 'object' && value !== null) {
      return (
        <div key={fieldPath} style={{ marginLeft: 0 }}>
          <Collapse
            defaultActiveKey={[]}
            ghost
            style={{ marginBottom: 16 }}
          >
            <Panel header={formatLabel(key)} key={fieldPath}>
              <DynamicFormFields
                data={value}
                prefix={fieldName}
                excludeKeys={excludeKeys}
                modelFields={modelFields}
              />
            </Panel>
          </Collapse>
        </div>
      );
    }

    return null;
  };

  const formatLabel = (key: string): string => {
    // Convert snake_case to Title Case
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <>
      {Object.entries(data).map(([key, value]) =>
        renderField(key, value, prefix)
      )}
    </>
  );
};
