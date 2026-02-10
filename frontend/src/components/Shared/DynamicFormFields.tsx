/**
 * Dynamic Form Fields Component
 * Automatically generates form fields based on YAML/JSON structure
 */

import React from 'react';
import { Form, Input, InputNumber, Switch, Select, Collapse } from 'antd';
import { ModelSelector } from './ModelSelector';
import { CalibratorSelector } from './CalibratorSelector';
import { FieldHelpText } from './FieldHelpText';
import { getDetectorModelRole, isModelFieldKey } from '../../config/detectorModelRoles';

const { Panel } = Collapse;

const FINETUNED_BACKBONE_PRESETS: string[] = [
  'roberta-base',
  'roberta-large',
  'bert-base-uncased',
  'bert-large-uncased',
  'distilbert-base-uncased',
  'microsoft/deberta-v3-base',
  'microsoft/deberta-v3-large',
  'gpt2',
  'gpt2-medium',
];

interface DynamicFormFieldsProps {
  data: any;
  prefix?: string[];
  excludeKeys?: string[];
  includeKeys?: string[];  // If provided, only render these keys
  modelFields?: string[];
  rootContext?: Record<string, unknown>;
}

export const DynamicFormFields: React.FC<DynamicFormFieldsProps> = ({
  data,
  prefix = [],
  excludeKeys = [],
  includeKeys,
  modelFields = ['model', 'model1', 'model2', 'model3', 'hf_model', 'api_model', 'ppl_model', 'bertscore_model'],
  rootContext,
}) => {
  if (!data || typeof data !== 'object') {
    return null;
  }

  const effectiveRootContext: Record<string, unknown> =
    rootContext ?? (typeof data === 'object' && data !== null ? data : {});
  const detectorHint = typeof effectiveRootContext.detector === 'string' ? effectiveRootContext.detector : '';
  const detectorKwargs = (
    typeof effectiveRootContext.detector_kwargs === 'object' && effectiveRootContext.detector_kwargs !== null
      ? (effectiveRootContext.detector_kwargs as Record<string, unknown>)
      : {}
  );

  const collectCalibratorModelHints = (): string[] => {
    const hints = new Set<string>();
    const add = (v: unknown) => {
      if (typeof v !== 'string') return;
      const text = v.trim();
      if (!text) return;
      hints.add(text);
    };
    const keys = [
      'model',
      'model1',
      'model2',
      'model3',
      'observer_model',
      'performer_model',
      'checkpoint',
      'checkpoint_dir',
      'hf_model',
      'api_model',
      'ppl_model',
      'bertscore_model',
    ];
    keys.forEach((k) => {
      add(effectiveRootContext[k]);
      add(detectorKwargs[k]);
    });
    return Array.from(hints);
  };
  const calibratorModelHints = collectCalibratorModelHints();

  const binaryFieldKeys = new Set([
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
    'bf16',
    'fp16',
    'use_bfloat16',
    'mask_use_bfloat16',
  ]);
  // Keys inside detector_kwargs that duplicate top-level bf16/fp16 flags
  const precisionDupKeys = new Set(['use_bfloat16', 'mask_use_bfloat16']);
  const excludeKeySet = new Set(excludeKeys.map((item) => item.toLowerCase()));

  const getNumberInputSpec = (fieldPath: string, key: string, value: number): {
    step: number;
    precision?: number;
  } => {
    const normalized = `${fieldPath}.${key}`.toLowerCase();

    const integerLikePattern =
      /(token|tokens|steps?|epochs?|batch|num_|_num|k_runs|sample_k|seed|workers?|top_k|num_beams|max_new_tokens|min_new_tokens|max_length|min_length|prefix_k_tokens|write_every|retry|count|n_variants|n_pairs|span_length|chunk_size|max_prompts|gpu_ids|gcn_layers|max_nodes_num|tensor_parallel_size)/;
    const floatLikePattern =
      /(temperature|top_p|learning_rate|(^|_)lr($|_)|dropout|penalty|alpha|beta|gamma|weight_decay|threshold|ratio|pct|prob|confidence)/;

    if (integerLikePattern.test(normalized)) {
      return { step: 1, precision: 0 };
    }

    if (floatLikePattern.test(normalized)) {
      if (/(^|_)lr($|_)|learning_rate/.test(normalized)) {
        return { step: 0.001, precision: 4 };
      }
      return { step: 0.01, precision: 3 };
    }

    if (Number.isInteger(value)) {
      return { step: 1, precision: 0 };
    }

    return { step: 0.01, precision: 3 };
  };

  const renderField = (key: string, value: any, currentPrefix: string[]) => {
    const fieldName = [...currentPrefix, key];
    const fieldPath = fieldName.join('.');
    const keyLower = key.toLowerCase();
    const isBinaryField = binaryFieldKeys.has(keyLower);
    const detectorValue = typeof effectiveRootContext.detector === 'string' ? effectiveRootContext.detector : null;
    const normalizedDetectorValue = String(detectorValue || '').toLowerCase();
    const modelRole = isModelFieldKey(keyLower) ? getDetectorModelRole(detectorValue, keyLower) : undefined;
    const resolvedLabel = modelRole ? `${formatLabel(key)} (${modelRole.label})` : formatLabel(key);
    const fieldHelp = (
      <FieldHelpText
        path={fieldPath}
        value={value}
        context={effectiveRootContext}
      />
    );

    // If includeKeys is provided, only render those keys
    if (includeKeys && !includeKeys.includes(key)) {
      return null;
    }

    // Skip excluded keys
    if (excludeKeys.includes(key) || excludeKeySet.has(keyLower)) {
      return null;
    }

    // Skip detector_kwargs precision keys that duplicate top-level bf16/fp16
    if (precisionDupKeys.has(keyLower) && fieldName.length > 1) {
      const hasToplevelBf16 = 'bf16' in data || 'fp16' in data;
      if (hasToplevelBf16) {
        return null;
      }
    }

    // Handle null values
    if (value === null) {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={resolvedLabel}
          extra={fieldHelp}
        >
          <Input placeholder="null" />
        </Form.Item>
      );
    }

    if (keyLower === 'only_human_prompts') {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={resolvedLabel}
          extra={fieldHelp}
        >
          <Select>
            <Select.Option value={1}>1 (Force Human Prompts)</Select.Option>
            <Select.Option value={0}>0 (Use prompt_from_label)</Select.Option>
          </Select>
        </Form.Item>
      );
    }

    // Render binary (0/1) fields
    if (isBinaryField) {
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={resolvedLabel}
          tooltip="0 = False, 1 = True"
          extra={fieldHelp}
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
          label={resolvedLabel}
          valuePropName="checked"
          extra={fieldHelp}
        >
          <Switch />
        </Form.Item>
      );
    }

    // Handle number
    if (typeof value === 'number') {
      const numberSpec = getNumberInputSpec(fieldPath, keyLower, value);
      return (
        <Form.Item
          key={fieldPath}
          name={fieldName}
          label={resolvedLabel}
          extra={fieldHelp}
        >
          <InputNumber
            style={{ width: '100%' }}
            step={numberSpec.step}
            precision={numberSpec.precision}
          />
        </Form.Item>
      );
    }

    // Handle string - check if it's a model field
    if (typeof value === 'string') {
      // Detector field for HF-style finetuned detector aliases (e.g., hf:roberta-base)
      if (
        keyLower === 'detector'
        && (value.toLowerCase().startsWith('hf:') || normalizedDetectorValue.startsWith('hf:'))
      ) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={resolvedLabel}
            extra={fieldHelp}
            normalize={(inputValue: any) => {
              if (inputValue === null || inputValue === undefined) {
                return inputValue;
              }
              const text = String(inputValue).trim();
              if (!text) {
                return '';
              }
              if (text.toLowerCase().startsWith('hf:')) {
                return text;
              }
              return `hf:${text}`;
            }}
            getValueProps={(inputValue: any) => {
              if (typeof inputValue === 'string' && inputValue.toLowerCase().startsWith('hf:')) {
                return { value: inputValue.slice(3) };
              }
              return { value: inputValue };
            }}
          >
            <ModelSelector
              allowManual
              presetOptions={FINETUNED_BACKBONE_PRESETS}
              presetLabel="Finetuned Backbones"
            />
          </Form.Item>
        );
      }

      // Enumerated field: machine_text_mode
      if (keyLower === 'machine_text_mode') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={resolvedLabel}
            extra={fieldHelp}
          >
            <Select>
              <Select.Option value="prompt_plus">prompt_plus</Select.Option>
              <Select.Option value="completion_only">completion_only</Select.Option>
            </Select>
          </Form.Item>
        );
      }

      if (keyLower === 'calibrator_path' || keyLower === 'calibrator') {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={resolvedLabel}
            extra={fieldHelp}
          >
            <CalibratorSelector
              allowManual
              detectorKey={detectorHint || undefined}
              modelHints={calibratorModelHints}
            />
          </Form.Item>
        );
      }

      // Model fields use ModelSelector
      if (modelFields.includes(key)) {
        const isFinetunedDetector =
          normalizedDetectorValue === 'finetuned'
          || normalizedDetectorValue === 'hfcls'
          || normalizedDetectorValue.startsWith('hf:');
        const finetunedPresetOptions = isFinetunedDetector
          ? FINETUNED_BACKBONE_PRESETS
          : undefined;
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={resolvedLabel}
            extra={fieldHelp}
          >
            <ModelSelector
              allowManual
              presetOptions={finetunedPresetOptions}
              presetLabel="Finetuned Backbones"
            />
          </Form.Item>
        );
      }

      // Device fields
      if (key === 'device' || key.includes('device')) {
        return (
          <Form.Item
            key={fieldPath}
            name={fieldName}
            label={resolvedLabel}
            extra={fieldHelp}
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
            label={resolvedLabel}
            extra={fieldHelp}
          >
            <Select>
              <Select.Option value="auto">auto</Select.Option>
              <Select.Option value="float32">float32</Select.Option>
              <Select.Option value="float16">float16 (FP16)</Select.Option>
              <Select.Option value="bfloat16">bfloat16 (BF16)</Select.Option>
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
            label={resolvedLabel}
            extra={fieldHelp}
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
            label={resolvedLabel}
            extra={fieldHelp}
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
            label={resolvedLabel}
            extra={fieldHelp}
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
          label={resolvedLabel}
          extra={fieldHelp}
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
            label={resolvedLabel}
            tooltip="Enter comma-separated values"
            extra={fieldHelp}
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
                rootContext={effectiveRootContext}
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
      .replace(/([a-zA-Z])(\d+)/g, '$1 $2')
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
