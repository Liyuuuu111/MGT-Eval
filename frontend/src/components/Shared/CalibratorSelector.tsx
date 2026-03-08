/**
 * Calibrator path selector for metric detectors.
 * Supports dropdown search and manual path input.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Form,
  Input,
  Select,
  Space,
  Spin,
  Tag,
  Typography,
} from 'antd';
import {
  FileTextOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import api from '../../services/api';
import { useUILanguage } from '../../hooks/useUILanguage';

interface CalibratorInfo {
  name: string;
  path: string;
  size: string;
}

interface CalibratorSelectorProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  allowManual?: boolean;
  detectorKey?: string;
  modelHints?: string[];
}

export const CalibratorSelector: React.FC<CalibratorSelectorProps> = ({
  value,
  onChange,
  placeholder,
  allowManual = true,
  detectorKey,
  modelHints = [],
}) => {
  const { t } = useUILanguage();
  const form = Form.useFormInstance();
  const liveFormValues = Form.useWatch([], form);
  const [calibrators, setCalibrators] = useState<CalibratorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [manualInput, setManualInput] = useState(false);
  const [customDirs, setCustomDirs] = useState('');
  const prevHintSignatureRef = useRef<string | null>(null);
  const resolvedPlaceholder = placeholder || t('calibratorSelectPlaceholder');

  const isJsonFile = (path: string): boolean => path.toLowerCase().endsWith('.json');
  const isDirectoryPath = (path: string): boolean => path.endsWith('/');

  const normalizeToken = (token: string): string => token.toLowerCase().replace(/[^a-z0-9]+/g, '');

  const runtimeDetectorKey = useMemo(() => {
    if (typeof detectorKey === 'string' && detectorKey.trim()) {
      return detectorKey.trim();
    }
    const det = (liveFormValues as any)?.detector;
    return typeof det === 'string' ? det.trim() : '';
  }, [detectorKey, liveFormValues]);

  const runtimeModelHints = useMemo(() => {
    const hints = new Set<string>();
    const add = (raw: unknown) => {
      if (typeof raw !== 'string') return;
      const text = raw.trim();
      if (!text) return;
      hints.add(text);
    };
    modelHints.forEach(add);

    const walk = (node: unknown, depth = 0) => {
      if (depth > 4 || node === null || node === undefined) return;
      if (Array.isArray(node)) {
        node.forEach((item) => walk(item, depth + 1));
        return;
      }
      if (typeof node !== 'object') return;
      for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
        if (typeof v === 'string' && /(model|checkpoint|tokenizer|observer|performer)/i.test(k)) {
          add(v);
        }
        if (typeof v === 'object' && v !== null) {
          walk(v, depth + 1);
        }
      }
    };
    walk(liveFormValues);

    return Array.from(hints);
  }, [liveFormValues, modelHints]);

  const collectHintTokens = (): string[] => {
    const all = new Set<string>();
    const add = (raw: string) => {
      const text = String(raw || '').trim();
      if (!text) return;
      const basename = text.split('/').pop() || text;
      const chunks = basename.split(/[^a-zA-Z0-9]+/g).filter(Boolean);
      chunks.forEach((c) => {
        const t = normalizeToken(c);
        if (t.length >= 3) {
          all.add(t);
        }
      });
      const merged = normalizeToken(basename);
      if (merged.length >= 4) {
        all.add(merged);
      }
    };
    if (runtimeDetectorKey) {
      add(runtimeDetectorKey);
    }
    runtimeModelHints.forEach(add);
    return Array.from(all);
  };

  const detectorTokens = useMemo(collectHintTokens, [runtimeDetectorKey, runtimeModelHints]);
  const hintSignature = useMemo(
    () => `${runtimeDetectorKey}::${runtimeModelHints.join('|')}`.toLowerCase(),
    [runtimeDetectorKey, runtimeModelHints],
  );

  const pickBestJson = (rows: CalibratorInfo[], currentValue?: string): CalibratorInfo | null => {
    const jsonRows = rows.filter((item) => isJsonFile(item.path));
    if (jsonRows.length === 0) {
      return null;
    }

    const current = String(currentValue || '').trim();
    const currentLower = current.toLowerCase();
    const currentDir = isDirectoryPath(current) ? current : '';

    const score = (item: CalibratorInfo): number => {
      const path = item.path;
      const p = path.toLowerCase();
      const pNorm = normalizeToken(p);
      let s = 0;

      if (currentDir && path.startsWith(currentDir)) s += 300;
      if (currentLower && p === currentLower) s += 500;

      if (/best|latest|final/.test(p)) s += 120;
      if (/calibrator|threshold|decision|boundary/.test(p)) s += 80;
      if (/tpr|fpr|roc|pr[_\-]?curve/.test(p)) s += 60;
      if (/dev|valid|validation/.test(p)) s += 25;
      if (/manifest|summary|metrics|prediction|predictions|log|asr/.test(p)) s -= 180;

      if (detectorTokens.length > 0) {
        for (const token of detectorTokens) {
          if (token && pNorm.includes(token)) {
            s += 35;
          }
        }
      }

      const depth = path.split('/').length;
      s -= depth * 2;
      s -= path.length * 0.03;
      return s;
    };

    const ranked = [...jsonRows].sort((a, b) => {
      const diff = score(b) - score(a);
      if (diff !== 0) {
        return diff;
      }
      return a.path.localeCompare(b.path);
    });
    return ranked[0] || null;
  };

  const resolveToJsonPath = (raw: string, rows: CalibratorInfo[]): string => {
    const text = String(raw || '').trim();
    if (!text) return text;
    if (isJsonFile(text)) return text;
    const dir = text.endsWith('/') ? text : `${text}/`;
    const jsonInDir = rows.filter((r) => isJsonFile(r.path) && r.path.startsWith(dir));
    const best = pickBestJson(jsonInDir, text);
    return best?.path || text;
  };

  const loadCalibrators = async (
    dirs?: string,
    options?: { forceRepick?: boolean },
  ) => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.getCalibrators(dirs);
      const rawRows: any[] = Array.isArray(result?.calibrators) ? result.calibrators : [];
      let rows: CalibratorInfo[] = rawRows.filter((item: any): item is CalibratorInfo => (
        item
        && typeof item.path === 'string'
        && typeof item.name === 'string'
        && typeof item.size === 'string'
      ));

      // Sort: JSON files first, then directories
      // Within each group, sort by path
      rows = rows.sort((a, b) => {
        const aIsDir = a.path.endsWith('/');
        const bIsDir = b.path.endsWith('/');
        const aIsJson = a.path.toLowerCase().endsWith('.json');
        const bIsJson = b.path.toLowerCase().endsWith('.json');

        // JSON files first
        if (aIsJson && !bIsJson) return -1;
        if (!aIsJson && bIsJson) return 1;

        // Then non-JSON files (but not directories)
        if (!aIsDir && bIsDir) return -1;
        if (aIsDir && !bIsDir) return 1;

        // Within same type, sort alphabetically
        return a.path.localeCompare(b.path);
      });

      setCalibrators(rows);

      // Auto-pick a suitable JSON calibrator when value is empty, invalid, or a directory path.
      if (onChange && rows.length > 0) {
        const forceRepick = !!options?.forceRepick;
        const bestJson = pickBestJson(rows, forceRepick ? undefined : value);
        const hasCurrent = !!value && rows.some((item) => item.path === value);
        const currentIsJson = !!value && isJsonFile(value);
        const shouldAutofill =
          forceRepick ||
          !value ||
          !hasCurrent ||
          !currentIsJson ||
          isDirectoryPath(String(value));
        if (shouldAutofill && bestJson && bestJson.path !== value) {
          onChange(bestJson.path);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || t('calibratorDetectionFailed'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (manualInput) {
      return;
    }
    const forceRepick =
      prevHintSignatureRef.current !== null &&
      prevHintSignatureRef.current !== hintSignature;
    prevHintSignatureRef.current = hintSignature;
    loadCalibrators(undefined, { forceRepick });
  }, [hintSignature, manualInput]);

  if (loading) {
    return <Spin tip={t('calibratorScanning')} />;
  }

  if (error) {
    return (
      <Alert
        type="warning"
        message={t('calibratorDetectionFailed')}
        description={
          <div>
            {error}
            <br />
            <a onClick={() => loadCalibrators(undefined, { forceRepick: false })} style={{ cursor: 'pointer' }}>
              <ReloadOutlined /> {t('calibratorRetry')}
            </a>
          </div>
        }
        showIcon
      />
    );
  }

  if (manualInput) {
    return (
      <div>
        <Input
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={resolvedPlaceholder}
          prefix={<FileTextOutlined />}
        />
        <div style={{ marginTop: 6 }}>
          <a onClick={() => setManualInput(false)} style={{ fontSize: 12 }}>
            {t('calibratorBack')}
          </a>
        </div>
      </div>
    );
  }

  return (
    <div>
      <Select
        showSearch
        value={value}
        onChange={(nextValue: string) => {
          if (!onChange) {
            return;
          }
          onChange(resolveToJsonPath(nextValue, calibrators));
        }}
        placeholder={resolvedPlaceholder}
        style={{ width: '100%' }}
        optionFilterProp="label"
        filterOption={(input, option) =>
          String(option?.label || '').toLowerCase().includes(input.toLowerCase())
        }
      >
        {calibrators.map((item) => {
          const isDir = item.path.endsWith('/');
          return (
            <Select.Option key={item.path} value={item.path} label={`${item.path} ${item.size}`}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={item.path}>
                  {isDir ? <FolderOpenOutlined /> : <FileTextOutlined />} {item.path}
                </div>
                <Tag color={isDir ? 'gold' : 'blue'} style={{ margin: 0 }}>
                  {item.size}
                </Tag>
              </div>
            </Select.Option>
          );
        })}
      </Select>

      <Space style={{ marginTop: 8, width: '100%' }}>
        <Input
          value={customDirs}
          onChange={(e) => setCustomDirs(e.target.value)}
          placeholder={t('calibratorAdditionalDirs')}
        />
        <Button
          icon={<SearchOutlined />}
          onClick={() => loadCalibrators(customDirs.trim() || undefined, { forceRepick: false })}
        >
          {t('calibratorSearch')}
        </Button>
        <Button
          icon={<ReloadOutlined />}
          onClick={() => {
            setCustomDirs('');
            loadCalibrators(undefined, { forceRepick: false });
          }}
        >
          {t('calibratorReset')}
        </Button>
      </Space>

      <div style={{ marginTop: 6 }}>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          {t('calibratorAutoScanHint')}
        </Typography.Text>
        {allowManual && (
          <>
            {' '}
            <a onClick={() => setManualInput(true)} style={{ fontSize: 12 }}>
              {t('calibratorManualEntry')}
            </a>
          </>
        )}
      </div>
    </div>
  );
};
