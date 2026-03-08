/**
 * Type definitions
 */

export interface LogEntry {
  level: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
}

export type UILanguage = 'en' | 'zh';

export interface LocalizedText {
  en: string;
  zh: string;
}

export interface AttackMethodInfo {
  key: string;
  displayName: LocalizedText;
  whatItDoes: LocalizedText;
  whenToUse: LocalizedText;
  riskTradeoff: LocalizedText;
  computeCostHint: LocalizedText;
}

export interface HFDownloadItem {
  path: string;
  size_bytes: number;
  total_bytes?: number | null;
  percent?: number | null;
  model?: string | null;
  mtime?: string | null;
}

export interface HFDownloadStatus {
  cache_dir: string;
  active: boolean;
  downloads: HFDownloadItem[];
  total_downloaded_bytes: number;
  total_expected_bytes?: number | null;
  timestamp: string;
}

export interface GPUMonitorInfo {
  index: number;
  name: string;
  utilization: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature: number;
}

export interface SystemMonitorResponse {
  cpu_percent: number;
  cpu_count: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  gpus: GPUMonitorInfo[];
}

export interface JobResultResponse {
  job_id: string;
  command: string;
  status: string;
  exit_code?: number | null;
  artifacts: Record<string, string | null>;
  result: Record<string, any>;
}

export interface DetectorMetadata {
  key: string;
  name?: string;
  type?: string;
  description?: string | LocalizedText;
  description_en?: string;
  description_zh?: string;
  paper?: string;
  authors?: string;
  venue?: string;
  link?: string;
}

export interface CalibratorThresholdPreset {
  key: string;
  label: string;
  threshold: number;
  source: string;
  tpr?: number;
  fpr?: number;
  target_fpr?: number;
  acc?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
}

export interface CalibratorThresholdsResponse {
  path: string;
  presets: CalibratorThresholdPreset[];
  default_threshold?: number | null;
}

export interface DemoPredictResponse {
  label: 'human' | 'machine';
  confidence: number;
  ai_probability: number;
  threshold: number;
  artifact_paths: Record<string, string | null>;
}

export interface FieldHelpEntry {
  purpose: string;
  higher: string;
  lower: string;
}

export interface Job {
  id: string;
  command: string;
  status: string;
}

export interface BuildConfig {
  data: string;
  out: string;
  [key: string]: any;
}

export interface AttackConfig {
  data: string;
  out: string;
  attacks_config?: any;
  [key: string]: any;
}

export interface TrainConfig {
  detector: string;
  dataset_train: string;
  [key: string]: any;
}

export interface DetectConfig {
  detector: string;
  data: string;
  [key: string]: any;
}

export type Section = 'build' | 'attack' | 'train' | 'detect' | 'demo';
