/**
 * API client
 */

import axios from 'axios';

const normalizeApiBase = (raw: string): string => {
  const value = raw.trim();
  if (!value) return '/api';

  if (/^https?:\/\//i.test(value)) {
    const withoutSlash = value.replace(/\/+$/, '');
    return withoutSlash.endsWith('/api') ? withoutSlash : `${withoutSlash}/api`;
  }

  const prefixed = value.startsWith('/') ? value : `/${value}`;
  const withoutSlash = prefixed.replace(/\/+$/, '');
  return withoutSlash.endsWith('/api') ? withoutSlash : `${withoutSlash}/api`;
};

const API_BASE_URL = normalizeApiBase(String(import.meta.env.VITE_PUBLIC_API_BASE_URL || ''));

export const resolveApiUrl = (rawPath: string): string => {
  const value = String(rawPath || '').trim();
  if (!value) return value;
  if (/^https?:\/\//i.test(value)) {
    return value;
  }

  if (/^https?:\/\//i.test(API_BASE_URL)) {
    const apiUrl = new URL(API_BASE_URL);
    if (value.startsWith('/')) {
      return `${apiUrl.protocol}//${apiUrl.host}${value}`;
    }
    return `${apiUrl.protocol}//${apiUrl.host}/${value.replace(/^\/+/, '')}`;
  }

  if (value.startsWith('/')) {
    return `${window.location.origin}${value}`;
  }
  return `${window.location.origin}/${value.replace(/^\/+/, '')}`;
};

export const getWebSocketLogsUrl = (jobId: string): string => {
  const safeJobId = encodeURIComponent(String(jobId || ''));
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    const url = new URL(API_BASE_URL);
    const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    const basePath = url.pathname.replace(/\/+$/, '');
    return `${protocol}//${url.host}${basePath}/ws/logs/${safeJobId}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const basePath = API_BASE_URL.startsWith('/')
    ? API_BASE_URL
    : `/${API_BASE_URL}`;
  const normalized = basePath.replace(/\/+$/, '');
  return `${protocol}//${window.location.host}${normalized}/ws/logs/${safeJobId}`;
};

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Build
  getBuildTemplate: () =>
    apiClient.get('/build/template').then(r => r.data.template),

  validateBuildConfig: (config: any) =>
    apiClient.post('/build/validate', { config }).then(r => r.data),

  executeBuild: (config: any) =>
    apiClient.post('/build/execute', { config }).then(r => r.data),

  // Attack
  getAttackTemplate: () =>
    apiClient.get('/attack/template').then(r => r.data.template),

  getAllAttacks: () =>
    apiClient.get('/attack/all-attacks').then(r => r.data.attacks),

  validateAttackConfig: (config: any) =>
    apiClient.post('/attack/validate', { config }).then(r => r.data),

  executeAttack: (config: any) =>
    apiClient.post('/attack/execute', { config }).then(r => r.data),

  // Train
  getTrainDetectors: () =>
    apiClient.get('/train/detectors').then(r => r.data.detectors),

  getTrainTemplate: (detector: string) =>
    apiClient.get(`/train/template/${detector}`).then(r => r.data.template),

  validateTrainConfig: (config: any) =>
    apiClient.post('/train/validate', { config }).then(r => r.data),

  executeTrain: (config: any) =>
    apiClient.post('/train/execute', { config }).then(r => r.data),

  // Detect
  getDetectDetectors: () =>
    apiClient.get('/detect/detectors').then(r => r.data.detectors),

  getDetectTemplate: (detector: string) =>
    apiClient.get(`/detect/template/${detector}`).then(r => r.data.template),

  validateDetectConfig: (config: any) =>
    apiClient.post('/detect/validate', { config }).then(r => r.data),

  executeDetect: (config: any) =>
    apiClient.post('/detect/execute', { config }).then(r => r.data),

  // Files
  uploadFile: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },

  uploadDataset: (file: File, phase?: 'build' | 'attack' | 'train' | 'detect') => {
    const formData = new FormData();
    formData.append('file', file);
    if (phase) {
      formData.append('phase', phase);
    }
    return apiClient.post('/files/upload-dataset', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },

  downloadFile: (url: string, destination?: string) =>
    apiClient.post('/files/download', { url, destination }).then(r => r.data),

  // System
  getGPUs: () =>
    apiClient.get('/system/gpus').then(r => r.data),

  getLocalModels: (customDirs?: string) =>
    apiClient.get('/system/models', { params: { custom_dirs: customDirs } }).then(r => r.data),

  getCalibrators: (customDirs?: string) =>
    apiClient.get('/system/calibrators', { params: { custom_dirs: customDirs } }).then(r => r.data),

  getCalibratorThresholds: (path: string) =>
    apiClient.get('/system/calibrator-thresholds', { params: { path } }).then(r => r.data),

  cancelJob: (jobId: string) =>
    apiClient.post(`/system/cancel/${jobId}`).then(r => r.data),

  getHealth: () =>
    apiClient.get('/system/health').then(r => r.data),

  getSystemMonitor: () =>
    apiClient.get('/system/monitor').then(r => r.data),

  getHfDownloads: () =>
    apiClient.get('/system/hf-downloads').then(r => r.data),

  getJobResult: (jobId: string) =>
    apiClient.get(`/system/job-result/${jobId}`).then(r => r.data),

  getDetectorMetadata: () =>
    apiClient.get('/system/detector-metadata').then(r => r.data),

  // Demo
  getDemoDetectors: () =>
    apiClient.get('/demo/detectors').then(r => r.data.detectors),

  getDemoTemplate: (detector: string) =>
    apiClient.get(`/demo/template/${detector}`).then(r => r.data.template),

  demoPredict: (payload: any) =>
    apiClient.post('/demo/predict', payload).then(r => r.data),

  demoExecute: (payload: any) =>
    apiClient.post('/demo/execute', payload).then(r => r.data),

  getDemoResult: (jobId: string) =>
    apiClient.get(`/demo/result/${jobId}`).then(r => r.data),
};

export default api;
