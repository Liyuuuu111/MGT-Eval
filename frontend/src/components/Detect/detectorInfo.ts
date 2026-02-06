/**
 * Detector Information and Label Formatting
 * Shared between Train and Detect sections
 */

export interface DetectorInfo {
  name: string;
  description: string;
  type?: string;
  paper?: string;
  authors?: string;
  link?: string;
}

export const DETECTOR_INFO: Record<string, DetectorInfo> = {
  binoculars: {
    name: 'Binoculars',
    description: 'Likelihood ratio between observer and performer models.',
  },
  detectgpt: {
    name: 'DetectGPT',
    description: 'Probability curvature detection using perturbations.',
  },
  fastdetectgpt: {
    name: 'Fast DetectGPT',
    description: 'Optimized DetectGPT with conditional probability curvature.',
  },
  gltr: {
    name: 'GLTR',
    description: 'Rank-based statistics for generated text detection.',
  },
  likelihood: {
    name: 'Likelihood',
    description: 'Log-probability scoring under a language model.',
  },
  rank: {
    name: 'Rank',
    description: 'Token rank statistics.',
  },
  logrank: {
    name: 'LogRank',
    description: 'Log-rank metric.',
  },
  entropy: {
    name: 'Entropy',
    description: 'Entropy-based scoring.',
  },
  lrr: {
    name: 'LRR',
    description: 'Likelihood Ratio with Rank.',
  },
  npr: {
    name: 'NPR',
    description: 'Nested Prediction Ratio.',
  },
  raidar: {
    name: 'RAIDAR',
    description: 'Rewriting-based detection method.',
  },
  tocsin: {
    name: 'TOCSIN',
    description: 'Token-level scoring with a pretrained classifier.',
  },
  dnadetectllm: {
    name: 'DNA DetectLLM',
    description: 'Mining token probability sequences for detection.',
  },
  dnagpt: {
    name: 'DNA GPT',
    description: 'DNA-style token probability sequence detection.',
  },
  lastde: {
    name: 'LASTDE',
    description: 'Training-free detection via token probability sequences.',
  },
  lastdepp: {
    name: 'LASTDE++',
    description: 'Enhanced LASTDE variant.',
  },
  pretrained: {
    name: 'Pretrained Generic',
    description: 'Generic loader for HuggingFace classification models.',
  },
  'openai-detector-base': {
    name: 'OpenAI Detector RoBERTa Base',
    description: 'RoBERTa-base detector trained by OpenAI community.',
  },
  'openai-detector-large': {
    name: 'OpenAI Detector RoBERTa Large',
    description: 'RoBERTa-large detector trained by OpenAI community.',
  },
  'simpleai-detector': {
    name: 'SimpleAI Detector',
    description: 'SimpleAI detector based on RoBERTa.',
  },
  radar: {
    name: 'RADAR',
    description: 'Robust detection with rewriting-aware scoring.',
  },
  greater: {
    name: 'GREATER',
    description: 'Adversarial training for robust MGT detection.',
  },
  detective: {
    name: 'DeTecTive',
    description: 'Multi-level contrastive learning for AI-generated text detection.',
  },
  detectibe: {
    name: 'DeTecTive',
    description: 'Multi-level contrastive learning for AI-generated text detection.',
  },
  coco: {
    name: 'CoCo',
    description: 'Coherence-enhanced contrastive learning under low resource.',
  },
  imbd: {
    name: 'ImBD',
    description: 'Imbalanced data detector with fine-tuned classifier.',
  },
  longformer: {
    name: 'Longformer',
    description: 'Long-document transformer classifier.',
  },
  longerformer: {
    name: 'Longerformer',
    description: 'Long-document transformer variant.',
  },
  mpu: {
    name: 'MPU',
    description: 'Multi-perspective uncertainty-based detector.',
  },
  pecola: {
    name: 'PECOLA',
    description: 'Contrastive learning with selective deletion.',
  },
  finetuned: {
    name: 'Finetuned Detector',
    description: 'Generic finetuned detector from local checkpoints.',
  },
  openaidet: {
    name: 'OpenAI Detector',
    description: 'OpenAI detector based on RoBERTa.',
  },
  simpleaidet: {
    name: 'SimpleAI Detector',
    description: 'SimpleAI detector based on RoBERTa.',
  },
  seqcls: {
    name: 'Sequence Classification',
    description: 'Generic sequence classification detector.',
  },
};

export const normalizeDetectorInfoFromApi = (rows: any[]): Record<string, DetectorInfo> => {
  const map: Record<string, DetectorInfo> = {};
  if (!Array.isArray(rows)) {
    return map;
  }
  rows.forEach((row) => {
    if (!row || typeof row !== 'object') {
      return;
    }
    const key = String(row.key || '').toLowerCase();
    if (!key) {
      return;
    }
    map[key] = {
      name: String(row.name || formatDetectorLabel(key)),
      description: String(row.description || 'No description available.'),
      type: row.type ? String(row.type) : undefined,
      paper: row.paper ? String(row.paper) : undefined,
      authors: row.authors ? String(row.authors) : undefined,
      link: row.link ? String(row.link) : undefined,
    };
  });
  return map;
};

export const mergeDetectorInfo = (apiRows: any[]): Record<string, DetectorInfo> => {
  const apiMap = normalizeDetectorInfoFromApi(apiRows);
  const merged: Record<string, DetectorInfo> = { ...DETECTOR_INFO };
  Object.entries(apiMap).forEach(([key, value]) => {
    const fallback = merged[key];
    merged[key] = {
      name: value.name || fallback?.name || formatDetectorLabel(key),
      description: value.description || fallback?.description || 'No description available.',
      type: value.type || fallback?.type,
      paper: value.paper || fallback?.paper,
      authors: value.authors || fallback?.authors,
      link: value.link || fallback?.link,
    };
  });
  return merged;
};

export const formatDetectorLabel = (
  detectorKey: string,
  infoMap: Record<string, DetectorInfo> = DETECTOR_INFO,
): string => {
  const key = (detectorKey || '').toLowerCase();
  if (infoMap[key]?.name) {
    return infoMap[key].name;
  }
  return detectorKey
    .split(/[_-]/g)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
};
