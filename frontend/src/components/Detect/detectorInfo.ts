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
  venue?: string;
  link?: string;
}

const resolveLocalizedDescription = (row: any, language: 'en' | 'zh'): string | undefined => {
  const preferred = language === 'zh'
    ? [row?.description_zh, row?.description?.zh, row?.description, row?.description_en, row?.description?.en]
    : [row?.description_en, row?.description?.en, row?.description, row?.description_zh, row?.description?.zh];

  for (const candidate of preferred) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim();
    }
  }
  return undefined;
};

export const DETECTOR_INFO: Record<string, DetectorInfo> = {
  binoculars: {
    name: 'Binoculars',
    description: "Compares cross-perplexity between two LLMs (an 'observer' and a 'performer') to zero-shot identify machine-generated passages.",
    venue: 'ICML 2024',
  },
  detectgpt: {
    name: 'DetectGPT',
    description: 'Perturbs input text and measures probability curvature — machine-generated text tends to occupy local maxima in log-probability space.',
    venue: 'ICML 2023 Oral',
  },
  fastdetectgpt: {
    name: 'Fast-DetectGPT',
    description: 'Accelerates DetectGPT by replacing expensive perturbation sampling with conditional probability curvature estimation via a single forward pass.',
    venue: 'ICLR 2024',
  },
  gltr: {
    name: 'GLTR',
    description: 'Visualizes and aggregates per-token rank statistics (top-k bucket counts) from a language model to distinguish human from machine text.',
    venue: 'ACL 2019',
  },
  likelihood: {
    name: 'Likelihood',
    description: 'Computes the average log-probability of each token under a reference language model as a baseline detection signal.',
  },
  rank: {
    name: 'Rank',
    description: 'Uses the average prediction rank of each token as a detection statistic — machine text tends to have lower (better) average ranks.',
  },
  logrank: {
    name: 'LogRank',
    description: 'Applies a logarithmic transform to per-token ranks before averaging, providing a more robust baseline metric than raw rank.',
  },
  entropy: {
    name: 'Entropy',
    description: 'Measures average prediction entropy at each token position — machine-generated text often shows lower entropy patterns.',
  },
  lrr: {
    name: 'LRR',
    description: 'Combines log-likelihood with log-rank information to zero-shot detect machine text, exploiting complementary statistical signals.',
    venue: 'EMNLP 2023 Findings',
  },
  npr: {
    name: 'NPR',
    description: 'Normalizes prediction probability ratios across nested contexts to detect machine text without any training.',
    venue: 'EMNLP 2023 Findings',
  },
  raidar: {
    name: 'RAIDAR',
    description: 'Detects AI-generated text by rewriting the input and comparing semantic similarity — human text changes more substantially when rewritten.',
    venue: 'ICLR 2024',
  },
  tocsin: {
    name: 'TOCSIN',
    description: 'Zero-shot detector that measures token cohesiveness — the consistency of token-level predictions — to identify LLM-generated text.',
    venue: 'EMNLP 2024',
  },
  dnadetectllm: {
    name: 'DNA-DetectLLM',
    description: "Applies a DNA-inspired mutation-repair paradigm: mutates text tokens and observes how well a language model 'repairs' them to distinguish human from AI text.",
    venue: 'NIPS 2025 Spotlight',
  },
  dnagpt: {
    name: 'DNA-GPT',
    description: 'Detects GPT-generated text through divergent N-gram analysis — comparing N-gram divergence patterns between the original and re-generated versions.',
    venue: 'ICLR 2024',
  },
  lastde: {
    name: 'LASTDE',
    description: 'Mines token probability sequences to detect LLM-generated text without any fine-tuning, using statistical patterns in probability distributions.',
    venue: 'ICLR 2025',
  },
  lastdepp: {
    name: 'LASTDE++',
    description: 'An enhanced version of LASTDE with improved probability sequence mining and additional statistical features for stronger detection.',
    venue: 'ICLR 2025',
  },
  pretrained: {
    name: 'Pretrained Generic',
    description: 'Generic loader for any HuggingFace sequence-classification model — specify a model path to use it as a detector.',
  },
  'openai-detector-base': {
    name: 'OpenAI Detector RoBERTa Base',
    description: 'Community-trained RoBERTa-base model for GPT-2 output detection, originally released alongside GPT-2.',
  },
  'openai-detector-large': {
    name: 'OpenAI Detector RoBERTa Large',
    description: 'Community-trained RoBERTa-large model for GPT-2 output detection, a larger variant with higher capacity.',
  },
  'simpleai-detector': {
    name: 'SimpleAI Detector',
    description: 'RoBERTa-based binary classifier from the SimpleAI project for distinguishing human and ChatGPT text.',
  },
  radar: {
    name: 'RADAR',
    description: 'Trains a paraphrase detector jointly with a paraphrase generator to build robustness against common evasion attacks.',
    venue: 'NIPS 2023',
  },
  greater: {
    name: 'GREATER',
    description: "Applies adversarial training to harden machine text detectors against text perturbation attacks — the 'Iron Sharpens Iron' approach.",
    venue: 'ACL 2025',
  },
  detective: {
    name: 'DeTeCtive',
    description: 'Employs multi-level contrastive learning (word, sentence, document) to learn fine-grained representations for AI text detection.',
    venue: 'NIPS 2024',
  },
  detectibe: {
    name: 'DeTeCtive',
    description: 'Employs multi-level contrastive learning (word, sentence, document) to learn fine-grained representations for AI text detection.',
    venue: 'NIPS 2024',
  },
  coco: {
    name: 'CoCo',
    description: 'Enhances detection under low-resource settings by incorporating text coherence signals into a contrastive learning framework.',
    venue: 'EMNLP 2023',
  },
  imbd: {
    name: 'ImBD',
    description: "Addresses class imbalance in detection data by aligning machine stylistic preferences — 'Imitate Before Detect.'",
    venue: 'AAAI 2025 Oral',
  },
  longformer: {
    name: 'Longformer',
    description: 'Uses the Longformer architecture with global attention to classify long documents as human or machine-generated.',
  },
  longerformer: {
    name: 'Longerformer',
    description: 'Extended Longformer variant with expanded context window for very long document classification.',
  },
  mpu: {
    name: 'MPU',
    description: 'Tackles short-text detection through multiscale positive-unlabeled learning, effective even without labeled negative examples.',
    venue: 'ICLR 2024 Spotlight',
  },
  pecola: {
    name: 'PECOLA',
    description: "Bridges selective perturbation with fine-tuned contrastive learning, improving upon DetectGPT's perturbation utilization.",
    venue: 'ACL 2024',
  },
  finetuned: {
    name: 'Finetuned Detector',
    description: 'Loads a locally fine-tuned classification checkpoint as a detector — supports any HuggingFace-compatible model.',
  },
  taste: {
    name: 'TASTE',
    description: 'Enhances cross-lingual robustness of MGT detection via dictionary-driven adversarial training.',
    venue: 'ICLR 2026',
  },
  openaidet: {
    name: 'OpenAI Detector',
    description: 'Community-trained RoBERTa model for GPT-2 output detection, originally released alongside GPT-2.',
  },
  simpleaidet: {
    name: 'SimpleAI Detector',
    description: 'RoBERTa-based binary classifier from the SimpleAI project for distinguishing human and ChatGPT text.',
  },
  seqcls: {
    name: 'Sequence Classification',
    description: 'Generic wrapper for any HuggingFace sequence-classification checkpoint — pass any model path.',
  },
};

export const normalizeDetectorInfoFromApi = (
  rows: any[],
  language: 'en' | 'zh' = 'en',
): Record<string, DetectorInfo> => {
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
    const localizedDescription = resolveLocalizedDescription(row, language);
    map[key] = {
      name: String(row.name || formatDetectorLabel(key)),
      description: localizedDescription || 'No description available.',
      type: row.type ? String(row.type) : undefined,
      paper: row.paper ? String(row.paper) : undefined,
      authors: row.authors ? String(row.authors) : undefined,
      venue: row.venue ? String(row.venue) : undefined,
      link: row.link ? String(row.link) : undefined,
    };
  });
  return map;
};

export const mergeDetectorInfo = (
  apiRows: any[],
  language: 'en' | 'zh' = 'en',
): Record<string, DetectorInfo> => {
  const apiMap = normalizeDetectorInfoFromApi(apiRows, language);
  const merged: Record<string, DetectorInfo> = { ...DETECTOR_INFO };
  Object.entries(apiMap).forEach(([key, value]) => {
    const fallback = merged[key];
    merged[key] = {
      name: value.name || fallback?.name || formatDetectorLabel(key),
      description: value.description || fallback?.description || 'No description available.',
      type: value.type || fallback?.type,
      paper: value.paper || fallback?.paper,
      authors: value.authors || fallback?.authors,
      venue: value.venue || fallback?.venue,
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

export const formatDetectorVenue = (
  detectorKey: string,
  infoMap: Record<string, DetectorInfo> = DETECTOR_INFO,
): string => {
  const key = (detectorKey || '').toLowerCase();
  const venue = infoMap[key]?.venue;
  if (venue && String(venue).trim()) {
    return String(venue).trim();
  }
  return 'N/A';
};

export const hasDetectorVenue = (
  detectorKey: string,
  infoMap: Record<string, DetectorInfo> = DETECTOR_INFO,
): boolean => {
  return formatDetectorVenue(detectorKey, infoMap) !== 'N/A';
};

const getVenueSeries = (venue: string): string => {
  const normalized = String(venue || '').toUpperCase();
  if (normalized.includes('ICML')) return 'ICML';
  if (normalized.includes('ICLR')) return 'ICLR';
  if (normalized.includes('EMNLP')) return 'EMNLP';
  if (normalized.includes('ACL')) return 'ACL';
  if (normalized.includes('NIPS') || normalized.includes('NEURIPS')) return 'NIPS';
  return 'OTHER';
};

export const getDetectorVenueTagColor = (
  detectorKey: string,
  infoMap: Record<string, DetectorInfo> = DETECTOR_INFO,
): string => {
  const venue = formatDetectorVenue(detectorKey, infoMap);
  if (venue === 'N/A') {
    return 'default';
  }
  const series = getVenueSeries(venue);
  if (series === 'ICML') return 'blue';
  if (series === 'ICLR') return 'volcano';
  if (series === 'EMNLP') return 'purple';
  if (series === 'ACL') return 'green';
  if (series === 'NIPS') return 'magenta';
  return 'cyan';
};
