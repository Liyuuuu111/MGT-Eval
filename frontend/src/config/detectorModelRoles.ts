export type ModelFieldKey = 'model1' | 'model2' | 'model3';

export interface DetectorModelRole {
  label: string;
  purpose: string;
}

type DetectorModelRoleMap = Record<string, Partial<Record<ModelFieldKey, DetectorModelRole>>>;

const SCORE_MODEL_ROLE: DetectorModelRole = {
  label: 'Score Model',
  purpose: 'Primary scoring language model used by this detector to compute token/probability-based signals.',
};

const CLASSIFIER_MODEL_ROLE: DetectorModelRole = {
  label: 'Classifier Checkpoint',
  purpose: 'Fine-tuned detector/classifier checkpoint used directly for prediction.',
};

const DETECTOR_MODEL_ROLES: DetectorModelRoleMap = {
  binoculars: {
    model1: {
      label: 'Observer Model',
      purpose: 'Observer model used in the Binoculars ratio calculation.',
    },
    model2: {
      label: 'Performer Model',
      purpose: 'Performer model paired with observer model for Binoculars comparison.',
    },
  },
  detectgpt: {
    model1: {
      label: 'Score Model',
      purpose: 'Model used to score original and perturbed texts in DetectGPT.',
    },
    model2: {
      label: 'Mask Model',
      purpose: 'Mask/fill model used to generate perturbations for DetectGPT.',
    },
  },
  fastdetectgpt: {
    model1: {
      label: 'Scoring Model',
      purpose: 'Model used for scoring in FastDetectGPT.',
    },
    model2: {
      label: 'Sampling Model',
      purpose: 'Model used to sample/perturb candidates in FastDetectGPT.',
    },
  },
  npr: {
    model1: {
      label: 'Score Model',
      purpose: 'Primary scoring model in NPR.',
    },
    model2: {
      label: 'Mask Model',
      purpose: 'Mask model used to build perturbations in NPR.',
    },
  },
  dnadetectllm: {
    model1: {
      label: 'Observer Model',
      purpose: 'Observer model in DNA-DetectLLM.',
    },
    model2: {
      label: 'Performer Model',
      purpose: 'Performer model in DNA-DetectLLM.',
    },
  },
  lastdepp: {
    model1: {
      label: 'Score Model',
      purpose: 'Score model used for LASTDE++ signal extraction.',
    },
    model2: {
      label: 'Reference Model',
      purpose: 'Reference model paired with score model in LASTDE++.',
    },
  },
  tocsin: {
    model1: {
      label: 'Score Model',
      purpose: 'Score model used in TOCSIN.',
    },
    model2: {
      label: 'Reference Model',
      purpose: 'Reference model used in TOCSIN for comparative features.',
    },
  },
  greater: {
    model1: {
      label: 'Target Detector',
      purpose: 'Target detector model used for GREATER training/inference.',
    },
    model2: {
      label: 'Surrogate Detector',
      purpose: 'Surrogate detector model used for adversarial/augmentation steps in GREATER.',
    },
    model3: {
      label: 'MLM Model',
      purpose: 'Masked language model used for token replacement/augmentation in GREATER.',
    },
  },
  raidar: {
    model1: {
      label: 'Rewrite Model',
      purpose: 'Rewrite model used to generate rewritten variants in RAIDAR.',
    },
  },
  detective: {
    model1: {
      label: 'Embedding Model',
      purpose: 'Sentence embedding model used by DeTecTive.',
    },
  },
  coco: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  mpu: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  pecola: {
    model1: {
      label: 'Backbone Model',
      purpose: 'Backbone classifier model used in PECOLA.',
    },
    model2: {
      label: 'Augmentation Model',
      purpose: 'Auxiliary generation/augmentation model used by PECOLA.',
    },
  },
  longformer: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  hfcls: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  pretrained: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  openaidet: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  simpleaidet: {
    model1: CLASSIFIER_MODEL_ROLE,
  },
  likelihood: {
    model1: SCORE_MODEL_ROLE,
  },
  rank: {
    model1: SCORE_MODEL_ROLE,
  },
  logrank: {
    model1: SCORE_MODEL_ROLE,
  },
  entropy: {
    model1: SCORE_MODEL_ROLE,
  },
  lrr: {
    model1: SCORE_MODEL_ROLE,
  },
  gltr: {
    model1: SCORE_MODEL_ROLE,
  },
  dnagpt: {
    model1: SCORE_MODEL_ROLE,
  },
  lastde: {
    model1: SCORE_MODEL_ROLE,
  },
};

const MODEL_FIELD_KEYS = new Set<ModelFieldKey>(['model1', 'model2', 'model3']);

export const isModelFieldKey = (key: string): key is ModelFieldKey => {
  return MODEL_FIELD_KEYS.has((key || '').toLowerCase() as ModelFieldKey);
};

export const normalizeDetectorKey = (detector: string | null | undefined): string => {
  const raw = String(detector || '').trim().toLowerCase();
  if (!raw) return '';
  if (raw.startsWith('hf:')) return 'hfcls';
  if (raw === 'finetuned') return 'hfcls';
  return raw;
};

export const getDetectorModelRole = (
  detector: string | null | undefined,
  key: ModelFieldKey,
): DetectorModelRole | undefined => {
  const normalized = normalizeDetectorKey(detector);
  if (!normalized) return undefined;
  return DETECTOR_MODEL_ROLES[normalized]?.[key];
};
