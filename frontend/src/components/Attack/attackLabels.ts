import { AttackMethodInfo, UILanguage } from '../../types';

const DEFAULT_METHOD_INFO: AttackMethodInfo = {
  key: 'unknown',
  displayName: { en: 'Unknown Attack', zh: '未知攻击' },
  whatItDoes: {
    en: 'Applies a transformation to the base text to create an attacked variant.',
    zh: '对原始文本执行变换，生成攻击后的样本。',
  },
  whenToUse: {
    en: 'Use when evaluating robustness against generic perturbations.',
    zh: '当你需要评估检测器对通用扰动的鲁棒性时使用。',
  },
  riskTradeoff: {
    en: 'May introduce unnatural artifacts if parameters are too aggressive.',
    zh: '参数过于激进时可能引入不自然伪迹。',
  },
  computeCostHint: {
    en: 'Usually low-to-medium, depending on backend/model usage.',
    zh: '通常为低到中等开销，取决于是否依赖模型后端。',
  },
};

export const ATTACK_METHOD_INFO: Record<string, AttackMethodInfo> = {
  span: {
    key: 'span',
    displayName: { en: 'Span Perturbation', zh: '片段扰动' },
    whatItDoes: {
      en: 'Masks spans in text and infills them with a seq2seq model to create semantically similar but rephrased content.',
      zh: '对文本片段进行掩码并由序列到序列模型补全，得到语义相近但表达变化的文本。',
    },
    whenToUse: {
      en: 'Useful for stress-testing detectors against model-assisted rewriting with moderate semantic preservation.',
      zh: '适合评估检测器面对“模型辅助改写且语义基本保留”场景的鲁棒性。',
    },
    riskTradeoff: {
      en: 'High masking ratios can break coherence and create noisy text.',
      zh: '掩码比例过高会破坏连贯性并产生噪声文本。',
    },
    computeCostHint: {
      en: 'Medium to high (requires running a mask-filling model).',
      zh: '中到高（需要运行补全文本模型）。',
    },
  },
  para: {
    key: 'para',
    displayName: { en: 'Paraphrasing', zh: '释义改写' },
    whatItDoes: {
      en: 'Rewrites sentences while preserving the original meaning using model-based or API-based paraphrasers.',
      zh: '使用模型或 API 对文本进行释义改写，在尽量保持原意的前提下改变表达形式。',
    },
    whenToUse: {
      en: 'Best for measuring detector robustness against fluent semantic-preserving rewrites.',
      zh: '适合测试检测器对“流畅且语义保持”的改写攻击的鲁棒性。',
    },
    riskTradeoff: {
      en: 'Can drift in facts/style under aggressive decoding settings.',
      zh: '在激进采样参数下可能出现事实漂移或风格偏移。',
    },
    computeCostHint: {
      en: 'Medium to high, especially with large local or remote models.',
      zh: '中到高，尤其在使用大模型本地推理或远程 API 时。',
    },
  },
  typo: {
    key: 'typo',
    displayName: { en: 'Typo Mixed', zh: '综合拼写扰动' },
    whatItDoes: {
      en: 'Applies mixed character-level typo operations (insert/delete/substitute/transpose).',
      zh: '混合执行字符级拼写扰动（插入/删除/替换/换位）。',
    },
    whenToUse: {
      en: 'Use for robustness checks under realistic noisy user-input conditions.',
      zh: '用于评估检测器在真实用户输入噪声场景下的鲁棒性。',
    },
    riskTradeoff: {
      en: 'Too much noise can make text unreadable and less realistic.',
      zh: '噪声过大可能导致文本不可读、脱离真实场景。',
    },
    computeCostHint: {
      en: 'Low (string operations only).',
      zh: '低（仅字符串操作）。',
    },
  },
  inse: {
    key: 'inse',
    displayName: { en: 'Typo Insertion', zh: '拼写插入扰动' },
    whatItDoes: {
      en: 'Inserts extra characters into words to simulate keyboard/input mistakes.',
      zh: '在词中插入额外字符，模拟输入错误。',
    },
    whenToUse: {
      en: 'Good for evaluating tolerance to mild OCR/input corruption.',
      zh: '适合评估检测器对轻度输入损坏的容忍度。',
    },
    riskTradeoff: {
      en: 'Excessive insertion quickly harms readability.',
      zh: '插入过多会快速降低可读性。',
    },
    computeCostHint: {
      en: 'Low.',
      zh: '低。',
    },
  },
  dele: {
    key: 'dele',
    displayName: { en: 'Typo Deletion', zh: '拼写删除扰动' },
    whatItDoes: {
      en: 'Deletes characters from words to emulate dropped keystrokes.',
      zh: '删除词内字符，模拟漏打字符。',
    },
    whenToUse: {
      en: 'Useful for robustness tests against missing-character noise.',
      zh: '用于测试检测器对缺失字符噪声的鲁棒性。',
    },
    riskTradeoff: {
      en: 'Can severely damage short words and sentence clarity.',
      zh: '对短词影响尤其明显，可能破坏句子清晰度。',
    },
    computeCostHint: {
      en: 'Low.',
      zh: '低。',
    },
  },
  subs: {
    key: 'subs',
    displayName: { en: 'Typo Substitution', zh: '拼写替换扰动' },
    whatItDoes: {
      en: 'Substitutes characters with nearby/alternative characters to mimic human typos.',
      zh: '将字符替换为近邻字符或近形字符，模拟人工拼写错误。',
    },
    whenToUse: {
      en: 'Suitable for testing sensitivity to local spelling perturbations.',
      zh: '适合评估检测器对局部拼写扰动的敏感性。',
    },
    riskTradeoff: {
      en: 'May create non-words and reduce realism if too strong.',
      zh: '强度过高会产生大量无效词，降低真实性。',
    },
    computeCostHint: {
      en: 'Low.',
      zh: '低。',
    },
  },
  tran: {
    key: 'tran',
    displayName: { en: 'Typo Transposition', zh: '拼写换位扰动' },
    whatItDoes: {
      en: 'Swaps adjacent characters to reproduce common transposition typos.',
      zh: '交换相邻字符，模拟常见的换位拼写错误。',
    },
    whenToUse: {
      en: 'Useful for testing resilience to subtle character-order noise.',
      zh: '适合测试检测器对字符顺序微扰的鲁棒性。',
    },
    riskTradeoff: {
      en: 'Aggressive transposition can reduce lexical validity.',
      zh: '换位过强会降低词汇合法性。',
    },
    computeCostHint: {
      en: 'Low.',
      zh: '低。',
    },
  },
  homo: {
    key: 'homo',
    displayName: { en: 'Homoglyph Alteration', zh: '同形字符攻击' },
    whatItDoes: {
      en: 'Replaces characters with visually similar homoglyphs to preserve appearance while altering tokens.',
      zh: '用视觉相近字符替换原字符，在“看起来相似”的同时改变 token 表示。',
    },
    whenToUse: {
      en: 'Great for testing robustness against visual spoofing and Unicode attacks.',
      zh: '适合评估检测器抵御视觉欺骗和 Unicode 攻击的能力。',
    },
    riskTradeoff: {
      en: 'Can introduce encoding/display issues in downstream tools.',
      zh: '可能在下游工具中触发编码或显示兼容问题。',
    },
    computeCostHint: {
      en: 'Low to medium (depends on mapping strategy).',
      zh: '低到中（取决于映射策略）。',
    },
  },
  form: {
    key: 'form',
    displayName: { en: 'Format Character Editing', zh: '格式字符编辑' },
    whatItDoes: {
      en: 'Perturbs spacing/punctuation/case-like formatting cues without major semantic edits.',
      zh: '在不显著改变语义的前提下扰动空格、标点、大小写等格式特征。',
    },
    whenToUse: {
      en: 'Useful for probing detector dependence on superficial formatting signals.',
      zh: '用于分析检测器是否过度依赖表层格式特征。',
    },
    riskTradeoff: {
      en: 'Can overfit to synthetic formatting patterns if overused.',
      zh: '过度使用会产生不自然格式模式，影响评测真实性。',
    },
    computeCostHint: {
      en: 'Low.',
      zh: '低。',
    },
  },
  syno: {
    key: 'syno',
    displayName: { en: 'Synonym Substitution', zh: '同义词替换' },
    whatItDoes: {
      en: 'Replaces words/phrases with synonyms via dictionary or model-based candidate generation.',
      zh: '通过词典或模型候选生成进行词级同义替换。',
    },
    whenToUse: {
      en: 'Good for lexical-level robustness evaluation with semantic preservation targets.',
      zh: '适合做词汇层面的鲁棒性评估，并尽量保持语义不变。',
    },
    riskTradeoff: {
      en: 'Context mismatch may hurt fluency or factual precision.',
      zh: '上下文不匹配时会影响流畅性和事实准确性。',
    },
    computeCostHint: {
      en: 'Low to high depending on dictionary vs model-based backend.',
      zh: '取决于后端：词典方式低，模型方式中高。',
    },
  },
  back_trans: {
    key: 'back_trans',
    displayName: { en: 'Back Translation', zh: '回译攻击' },
    whatItDoes: {
      en: 'Translates text to an intermediate language and back, producing paraphrastic variation.',
      zh: '将文本翻译到中间语言再翻译回来，得到改写变体。',
    },
    whenToUse: {
      en: 'Effective for generating natural paraphrases while changing lexical/syntactic surface forms.',
      zh: '适合生成较自然的改写文本，同时改变词汇和句法表面形式。',
    },
    riskTradeoff: {
      en: 'May alter named entities/terminology and introduce translation artifacts.',
      zh: '可能改变专有名词或术语，并引入翻译伪迹。',
    },
    computeCostHint: {
      en: 'Medium to high (two translation passes).',
      zh: '中到高（需要双向翻译）。',
    },
  },
  humanize: {
    key: 'humanize',
    displayName: { en: 'Humanize Rewrite', zh: '人性化改写' },
    whatItDoes: {
      en: 'Rewrites machine-like text toward more human style, rhythm, and expression.',
      zh: '将偏机器风格文本改写为更接近人类表达的风格与节奏。',
    },
    whenToUse: {
      en: 'Use for adversarial robustness testing against style-humanization attacks.',
      zh: '用于测试检测器在“人性化改写”对抗场景下的鲁棒性。',
    },
    riskTradeoff: {
      en: 'Can inject stylistic bias and alter factual framing if prompts are too strong.',
      zh: '提示词过强时可能引入风格偏差或改变事实表达框架。',
    },
    computeCostHint: {
      en: 'Medium to high (typically model/API based).',
      zh: '中到高（通常依赖模型/API）。',
    },
  },
};

export interface AttackMethodMetrics {
  scores: {
    computeCost: number;
    semanticPreservation: number;
    fluency: number;
    stealth: number;
    attackPower: number;
  };
  costLevel: number;
  costTier: 'low' | 'medium' | 'high';
  methodType: 'rule' | 'model' | 'api' | 'hybrid';
}

export const ATTACK_METHOD_METRICS: Record<string, AttackMethodMetrics> = {
  typo: {
    scores: { computeCost: 2, semanticPreservation: 8, fluency: 6, stealth: 6, attackPower: 5 },
    costLevel: 2,
    costTier: 'low',
    methodType: 'rule',
  },
  inse: {
    scores: { computeCost: 1, semanticPreservation: 8, fluency: 6, stealth: 5, attackPower: 4 },
    costLevel: 1,
    costTier: 'low',
    methodType: 'rule',
  },
  dele: {
    scores: { computeCost: 1, semanticPreservation: 7, fluency: 5, stealth: 5, attackPower: 4 },
    costLevel: 1,
    costTier: 'low',
    methodType: 'rule',
  },
  subs: {
    scores: { computeCost: 1, semanticPreservation: 8, fluency: 6, stealth: 6, attackPower: 4 },
    costLevel: 1,
    costTier: 'low',
    methodType: 'rule',
  },
  tran: {
    scores: { computeCost: 1, semanticPreservation: 8, fluency: 6, stealth: 5, attackPower: 4 },
    costLevel: 1,
    costTier: 'low',
    methodType: 'rule',
  },
  homo: {
    scores: { computeCost: 2, semanticPreservation: 9, fluency: 9, stealth: 9, attackPower: 3 },
    costLevel: 2,
    costTier: 'low',
    methodType: 'rule',
  },
  form: {
    scores: { computeCost: 1, semanticPreservation: 10, fluency: 10, stealth: 10, attackPower: 2 },
    costLevel: 1,
    costTier: 'low',
    methodType: 'rule',
  },
  syno: {
    scores: { computeCost: 4, semanticPreservation: 8, fluency: 8, stealth: 7, attackPower: 6 },
    costLevel: 4,
    costTier: 'medium',
    methodType: 'hybrid',
  },
  span: {
    scores: { computeCost: 7, semanticPreservation: 7, fluency: 8, stealth: 6, attackPower: 7 },
    costLevel: 7,
    costTier: 'high',
    methodType: 'model',
  },
  para: {
    scores: { computeCost: 8, semanticPreservation: 8, fluency: 9, stealth: 8, attackPower: 8 },
    costLevel: 8,
    costTier: 'high',
    methodType: 'model',
  },
  back_trans: {
    scores: { computeCost: 7, semanticPreservation: 7, fluency: 7, stealth: 7, attackPower: 6 },
    costLevel: 7,
    costTier: 'high',
    methodType: 'model',
  },
  humanize: {
    scores: { computeCost: 9, semanticPreservation: 8, fluency: 10, stealth: 9, attackPower: 9 },
    costLevel: 9,
    costTier: 'high',
    methodType: 'api',
  },
};

// Optimized color palette with high contrast for better visual distinction
export const ATTACK_COLOR_MAP: Record<string, string> = {
  typo: '#1890ff',      // Bright Blue
  inse: '#52c41a',      // Bright Green
  dele: '#ff4d4f',      // Bright Red
  subs: '#faad14',      // Bright Orange
  tran: '#722ed1',      // Purple
  homo: '#eb2f96',      // Magenta
  form: '#13c2c2',      // Cyan
  syno: '#fa8c16',      // Dark Orange
  span: '#a0d911',      // Lime Green
  para: '#f5222d',      // Deep Red
  back_trans: '#2f54eb', // Indigo
  humanize: '#fadb14',  // Yellow
};

const titleCase = (value: string): string =>
  value
    .split(/[_-]/g)
    .filter(Boolean)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(' ');

export const getAttackMethodInfo = (attackType?: string): AttackMethodInfo => {
  const key = (attackType || '').toLowerCase();
  return ATTACK_METHOD_INFO[key] || {
    ...DEFAULT_METHOD_INFO,
    key: key || DEFAULT_METHOD_INFO.key,
    displayName: {
      en: attackType ? titleCase(attackType) : DEFAULT_METHOD_INFO.displayName.en,
      zh: attackType ? titleCase(attackType) : DEFAULT_METHOD_INFO.displayName.zh,
    },
  };
};

export const formatAttackLabel = (
  attackType?: string,
  backend?: string,
  language: UILanguage = 'en',
): string => {
  const info = getAttackMethodInfo(attackType);
  const base = info.displayName[language] || info.displayName.en;
  if (!backend) {
    return base;
  }
  return `${base} (${backend})`;
};

export const getAttackMetrics = (attackType?: string): AttackMethodMetrics => {
  const key = (attackType || '').toLowerCase();
  return ATTACK_METHOD_METRICS[key] ?? ATTACK_METHOD_METRICS.typo;
};

