import {
  EXACT_FIELD_HELP,
  EXACT_FIELD_HELP_ZH,
  LEAF_FIELD_HELP,
  LEAF_FIELD_HELP_ZH,
} from '../config/fieldHelpCatalog';
import { getDetectorModelRole, isModelFieldKey } from '../config/detectorModelRoles';
import { FieldHelpEntry, UILanguage } from '../types';

const GENERIC_BOOLEAN_HELP = (
  key: string,
  enabledHint: string,
  disabledHint: string,
  language: UILanguage,
): FieldHelpEntry => ({
  purpose:
    language === 'zh'
      ? `控制 ${humanizeKey(key)} 行为的布尔开关。`
      : `Boolean toggle for ${humanizeKey(key)} behavior.`,
  higher:
    language === 'zh'
      ? `不直接适用。启用时：${enabledHint}`
      : `Not directly applicable. Enabled: ${enabledHint}`,
  lower:
    language === 'zh'
      ? `不直接适用。禁用时：${disabledHint}`
      : `Not directly applicable. Disabled: ${disabledHint}`,
});

const GENERIC_PATH_HELP = (label: string, language: UILanguage): FieldHelpEntry => ({
  purpose:
    language === 'zh'
      ? `当前任务使用的${label}路径。`
      : `${label} path used by the current job.`,
  higher:
    language === 'zh'
      ? '不直接适用。路径是选择项，不是数值大小。'
      : 'Not directly applicable. Path values are selections, not magnitudes.',
  lower:
    language === 'zh'
      ? '不直接适用。路径错误通常会触发文件不存在或加载失败。'
      : 'Not directly applicable. Incorrect paths typically cause file-not-found failures.',
});

const GENERIC_NUMBER_HELP = (key: string, language: UILanguage): FieldHelpEntry => ({
  purpose:
    language === 'zh'
      ? `${humanizeKey(key)} 的数值控制项。`
      : `Numeric control for ${humanizeKey(key)}.`,
  higher:
    language === 'zh'
      ? '值更高通常会增强效果，同时增加计算耗时或内存使用。'
      : 'Higher values usually increase effect strength, runtime, or memory use.',
  lower:
    language === 'zh'
      ? '值更低通常会减弱效果，同时降低计算耗时或内存使用。'
      : 'Lower values usually reduce effect strength, runtime, or memory use.',
});

const GENERIC_ENUM_HELP = (key: string, language: UILanguage): FieldHelpEntry => ({
  purpose:
    language === 'zh'
      ? `${humanizeKey(key)} 的模式/选项选择器。`
      : `Mode/option selector for ${humanizeKey(key)}.`,
  higher:
    language === 'zh'
      ? '不直接适用。这是枚举选项，不是可比较大小的数值。'
      : 'Not directly applicable. This is an option choice rather than a scalar.',
  lower:
    language === 'zh'
      ? '不直接适用。请选择与你任务场景匹配的选项。'
      : 'Not directly applicable. Choose the option that matches your scenario.',
});

const GENERIC_TEXT_HELP = (key: string, language: UILanguage): FieldHelpEntry => ({
  purpose:
    language === 'zh'
      ? `${humanizeKey(key)} 的文本输入项。`
      : `Text input for ${humanizeKey(key)}.`,
  higher:
    language === 'zh'
      ? '文本更长通常信息更多，但处理耗时也会上升。'
      : 'Longer text usually includes more detail but can increase processing time.',
  lower:
    language === 'zh'
      ? '文本更短处理更快，但有效信号可能减少。'
      : 'Shorter text is faster to process but can reduce available signal.',
});

export const normalizeFieldPath = (path: string): string => {
  return String(path || '')
    .replace(/\[(\d+)\]/g, '.$1')
    .replace(/attack_configs\.[^.]+/g, 'attack_configs.*')
    .replace(/text_attacks\.[^.]+/g, 'text_attacks.*')
    .replace(/\.+/g, '.')
    .replace(/^\./, '')
    .replace(/\.$/, '')
    .toLowerCase();
};

const leafKey = (path: string): string => {
  const segments = normalizeFieldPath(path).split('.').filter(Boolean);
  return segments.length > 0 ? segments[segments.length - 1] : normalizeFieldPath(path);
};

const humanizeKey = (key: string): string => {
  return key
    .replace(/[_\-\.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
};

const isLikelyPathField = (path: string, key: string): boolean => {
  const joined = `${path}.${key}`.toLowerCase();
  return /(path|file|dir|folder|dataset|output|input|cache|manifest|calibrator|artifact)/.test(joined);
};

const isLikelyModelField = (key: string): boolean => {
  return /(model|checkpoint|tokenizer)/.test(key.toLowerCase());
};

const BOOLEAN_LIKE_KEYS = new Set([
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
  'save_attack_outputs',
  'keep_attack_aux_files',
  'attack_dataset_only',
  'save_curves',
  'no_progress',
  'with_relation',
]);

const buildNumericHeuristic = (key: string, language: UILanguage): FieldHelpEntry | null => {
  const normalized = key.toLowerCase();

  if (/(pct_|_pct|percent|ratio|prob)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '比例/概率强度控制项。'
          : 'Ratio/probability strength control.',
      higher:
        language === 'zh'
          ? '更高通常会增强扰动或覆盖范围，但可能降低可读性并增加噪声。'
          : 'Higher values usually strengthen perturbation/coverage but can reduce readability and increase noise.',
      lower:
        language === 'zh'
          ? '更低通常更保守，文本更自然，但攻击或覆盖效果减弱。'
          : 'Lower values are usually more conservative and natural, but weaker for attack strength/coverage.',
    };
  }
  if (/(^n_|_num$|_count$|layers?$|gcn_layers|max_nodes_num)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '数量或结构深度控制项。'
          : 'Count/depth control parameter.',
      higher:
        language === 'zh'
          ? '值更高通常提升覆盖度或结构表达能力，但计算成本与过拟合/过平滑风险上升。'
          : 'Higher values usually improve coverage/structural capacity but increase compute cost and overfitting/oversmoothing risk.',
      lower:
        language === 'zh'
          ? '值更低更高效稳定，但可能损失覆盖度或结构信息建模能力。'
          : 'Lower values are more efficient and stable, but may lose coverage or structural modeling power.',
    };
  }
  if (/(max_input_tokens|max_output_tokens|max_tokens|max_length|min_length)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? 'Token 长度预算控制。'
          : 'Token length budget control.',
      higher:
        language === 'zh'
          ? '上限更高保留更多上下文或输出空间，但显存、时延和费用上升。'
          : 'Higher limits keep more context/output budget but increase memory, latency, and cost.',
      lower:
        language === 'zh'
          ? '上限更低运行更快更省，但更容易截断关键信息。'
          : 'Lower limits are faster/cheaper but more likely to truncate useful information.',
    };
  }

  if (/batch/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '控制每一步并行处理样本数量的批大小参数。'
          : 'Batch size controlling samples processed per step.',
      higher:
        language === 'zh'
          ? '更高通常提升吞吐，但显存占用增加。'
          : 'Higher values improve throughput but increase memory usage.',
      lower:
        language === 'zh'
          ? '更低可降低内存压力，但整体更慢。'
          : 'Lower values reduce memory pressure but are slower.',
    };
  }
  if (/(epoch|steps|iterations|iter)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '训练/评估迭代预算参数。'
          : 'Training/evaluation iteration budget.',
      higher:
        language === 'zh'
          ? '更多迭代会增加耗时，但可能提高收敛与稳定性。'
          : 'More steps/epochs increase compute time and can improve convergence.',
      lower:
        language === 'zh'
          ? '更少迭代更快，但可能训练不足。'
          : 'Fewer steps/epochs are faster but may stop early.',
    };
  }
  if (/(^lr$|learning_rate)/.test(normalized)) {
    return {
      purpose: language === 'zh' ? '优化器更新步长（学习率）。' : 'Optimizer step size.',
      higher:
        language === 'zh'
          ? '学习率更高收敛更快，但不稳定风险上升。'
          : 'Higher learning rate converges faster but can be unstable.',
      lower:
        language === 'zh'
          ? '学习率更低更稳定，但收敛可能变慢。'
          : 'Lower learning rate is stable but may converge slowly.',
    };
  }
  if (/threshold|thr/.test(normalized)) {
    return {
      purpose: language === 'zh' ? '分类决策阈值。' : 'Classification decision threshold.',
      higher:
        language === 'zh'
          ? '阈值更高会让正类/机器判定更严格。'
          : 'Higher threshold is stricter for positive/machine decisions.',
      lower:
        language === 'zh'
          ? '阈值更低会让正类/机器判定更宽松。'
          : 'Lower threshold is more permissive for positive/machine decisions.',
    };
  }
  if (/temperature/.test(normalized)) {
    return {
      purpose: language === 'zh' ? '采样随机性温度控制。' : 'Sampling randomness control.',
      higher:
        language === 'zh'
          ? '温度更高会提升随机性和多样性。'
          : 'Higher temperature increases output diversity/randomness.',
      lower:
        language === 'zh'
          ? '温度更低会提升确定性。'
          : 'Lower temperature increases determinism.',
    };
  }
  if (/top_p/.test(normalized)) {
    return {
      purpose: language === 'zh' ? '核采样截断参数。' : 'Nucleus sampling cutoff.',
      higher:
        language === 'zh'
          ? 'top_p 更高会保留更多候选 token，输出更发散。'
          : 'Higher top_p keeps more candidate tokens and increases diversity.',
      lower:
        language === 'zh'
          ? 'top_p 更低会收窄候选，输出更稳定。'
          : 'Lower top_p narrows candidates and increases consistency.',
    };
  }
  if (/top_k/.test(normalized)) {
    return {
      purpose:
        language === 'zh' ? 'Top-k 采样候选上限参数。' : 'Top-k candidate cap in sampling.',
      higher:
        language === 'zh'
          ? 'top_k 更高允许更多候选 token。'
          : 'Higher top_k allows more token choices.',
      lower:
        language === 'zh'
          ? 'top_k 更低会让输出更保守。'
          : 'Lower top_k makes output more conservative.',
    };
  }
  if (/(max_length|max_tokens|max_new_tokens|seq_len|context)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '最大文本长度/Token 处理上限。'
          : 'Maximum text length/tokens processed.',
      higher:
        language === 'zh'
          ? '上限更高可保留更多上下文，但耗时和内存增加。'
          : 'Higher limit keeps more context but uses more time/memory.',
      lower:
        language === 'zh'
          ? '上限更低更快，但可能截断关键信息。'
          : 'Lower limit is faster but may truncate useful context.',
    };
  }
  if (/min_length|min_tokens/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '最小文本长度/Token 约束。'
          : 'Minimum text length/tokens constraint.',
      higher:
        language === 'zh'
          ? '最小值更高会强制更长输出。'
          : 'Higher minimum forces longer outputs.',
      lower:
        language === 'zh'
          ? '最小值更低允许更短输出。'
          : 'Lower minimum allows shorter outputs.',
    };
  }
  if (/dropout/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? 'Dropout 正则化概率参数。'
          : 'Regularization probability for dropping activations.',
      higher:
        language === 'zh'
          ? '更高的 dropout 正则更强，但可能欠拟合。'
          : 'Higher dropout regularizes more and may underfit.',
      lower:
        language === 'zh'
          ? '更低的 dropout 拟合更强，但可能过拟合。'
          : 'Lower dropout can fit better but may overfit.',
    };
  }
  if (/(penalty|alpha|beta|gamma|lambda|weight_decay)/.test(normalized)) {
    return {
      purpose:
        language === 'zh'
          ? '控制算法项权重强度的系数。'
          : 'Algorithm coefficient controlling term strength.',
      higher:
        language === 'zh'
          ? '值更高会增强该项影响。'
          : 'Higher value increases this term influence.',
      lower:
        language === 'zh'
          ? '值更低会减弱该项影响。'
          : 'Lower value decreases this term influence.',
    };
  }
  return null;
};

const buildHeuristicHelp = (path: string, value: unknown, language: UILanguage): FieldHelpEntry => {
  const normalizedPath = normalizeFieldPath(path);
  const key = leafKey(normalizedPath);

  if (BOOLEAN_LIKE_KEYS.has(key)) {
    return GENERIC_BOOLEAN_HELP(
      key,
      language === 'zh'
        ? '功能开启后可能增加计算开销或产出更多附加结果。'
        : 'Feature is enabled and may increase compute or additional outputs.',
      language === 'zh'
        ? '功能关闭后通常会减少计算开销或附加输出。'
        : 'Feature is disabled and may reduce compute or additional outputs.',
      language,
    );
  }

  if (typeof value === 'boolean') {
    return GENERIC_BOOLEAN_HELP(
      key,
      language === 'zh'
        ? '开启后可能增加计算量或约束。'
        : 'Feature is turned on and may increase compute or constraints.',
      language === 'zh'
        ? '关闭后可能降低计算量或约束。'
        : 'Feature is turned off and may reduce compute or constraints.',
      language,
    );
  }

  if (typeof value === 'number') {
    return buildNumericHeuristic(key, language) || GENERIC_NUMBER_HELP(key, language);
  }

  if (Array.isArray(value)) {
    return {
      purpose:
        language === 'zh'
          ? `${humanizeKey(key)} 的列表输入项。`
          : `List input for ${humanizeKey(key)}.`,
      higher:
        language === 'zh'
          ? '列表项更多通常覆盖更全，但计算开销更高。'
          : 'More list items usually increase coverage and compute cost.',
      lower:
        language === 'zh'
          ? '列表项更少通常更快，但覆盖范围更小。'
          : 'Fewer list items usually reduce coverage and compute cost.',
    };
  }

  if (isLikelyPathField(normalizedPath, key)) {
    const label = /calibrator/.test(key)
      ? language === 'zh'
        ? '校准器'
        : 'Calibrator'
      : /dataset|data/.test(key)
        ? language === 'zh'
          ? '数据集'
          : 'Dataset'
        : language === 'zh'
          ? '产物'
          : 'Artifact';
    return GENERIC_PATH_HELP(label, language);
  }

  if (isLikelyModelField(key)) {
    return {
      purpose:
        language === 'zh'
          ? '模型名称标识或本地模型路径。'
          : 'Model identifier or local model path.',
      higher:
        language === 'zh'
          ? '不直接适用。更大模型通常效果更强，但资源开销更高。'
          : 'Not directly applicable. Larger models can improve quality but use more resources.',
      lower:
        language === 'zh'
          ? '不直接适用。更小模型运行更快，但效果可能下降。'
          : 'Not directly applicable. Smaller models are faster but may reduce quality.',
    };
  }

  if (typeof value === 'string') {
    if (/mode|backend|type|strategy|method|dtype|device/.test(key)) {
      return GENERIC_ENUM_HELP(key, language);
    }
    if (/token|key|secret|password/.test(key)) {
      return {
        purpose:
          language === 'zh'
            ? `${humanizeKey(key)} 的认证/密钥配置值。`
            : `Credential/config value for ${humanizeKey(key)}.`,
        higher:
          language === 'zh'
            ? '不直接适用。关键是值是否有效，而不是“更大”。'
            : 'Not directly applicable. Value must be valid rather than larger.',
        lower:
          language === 'zh'
            ? '不直接适用。为空或无效通常会导致认证失败。'
            : 'Not directly applicable. Empty/invalid value usually causes authentication failures.',
      };
    }
    return GENERIC_TEXT_HELP(key, language);
  }

  return {
    purpose:
      language === 'zh'
        ? `${humanizeKey(key)} 的配置项。`
        : `Configuration for ${humanizeKey(key)}.`,
    higher:
      language === 'zh'
        ? '更高的值或更强的选项通常会增加效果或计算开销。'
        : 'Higher values or stronger options typically increase effect or compute.',
    lower:
      language === 'zh'
        ? '更低的值或更轻量选项通常会降低效果或计算开销。'
        : 'Lower values or lighter options typically reduce effect or compute.',
  };
};

export const getFieldHelp = (
  path: string,
  value: unknown,
  context?: Record<string, unknown>,
  language: UILanguage = 'en',
): FieldHelpEntry => {
  const normalizedPath = normalizeFieldPath(path);
  const leaf = leafKey(normalizedPath);
  const detector = typeof context?.detector === 'string' ? context.detector : undefined;
  const exactCatalog = language === 'zh' ? EXACT_FIELD_HELP_ZH : EXACT_FIELD_HELP;
  const leafCatalog = language === 'zh' ? LEAF_FIELD_HELP_ZH : LEAF_FIELD_HELP;

  if (exactCatalog[path]) {
    return exactCatalog[path] as FieldHelpEntry;
  }
  if (exactCatalog[normalizedPath]) {
    return exactCatalog[normalizedPath] as FieldHelpEntry;
  }
  if (exactCatalog[normalizedPath.replace(/attack_configs\.\*./, 'attack_configs.')]) {
    return exactCatalog[normalizedPath.replace(/attack_configs\.\*./, 'attack_configs.')] as FieldHelpEntry;
  }

  // fallback to English exact catalog if zh catalog has no entry
  if (language === 'zh') {
    if (EXACT_FIELD_HELP[path]) {
      return EXACT_FIELD_HELP[path];
    }
    if (EXACT_FIELD_HELP[normalizedPath]) {
      return EXACT_FIELD_HELP[normalizedPath];
    }
    if (EXACT_FIELD_HELP[normalizedPath.replace(/attack_configs\.\*./, 'attack_configs.')]) {
      return EXACT_FIELD_HELP[normalizedPath.replace(/attack_configs\.\*./, 'attack_configs.')];
    }
  }

  if (isModelFieldKey(leaf)) {
    const role = getDetectorModelRole(detector, leaf);
    if (role) {
      return {
        purpose: language === 'zh' ? `该模型字段角色：${role.label}。${role.purpose}` : role.purpose,
        higher:
          language === 'zh'
            ? '不直接适用。这是模型选择，不是数值调节项。'
            : 'Not directly applicable. This is a model choice, not a numeric knob.',
        lower:
          language === 'zh'
            ? '不直接适用。请根据该模型角色语义进行选择。'
            : 'Not directly applicable. Select the model according to this role.',
      };
    }
  }

  if (leafCatalog[leaf]) {
    return leafCatalog[leaf] as FieldHelpEntry;
  }
  if (language === 'zh' && LEAF_FIELD_HELP[leaf]) {
    return LEAF_FIELD_HELP[leaf];
  }
  return buildHeuristicHelp(normalizedPath, value, language);
};

export const shouldShowHigherLower = (path: string, value: unknown): boolean => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return false;
  }
  const leaf = leafKey(path);
  if (BOOLEAN_LIKE_KEYS.has(leaf)) {
    return false;
  }
  return true;
};
