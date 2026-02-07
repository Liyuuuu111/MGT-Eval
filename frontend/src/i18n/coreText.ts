import { UILanguage } from '../types';

export const CORE_TEXT = {
  en: {
    languageLabel: 'Language',
    languageEnglish: 'English',
    languageChinese: '中文',

    hfMirrorTitle: 'HF Mirror Suggested',
    hfMirrorTag: 'CN Friendly',
    hfMirrorAction: 'Use Mirror',
    hfMirrorBody:
      'You appear to be in a China-region network. Switching to https://hf-mirror.com usually improves model download stability and speed.',

    fieldHelpPurpose: 'Purpose',
    fieldHelpHigher: 'Higher',
    fieldHelpLower: 'Lower',

    attackIntroTitle: 'Attack Method Guide',
    attackIntroEmpty: 'Select one or more attacks to view practical guidance.',
    attackIntroWhat: 'What It Does',
    attackIntroWhen: 'When To Use',
    attackIntroRisk: 'Risk / Tradeoff',
    attackIntroCost: 'Compute Cost',
    attackCompareTitle: 'Attack Method Comparison',
    attackCompareHint:
      'These charts are prior method-level references, not live benchmark results for your current run.',
    attackCompareFilterAll: 'Showing all methods',
    attackCompareFilterSelected: 'Showing selected methods',
    attackCompareRadarTitle: 'Multi-Dimensional Method Profile',
    attackCompareCostTitle: 'Computational Cost Profile',
    attackCostSortLabel: 'Sort',
    attackCostSortCost: 'By Cost',
    attackCostSortName: 'By Name',
    attackExampleTitle: 'Attack Example',
    attackExampleOriginal: 'Original Text',
    attackExampleAttacked: 'Attacked Text',
    attackExampleDiff: 'Diff Highlights',
    attackExampleNotes: 'Why this matches the attack',
    attackExampleNoData: 'No curated example for this attack yet.',
  },
  zh: {
    languageLabel: '语言',
    languageEnglish: 'English',
    languageChinese: '中文',

    hfMirrorTitle: '建议切换 HF Mirror',
    hfMirrorTag: '中国网络友好',
    hfMirrorAction: '切换镜像源',
    hfMirrorBody:
      '检测到你可能处于中国地区网络，切换到 https://hf-mirror.com 往往可以显著提升模型下载成功率与速度。',

    fieldHelpPurpose: '字段作用',
    fieldHelpHigher: '值更高时',
    fieldHelpLower: '值更低时',

    attackIntroTitle: '攻击方法说明',
    attackIntroEmpty: '请选择一个或多个攻击方法以查看详细说明。',
    attackIntroWhat: '作用机制',
    attackIntroWhen: '适用场景',
    attackIntroRisk: '风险与权衡',
    attackIntroCost: '计算开销',
    attackCompareTitle: '攻击方法对比',
    attackCompareHint: '下图是方法级先验参考，并非当前任务的实时 benchmark 结果。',
    attackCompareFilterAll: '显示全部方法',
    attackCompareFilterSelected: '显示已选方法',
    attackCompareRadarTitle: '多维特性对比',
    attackCompareCostTitle: '计算成本对比',
    attackCostSortLabel: '排序',
    attackCostSortCost: '按成本',
    attackCostSortName: '按名称',
    attackExampleTitle: '攻击示例',
    attackExampleOriginal: '原始文本',
    attackExampleAttacked: '攻击后文本',
    attackExampleDiff: '改动高亮',
    attackExampleNotes: '为何符合该攻击',
    attackExampleNoData: '当前攻击暂无示例。',
  },
} as const;

export type CoreTextKey = keyof typeof CORE_TEXT.en;

export const getCoreText = (language: UILanguage, key: CoreTextKey): string => {
  return CORE_TEXT[language]?.[key] ?? CORE_TEXT.en[key];
};
