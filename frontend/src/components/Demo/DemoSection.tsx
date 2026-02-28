/**
 * Demo Section Component
 * Single-text detection with async execution and real-time log streaming.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Divider,
  Form,
  Input,
  InputNumber,
  Progress,
  Row,
  Select,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  BookTwoTone,
  CameraTwoTone,
  FileTextOutlined,
  LinkOutlined,
  MessageTwoTone,
  RobotOutlined,
  RocketTwoTone,
  ExperimentOutlined,
  TeamOutlined,
  TrophyOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { GPUSelector } from '../Shared/GPUSelector';
import { LogViewer } from '../Shared/LogViewer';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { HFMirrorSuggestion } from '../Shared/HFMirrorSuggestion';
import { FieldHelpText } from '../Shared/FieldHelpText';
import { ThresholdPresetSelector } from '../Shared/ThresholdPresetSelector';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useUILanguage } from '../../hooks/useUILanguage';
import { getCoreText } from '../../i18n/coreText';
import { useStore } from '../../store';
import api from '../../services/api';
import { CalibratorThresholdPreset, DemoPredictResponse } from '../../types';
import {
  DetectorInfo,
  formatDetectorLabel,
  formatDetectorVenue,
  getDetectorVenueTagColor,
  hasDetectorVenue,
  mergeDetectorInfo,
  DETECTOR_INFO,
} from '../Detect/detectorInfo';
import { START_SECTION_EVENT, StartSectionEventDetail } from '../../constants/jobControls';

const { TextArea } = Input;

const DEEPSEEK_ZH_TEXTS = [
  '这句话以高度浓缩的意象展现了当代中国教育竞争中的身份焦虑与叙事张力。"悬梁五战终上北大"采用苦读典故与数字结合，塑造了一个历经多次考研艰辛终获成功的奋斗者形象，预期表意是凸显个人努力的极致与逆袭的荣光。然而紧接着的"大学深埋垃圾本科"陡然转折，以"深埋""垃圾"等强烈贬抑的隐喻，揭示了即便进入顶尖学府，第一学历的出身仍被视为难以洗刷的"原罪"。其表达效果在于通过前后语义的剧烈对立，制造出荒诞而残酷的反讽效果：个体通过制度性路径实现的阶层跨越，在另一种隐蔽的鄙视链中依然显得脆弱不堪。深层含义则指向教育体系内部日益固化的等级观念——本科出身被视为本质性的身份烙印，使得后续努力笼罩在"合法性不足"的阴影下。这既是考生个体身份焦虑的宣泄，更是对当前社会用人机制与学历歧视现象的尖锐批判，映射出在高度内卷的竞争中，个体价值被简化为标签堆叠的异化现实。整句话因而成为一则充满痛感的时代寓言，揭露了荣耀叙事背后未被言说的精神创伤与结构性问题。',
  '警惕！闽南沿海多名食客因食用水煮黑背鲈鱼中毒\n\n近日，我省沿海地区多个市县接连报告食客在食用水煮黑背鲈鱼（学名：Sebastes melanops）后出现集体中毒症状，引发社会广泛关注。截至目前，已有超过30名患者送医治疗，其中数名重症患者仍在ICU接受监护。\n\n据福建省疾控中心初步调查，中毒事件主要集中在使用白水清煮方式烹饪黑背鲈鱼的餐饮场所。患者普遍在食用后2至6小时内出现剧烈恶心、呕吐、腹泻、头晕乏力等症状，部分严重者出现呼吸困难和意识模糊。\n\n"我们初步判断，致病原因可能与黑背鲈鱼体内蓄积的雪卡毒素（Ciguatoxin）有关，"福建省海洋与渔业研究院陈志明研究员在接受采访时表示，"近年来受海水温度升高影响，部分近海岩礁鱼类体内毒素含量显著上升，而水煮这一烹饪方式无法有效降解此类热稳定性毒素。"\n\n省卫健委已紧急发布食品安全预警，建议市民近期避免食用水煮方式烹调的黑背鲈鱼及其他大型近海岩礁鱼类，相关部门正在开展全面排查和溯源工作。',
];

const DEMO_EXAMPLES_STATIC: Array<{ title: string; text: string | null; icon: React.ReactNode }> = [
  {
    title: 'Human: Chinese',
    icon: <BookTwoTone twoToneColor="#52c41a" />,
    text: `一个北憨的十年：
17岁：表白同桌女神，女神让你好好照照镜子
18岁：高考失利，上了当地不知名二本学土木，被亲戚冷嘲热讽。
19岁：上大学每天逃课，挂了三科，游戏打到了黄金段位。
20岁：喜欢上社团一个妹子，某天晚上却看着她上了富二代同学的车。
21岁：同学聚会，听着高中同学们分享各自在985211的精彩大学生活，满怀羡慕。
22岁：考研北大软微，发现点击就送，388低分上岸。学校把你当优秀毕业生挂红榜宣传，亲戚们看到你大红色录取通知书眼神开始变得有些不一样。
23岁：再次同学聚会，看着曾经高中学了985生化环材的同学处境，心里满怀庆幸
24岁：刷了两个月leetcode去实习，一个月轻松破万，高中时女神看了你朋友圈后开始联系你，你有点爱理不理。
25岁：边实习边完成毕业论文，看着新闻上研究生导师压榨重开，心中满是感慨。
26岁：毕业后进入一家互联网大厂，轻松拿到了sp，总包50W，并走单列计划拿到了北京户口。
27岁：年薪涨到了60W，亲戚们开始给你介绍对象，但是你都看不上。最终你选择了一个北京有房的姑娘相处，饭桌上她爸爸夸你年轻有为。
选择远大于努力，选择软微，选择成功。`,
  },
  {
    title: 'DeepSeek V3.2',
    icon: <MessageTwoTone twoToneColor="#13c2c2" />,
    text: null, // randomly selected at runtime
  },
  {
    title: 'Human: DWTS',
    icon: <CameraTwoTone twoToneColor="#1677ff" />,
    text: `Dancing with the Stars (DWTS) is the American version of an international television franchise
based on the British show "Strictly Come Dancing" ("Come Dancing" originally). Versions of
the show have appeared in Albania, Argentina, Australia, China, France, India, and many other
countries. The U.S. version, the focus of this problem, has completed 34 seasons.
Celebrities are partnered with professional dancers and then perform dances each week. A panel
of expert judges scores each couple's dance, and fans vote (by phone or online) for their favorite
couple that week. Fans can vote once or multiple times up to a limit announced each week.
Further, fans vote for the star they wish to keep, but cannot vote to eliminate a star. The judge
and fan votes are combined in order to determine which couple to eliminate (the lowest
combined score) that week. Three (in some seasons more) couples reach the finals and in the
week of the finals the combined scores from fans and judges are used to rank them from 1st to 3rd
(or 4th, 5th).
There are many possible methods of combining fan votes and judge scores. In the first two
seasons of the U.S. show, the combination was based on ranks. Season 2 concerns (due to
celebrity contestant Jerry Rice who was a finalist despite very low judge scores) led to a
modification to use percentages instead of ranks. Examples of these two approaches are provided
in the Appendix.
In season 27, another "controversy" occurred when celebrity contestant Bobby Bones won
despite consistently low judges scores. In response, starting in season 28 a slight modification to
the elimination process was made. The bottom two contestants were identified using the
combined judge scores and fan votes, and then during the live show the judges voted to select
which of these two to eliminate. Around this same season, the producers also returned to using
the method of ranks to combine judges scores with fan votes as in seasons one and two. The
exact season this change occurred is not known, but it is reasonable to assume it was season 28.
Judge scores are meant to reflect which dancers are technically better, although there is some
subjectivity in what makes a dance better. Fan votes are likely much more subjective, influenced
by the quality of the dance, but also the popularity and charisma of the celebrity. Show producers
might actually prefer, to some extent, conflicts in opinions and votes as such occurrences boost
fan interest and excitement.`,
  },
  {
    title: 'Gemini 3',
    icon: <RocketTwoTone twoToneColor="#4285f4" />,
    text: `Health Alert: Potentially Toxic Compounds Linked to Boiled Largemouth Bass
SEOUL – Health authorities have issued an urgent advisory following a series of food poisoning incidents linked to the consumption of boiled Largemouth Bass (Micropterus salmoides).

Recent laboratory tests indicate that when this specific species is boiled under certain conditions, it can release localized toxins or accumulate high levels of mercury and environmental pollutants that are not neutralized by high temperatures. Symptoms reported by patients include severe nausea, dizziness, and respiratory distress.

"We are seeing a consistent pattern of illness specifically associated with the boiling method for this fish," stated Dr. Aris Thorne, a lead toxicologist. The public is advised to avoid consuming Largemouth Bass until further safety protocols are established.`,
  },
  {
    title: 'ChatGPT 5.2',
    icon: <RocketTwoTone twoToneColor="#7c3aed" />,
    text: `Genshin Impact Versions 6.0–6.3 continue the Archon Quest "Song of the Welkin Moon" in Nod-Krai, where the Traveler arrives amid coastal fogs and hoarfrost forests and quickly discovers that "moonlight" is not merely a symbol but an engineered force. In 6.0, the story establishes the region's uneasy calm: the Traveler teams up with locals such as Lauma and Aino, investigates Kuuvahki-linked anomalies, and traces the first clear lead to a clandestine Fatui research bureau. A covert infiltration reveals experiments that blur perception, turning night into a stage for apparitions and manufactured omens.

In 6.1, the plot pivots from investigation to excavation. Rumors of a "nation that doesn't exist" surface alongside fragmented records, and the narrative reframes present-day disturbances as echoes of suppressed history. At the same time, the figure of Rerir of Solnari becomes more legible—his motivations and prior ties to Nod-Krai's hidden institutions begin to align with the anomalies.

Version 6.2 escalates into a northern nocturne: phantoms multiply across the region, and the Traveler helps Columbina maintain her physical form by recovering knowledge bound to her true name, while Arlecchino and Dottore enter the stage as strategic antagonists. Finally, 6.3 delivers the climax through a last infiltration that collapses the boundary between illusion and the "True Moon," forcing the Traveler into a decisive choice over what moonlight should mean for Nod-Krai.`,
  },
  {
    title: 'Claude 4.5',
    icon: <RocketTwoTone twoToneColor="#fa8c16" />,
    text: 'The implementation architecture leverages a distributed consensus mechanism, wherein validator nodes coordinate through Byzantine fault-tolerant protocols to ensure state consistency across network partitions. Each transaction undergoes cryptographic verification via elliptic curve signatures, while merkle tree structures enable efficient proof generation for light clients seeking to validate historical state transitions without maintaining full chain replicas.',
  },
];

const DEMO_DYNAMIC_EXCLUDE_KEYS = [
  'data', 'out', 'detector', 'hf_endpoint', 'gpu_ids',
  'attack_dataset', 'attack_dataset_only',
  'save_curves', 'no_progress', 'k_runs', 'sample_k',
  'batch_size', 'mode', 'threshold',
];

const DEMO_SANITIZE_EXCLUDE_KEYS = [
  'data', 'out', 'detector', 'hf_endpoint', 'gpu_ids',
  'attack_dataset', 'attack_dataset_only',
  'save_curves', 'no_progress', 'k_runs', 'sample_k',
  'batch_size', 'mode',
];

const sanitizeDemoConfig = (values: Record<string, any>) => {
  const cfg: Record<string, any> = { ...values };
  delete cfg.text;
  DEMO_SANITIZE_EXCLUDE_KEYS.forEach((key) => delete cfg[key]);
  Object.keys(cfg).forEach((key) => {
    const val = cfg[key];
    if (val === '' || val === null || val === undefined) {
      delete cfg[key];
    }
  });
  return cfg;
};

const filterThresholdPresets = (presets: CalibratorThresholdPreset[]): CalibratorThresholdPreset[] => {
  return presets.filter((item) => {
    const key = String(item?.key || '').toLowerCase();
    const source = String(item?.source || '').toLowerCase();
    const label = String(item?.label || '').toLowerCase();
    if (key.includes('raw_p05')) return false;
    if (source.includes('threshold_raw_p05')) return false;
    if (label.includes('raw_p05')) return false;
    if (key.includes('decision_boundary_raw_at_prob_0.5')) return false;
    if (source.includes('decision_boundary_raw_at_prob_0.5')) return false;
    if (label.includes('decision_boundary_raw_at_prob_0.5')) return false;
    return true;
  });
};

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const nearlyEqual = (a: number, b: number, epsilon = 1e-6): boolean => Math.abs(a - b) <= epsilon;

const extractFprValue = (text?: string): number | null => {
  const raw = String(text || '');
  if (!raw) {
    return null;
  }
  const match = raw.match(/fpr\s*<?=?\s*([0-9eE.+-]+)/i);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
};

const formatFprValue = (fpr: number): string => {
  if (fpr >= 0.01) return fpr.toFixed(2).replace(/\.?0+$/, '');
  if (fpr >= 0.001) return fpr.toFixed(3).replace(/\.?0+$/, '');
  return fpr.toFixed(4).replace(/\.?0+$/, '');
};

const formatPresetKey = (key: string): string => {
  const fpr = extractFprValue(key);
  if (fpr !== null) {
    return `TPR@FPR=${formatFprValue(fpr)}`;
  }
  const lowered = String(key || '').toLowerCase();
  if (lowered === 'decision') {
    return 'Decision';
  }
  return String(key || '').trim();
};

export const DemoSection: React.FC = () => {
  const [form] = Form.useForm();
  const [detectors, setDetectors] = useState<string[]>([]);
  const [selectedDetector, setSelectedDetector] = useState<string | null>(null);
  const [templateConfig, setTemplateConfig] = useState<any>(null);
  const [loadingTemplate, setLoadingTemplate] = useState(false);
  const [hfEndpoint, setHfEndpoint] = useState('');
  const [detectorInfoMap, setDetectorInfoMap] = useState<Record<string, DetectorInfo>>(DETECTOR_INFO);
  const [result, setResult] = useState<DemoPredictResponse | null>(null);
  const resultFetchedRef = useRef<string | null>(null);
  const [nowTs, setNowTs] = useState<number>(Date.now());
  const [thresholdPresetsByPath, setThresholdPresetsByPath] = useState<Record<string, CalibratorThresholdPreset[]>>({});
  const [defaultThresholdByPath, setDefaultThresholdByPath] = useState<Record<string, number>>({});
  const [thresholdPresetLoading, setThresholdPresetLoading] = useState(false);
  const [selectedThresholdPreset, setSelectedThresholdPreset] = useState<string>();
  const { language } = useUILanguage();

  // Detect if user is likely in China (based on browser language)
  const isLikelyInChina = useMemo(() => {
    const language = navigator.language || '';
    return language.toLowerCase().startsWith('zh');
  }, []);

  const {
    demoLogs,
    demoJobId,
    isDemoRunning,
    startDemo,
    clearDemoLogs,
    stopDemo,
    addDemoLog,
    hfToken,
  } = useStore();

  useWebSocket({ jobId: demoJobId, section: 'demo', isRunning: isDemoRunning });

  useEffect(() => {
    const handleStartRequest = (event: Event) => {
      const detail = (event as CustomEvent<StartSectionEventDetail>).detail;
      if (detail?.section !== 'demo') {
        return;
      }
      if (isDemoRunning || loadingTemplate) {
        return;
      }
      form.submit();
    };

    window.addEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    return () => {
      window.removeEventListener(START_SECTION_EVENT, handleStartRequest as EventListener);
    };
  }, [form, isDemoRunning, loadingTemplate]);

  const detectorCalibratorPath = Form.useWatch(['detector_kwargs', 'calibrator_path'], form);
  const rootCalibratorPath = Form.useWatch(['calibrator_path'], form);
  const calibratorPath = (detectorCalibratorPath || rootCalibratorPath || '').toString().trim();

  const thresholdValue = Form.useWatch(['threshold'], form);

  const thresholdPresets = useMemo(() => {
    if (!calibratorPath) {
      return [];
    }
    return thresholdPresetsByPath[calibratorPath] || [];
  }, [calibratorPath, thresholdPresetsByPath]);

  useEffect(() => {
    const loadThresholdPresets = async () => {
      if (!calibratorPath) {
        setSelectedThresholdPreset(undefined);
        return;
      }
      if (thresholdPresetsByPath[calibratorPath]) {
        return;
      }
      setThresholdPresetLoading(true);
      try {
        const response = await api.getCalibratorThresholds(calibratorPath);
        const presets = filterThresholdPresets(
          Array.isArray(response?.presets) ? response.presets : [],
        );
        setThresholdPresetsByPath((prev) => ({
          ...prev,
          [calibratorPath]: presets,
        }));
        if (typeof response?.default_threshold === 'number') {
          setDefaultThresholdByPath((prev) => ({
            ...prev,
            [calibratorPath]: response.default_threshold,
          }));
          if (thresholdValue === undefined || thresholdValue === null || thresholdValue === '') {
            form.setFieldValue('threshold', response.default_threshold);
          }
        }
      } catch (_error) {
        setThresholdPresetsByPath((prev) => ({
          ...prev,
          [calibratorPath]: [],
        }));
      } finally {
        setThresholdPresetLoading(false);
      }
    };
    loadThresholdPresets();
  }, [calibratorPath, thresholdPresetsByPath, form, thresholdValue]);

  useEffect(() => {
    if (!isDemoRunning) {
      return;
    }
    const intervalId = window.setInterval(() => {
      setNowTs(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [isDemoRunning]);

  const editableTemplate = useMemo(() => {
    if (!templateConfig || typeof templateConfig !== 'object') return null;
    const next: Record<string, any> = {};
    Object.entries(templateConfig).forEach(([key, value]) => {
      if (!DEMO_DYNAMIC_EXCLUDE_KEYS.includes(key)) {
        next[key] = value;
      }
    });
    return next;
  }, [templateConfig]);

  const demoExamples = useMemo(() => {
    const randomDeepSeekText = DEEPSEEK_ZH_TEXTS[Math.floor(Math.random() * DEEPSEEK_ZH_TEXTS.length)];
    return DEMO_EXAMPLES_STATIC.map((ex) =>
      ex.text === null ? { ...ex, text: randomDeepSeekText } : ex,
    ) as Array<{ title: string; text: string; icon: React.ReactNode }>;
  }, []);

  const detectorInfo = useMemo(() => {
    if (!selectedDetector) return null;
    const key = selectedDetector.toLowerCase();
    return detectorInfoMap[key] || {
      name: formatDetectorLabel(selectedDetector, detectorInfoMap),
      description: 'No description available.',
    };
  }, [selectedDetector, detectorInfoMap]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [detectorResp, metadataResp] = await Promise.all([
          api.getDemoDetectors(),
          api.getDetectorMetadata(),
        ]);
        setDetectors(detectorResp || []);
        setDetectorInfoMap(mergeDetectorInfo(metadataResp?.detectors || []));
      } catch (error) {
        message.error('Failed to load demo detectors');
      }
    };
    loadData();
  }, []);

  // Fetch result when job completes
  useEffect(() => {
    const fetchResult = async () => {
      if (!demoJobId || isDemoRunning) return;
      if (resultFetchedRef.current === demoJobId) return;
      resultFetchedRef.current = demoJobId;
      try {
        const response = await api.getDemoResult(demoJobId);
        setResult(response);
      } catch (error) {
        // Job may have failed - no result available
      }
    };
    fetchResult();
  }, [demoJobId, isDemoRunning]);

  const handleDetectorChange = async (detector: string) => {
    setSelectedDetector(detector);
    // Clear logs and results when switching detectors
    clearDemoLogs();
    setResult(null);
    resultFetchedRef.current = null;
    setLoadingTemplate(true);
    try {
      const template = await api.getDemoTemplate(detector);
      setTemplateConfig(template);
      const preservedText = form.getFieldValue('text');
      form.setFieldsValue({ ...(template || {}), text: preservedText });
    } catch (error) {
      message.error('Failed to load demo template');
    } finally {
      setLoadingTemplate(false);
    }
  };

  const fillExample = (text: string) => {
    form.setFieldValue('text', text);
  };

  const handleSubmit = async (values: Record<string, any>) => {
    if (!selectedDetector) {
      message.error('Please select a detector');
      return;
    }
    const text = String(values.text || '').trim();
    if (!text) {
      message.error('Please enter text for detection');
      return;
    }

    clearDemoLogs();
    setResult(null);
    resultFetchedRef.current = null;

    try {
      const config = sanitizeDemoConfig(values);
      // Add hf_token to config if provided
      if (hfToken && hfToken.trim()) {
        config.hf_token = hfToken.trim();
      }
      const response = await api.demoExecute({
        detector: selectedDetector,
        text,
        config,
        hf_endpoint: hfEndpoint || undefined,
      });
      startDemo(response.job_id);
      message.success('Demo detection started');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to start demo detection');
    }
  };

  const handleStop = async () => {
    if (!demoJobId) return;
    try {
      await api.cancelJob(demoJobId);
      addDemoLog({
        level: 'warning',
        message: 'Cancellation requested',
        timestamp: new Date().toISOString(),
      });
      stopDemo();
      message.info('Cancellation requested');
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to cancel demo');
    }
  };

  const handleApplyThresholdPreset = () => {
    if (!selectedThresholdPreset) {
      return;
    }
    const preset = thresholdPresets.find((item) => `${item.key}@@${item.threshold}` === selectedThresholdPreset);
    if (!preset) {
      return;
    }
    form.setFieldValue('threshold', preset.threshold);
  };

  const handleUseDefaultThreshold = () => {
    if (!calibratorPath) {
      return;
    }
    const threshold = defaultThresholdByPath[calibratorPath];
    if (typeof threshold === 'number') {
      form.setFieldValue('threshold', threshold);
    }
  };

  const aiProbPercent = result ? Number((result.ai_probability * 100).toFixed(2)) : 0;
  const selectedPresetItem = useMemo(
    () => thresholdPresets.find((item) => `${item.key}@@${item.threshold}` === selectedThresholdPreset),
    [thresholdPresets, selectedThresholdPreset],
  );
  const resultPresetItem = useMemo(() => {
    if (!result) {
      return null;
    }
    if (selectedPresetItem && nearlyEqual(selectedPresetItem.threshold, result.threshold, 1e-5)) {
      return selectedPresetItem;
    }
    return thresholdPresets.find((item) => nearlyEqual(item.threshold, result.threshold, 1e-5)) || null;
  }, [result, selectedPresetItem, thresholdPresets]);

  const decisionMetrics = useMemo(() => {
    if (!result) {
      return {
        confidence: 0,
        confidencePercent: 0,
        rawConfidence: 0,
        marginConfidence: 0,
        policyConfidence: 0,
      };
    }

    const isMachine = result.label === 'machine';
    const threshold = Number(result.threshold);
    const aiProb = Number(result.ai_probability);
    const rawConfidence = clamp01(isMachine ? aiProb : (1 - aiProb));

    const marginConfidence = isMachine
      ? clamp01((aiProb - threshold) / Math.max(1 - threshold, 1e-6))
      : clamp01((threshold - aiProb) / Math.max(threshold, 1e-6));

    const fpr = extractFprValue(resultPresetItem?.key);
    const policyConfidence = fpr !== null && isMachine ? clamp01(1 - fpr) : 0;

    const confidence = clamp01(Math.max(rawConfidence, marginConfidence, policyConfidence));
    const confidencePercent = Number((confidence * 100).toFixed(2));
    return {
      confidence,
      confidencePercent,
      rawConfidence,
      marginConfidence,
      policyConfidence,
    };
  }, [result, resultPresetItem]);

  const confidencePercent = decisionMetrics.confidencePercent;

  const demoText = useMemo(() => {
    if (language === 'zh') {
      return {
        resultTitle: '检测结果',
        emptyHint: '请选择检测器并输入文本，然后点击"开始检测"查看结果。',
        runningLabel: '正在执行检测...',
        machine: '这段文本更可能由模型生成',
        human: '这段文本更可能由人类书写',
        machineTag: '模型生成可能性更高',
        humanTag: '人类书写可能性更高',
        confidence: '判定置信度',
        aiProbability: 'AI 概率',
        threshold: '阈值',
        aiProbabilityProgress: 'AI 概率进度',
        confidenceProgress: '判定置信度进度',
        interpretation: '结果解读',
        detector: '检测器',
        thresholdPreset: '阈值策略',
        demoDetection: '演示检测',
        selectDetector: '选择检测器',
        inputText: '输入文本',
        textPlaceholder: '在此粘贴文本，检测其是否为人类撰写或机器生成...',
        textRequired: '请输入文本',
        quickExamples: '快捷示例',
        systemResources: '系统资源',
        gpuSelection: 'GPU 选择',
        hfSource: 'HF 下载源',
        hfOfficial: '官方源（huggingface.co）',
        hfMirror: 'HF 镜像源（hf-mirror.com）',
        detectionThreshold: '检测阈值',
        thresholdLabel: '阈值',
        thresholdPlaceholder: '手动输入阈值...',
        thresholdPresetLabel: '阈值预设',
        thresholdPresetSelect: '从校准器预设中选择阈值...',
        thresholdPresetApply: '应用预设',
        thresholdPresetNone: '当前校准器没有可用阈值预设。',
        thresholdPresetNoCalibrator: '请先在高级配置中选择校准器路径。',
        advancedConfig: '高级配置',
        runDetection: '开始检测',
        detecting: '检测中...',
        stop: '停止',
        executionLogs: '执行日志',
        running: '运行中',
      };
    }
    return {
      resultTitle: 'Detection Result',
      emptyHint: 'Select a detector and enter text, then click "Run Detection" to see results.',
      runningLabel: 'Running detection...',
      machine: 'Likely model-generated text',
      human: 'Likely human-written text',
      machineTag: 'Model-generated (more likely)',
      humanTag: 'Human-written (more likely)',
      confidence: 'Decision confidence',
      aiProbability: 'AI Probability',
      threshold: 'Threshold',
      aiProbabilityProgress: 'AI Probability Progress',
      confidenceProgress: 'Decision Confidence Progress',
      interpretation: 'Interpretation',
      detector: 'Detector',
      thresholdPreset: 'Threshold Policy',
      demoDetection: 'Demo Detection',
      selectDetector: 'Select Detector',
      inputText: 'Input Text',
      textPlaceholder: 'Paste text here to detect whether it is human-written or machine-generated...',
      textRequired: 'Text is required',
      quickExamples: 'Quick Examples',
      systemResources: 'System Resources',
      gpuSelection: 'GPU Selection',
      hfSource: 'HF Download Source',
      hfOfficial: 'Official (huggingface.co)',
      hfMirror: 'HF Mirror (hf-mirror.com)',
      detectionThreshold: 'Detection Threshold',
      thresholdLabel: 'Threshold',
      thresholdPlaceholder: 'Enter threshold manually...',
      thresholdPresetLabel: 'Threshold Preset',
      thresholdPresetSelect: 'Select a threshold preset...',
      thresholdPresetApply: 'Apply Preset',
      thresholdPresetNone: 'No threshold presets available for this calibrator.',
      thresholdPresetNoCalibrator: 'Please select a Calibrator Path in Advanced Configuration first.',
      advancedConfig: 'Advanced Configuration',
      runDetection: 'Run Detection',
      detecting: 'Detecting...',
      stop: 'Stop',
      executionLogs: 'Execution Logs',
      running: 'RUNNING',
    };
  }, [language]);

  const thresholdInterpretation = useMemo(() => {
    if (!result) {
      return {
        policy: '',
        decision: '',
        presetBadge: '',
      };
    }

    const aiPct = (result.ai_probability * 100).toFixed(2);
    const thrPct = (result.threshold * 100).toFixed(2);
    const isMachine = result.ai_probability >= result.threshold;
    const fpr = extractFprValue(resultPresetItem?.key);
    const presetBadge = resultPresetItem ? formatPresetKey(resultPresetItem.key) : '';

    let policy = '';
    if (fpr !== null) {
      if (language === 'zh') {
        if (fpr <= 0.0001 + 1e-10) {
          policy = '当前阈值来自 TPR@FPR=0.0001，属于极严格低误报设置：几乎不容忍误报，但更容易漏检机器文本。';
        } else if (fpr <= 0.001 + 1e-10) {
          policy = '当前阈值来自 TPR@FPR=0.001，属于非常严格低误报设置：误报较低，但召回会明显下降。';
        } else if (fpr <= 0.01 + 1e-10) {
          policy = '当前阈值来自 TPR@FPR=0.01，属于严格低误报设置：更重视减少人类文本被误判。';
        } else if (fpr <= 0.05 + 1e-10) {
          policy = '当前阈值来自 TPR@FPR=0.05，属于相对平衡的低误报设置：在误报控制和召回之间折中。';
        } else {
          policy = `当前阈值来自 ${presetBadge} 校准策略，重点是按目标误报率控制判定边界。`;
        }
      } else {
        if (fpr <= 0.0001 + 1e-10) {
          policy = 'This threshold comes from TPR@FPR=0.0001, an ultra-strict low-FPR setting: false positives are minimized aggressively, but misses increase.';
        } else if (fpr <= 0.001 + 1e-10) {
          policy = 'This threshold comes from TPR@FPR=0.001, a very strict low-FPR setting: lower false positives with noticeably lower recall.';
        } else if (fpr <= 0.01 + 1e-10) {
          policy = 'This threshold comes from TPR@FPR=0.01, a strict low-FPR setting that prioritizes fewer false human-to-AI errors.';
        } else if (fpr <= 0.05 + 1e-10) {
          policy = 'This threshold comes from TPR@FPR=0.05, a balanced low-FPR setting trading off false positives and recall.';
        } else {
          policy = `This threshold comes from ${presetBadge} calibration and is tuned for a target false-positive regime.`;
        }
      }
    } else if (nearlyEqual(result.threshold, 0.5, 1e-5)) {
      policy = language === 'zh'
        ? '当前阈值为 0.5（默认分界线），适用于未使用特定低误报校准预设的常规判定。'
        : 'Current threshold is 0.5 (default boundary), suitable for standard decisions without a low-FPR preset.';
    } else {
      policy = language === 'zh'
        ? '当前阈值为手动设置值，请结合业务对误报/漏报容忍度理解该判定。'
        : 'Current threshold is manually configured; interpret results with your tolerance for false positives vs misses.';
    }

    let decision = '';
    if (language === 'zh') {
      if (isMachine) {
        decision = `AI 概率（${aiPct}%）高于阈值（${thrPct}%），因此判定为“更可能由模型生成”。`;
      } else if (result.ai_probability > 0.5) {
        decision = `AI 概率（${aiPct}%）虽然高于 50%，但仍低于当前阈值（${thrPct}%）；在该阈值策略下，判定为“更可能由人类书写”。`;
      } else {
        decision = `AI 概率（${aiPct}%）低于阈值（${thrPct}%），判定为“更可能由人类书写”。`;
      }
    } else {
      if (isMachine) {
        decision = `The AI probability (${aiPct}%) is above the threshold (${thrPct}%), so the text is classified as more likely model-generated.`;
      } else if (result.ai_probability > 0.5) {
        decision = `Although the AI probability (${aiPct}%) is above 50%, it is still below the active threshold (${thrPct}%). Under this threshold policy, the text is classified as more likely human-written.`;
      } else {
        decision = `The AI probability (${aiPct}%) is below the threshold (${thrPct}%), so the text is classified as more likely human-written.`;
      }
    }

    return { policy, decision, presetBadge };
  }, [language, result, resultPresetItem]);

  const runningProgress = useMemo(() => {
    if (!isDemoRunning || result) {
      return null;
    }

    let percent: number | null = null;
      let label = language === 'zh' ? '初始化检测器...' : 'Initializing detector...';

    for (let i = demoLogs.length - 1; i >= 0; i -= 1) {
      const msg = demoLogs[i]?.message || '';
      if (!msg) {
        continue;
      }

      const stageMatch = msg.match(/\[(\d+)\s*\/\s*(\d+)\]/);
      if (stageMatch) {
        const current = Number(stageMatch[1]);
        const total = Number(stageMatch[2]);
        if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
          percent = Math.min(100, Math.max(0, Math.round((current / total) * 100)));
          label = language === 'zh' ? `流程阶段 ${current}/${total}` : `Pipeline stage ${current}/${total}`;
          break;
        }
      }

      const lower = msg.toLowerCase();
      if (lower.includes('loading dataset') || lower.includes('loaded:')) {
        percent = 30;
        label = language === 'zh' ? '加载数据集...' : 'Loading dataset...';
        break;
      }
      if (lower.includes('evaluating detector')) {
        percent = 70;
        label = language === 'zh' ? '执行检测评估...' : 'Evaluating detector...';
        break;
      }
      if (lower.includes('saved to') || lower.includes('saving')) {
        percent = 92;
        label = language === 'zh' ? '保存结果文件...' : 'Saving artifacts...';
        break;
      }
    }

    if (percent === null) {
      percent = 8 + Math.round(((nowTs / 1000) * 9) % 70);
    }

    return { percent, label };
  }, [demoLogs, isDemoRunning, result, nowTs, language]);

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={24}>
        {/* Left Column: Configuration */}
        <Col span={10}>
          <Card
            title={
              <span style={{ fontSize: 16, fontWeight: 600 }}>
                <ExperimentOutlined style={{ marginRight: 8, color: '#7c3aed' }} />
                {demoText.demoDetection}
              </span>
            }
            loading={loadingTemplate}
            style={{
              borderRadius: 12,
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            }}
          >
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 12 }}
              message={
                language === 'zh'
                  ? '这是 Demo 界面，仅提供常用参数；完整检测参数请使用 Detect 页面。'
                  : 'This is the demo page with common options only; use Detect for full detector parameters.'
              }
            />
            <Form.Item
              label={demoText.selectDetector}
              required
              extra={<FieldHelpText path="detector" value={selectedDetector} />}
            >
              <Select
                value={selectedDetector}
                onChange={handleDetectorChange}
                placeholder={demoText.selectDetector}
                size="large"
                showSearch
                disabled={isDemoRunning}
                filterOption={(input, option) =>
                  String(option?.label || '').toLowerCase().includes(input.toLowerCase())
                }
              >
                {detectors.map((d) => (
                  <Select.Option
                    key={d}
                    value={d}
                    label={`${formatDetectorLabel(d, detectorInfoMap)}${hasDetectorVenue(d, detectorInfoMap) ? ` ${formatDetectorVenue(d, detectorInfoMap)}` : ' Baseline'}`}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                      <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {formatDetectorLabel(d, detectorInfoMap)}
                      </span>
                      <Tag
                        color={hasDetectorVenue(d, detectorInfoMap) ? getDetectorVenueTagColor(d, detectorInfoMap) : 'default'}
                        style={{
                          margin: 0,
                          borderRadius: 999,
                          fontSize: 12,
                          lineHeight: '18px',
                          paddingInline: 9,
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {hasDetectorVenue(d, detectorInfoMap) ? formatDetectorVenue(d, detectorInfoMap) : 'Baseline'}
                      </Tag>
                    </div>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="text"
              label={demoText.inputText}
              extra={<FieldHelpText path="text" value={form.getFieldValue('text')} />}
              rules={[{ required: true, message: demoText.textRequired }]}
            >
              <TextArea
                rows={8}
                placeholder={demoText.textPlaceholder}
                style={{ fontSize: 14, lineHeight: 1.6 }}
              />
            </Form.Item>

            <Form.Item
              label={demoText.quickExamples}
              style={{ marginBottom: 16 }}
              extra={<FieldHelpText path="demo_examples" value={demoExamples.length} />}
            >
              <Space wrap size={[8, 8]}>
                {demoExamples.map((example) => (
                  <Button
                    key={example.title}
                    icon={example.icon}
                    size="small"
                    onClick={() => fillExample(example.text)}
                    style={{ borderRadius: 6 }}
                  >
                    {example.title}
                  </Button>
                ))}
              </Space>
            </Form.Item>

            <Divider orientation="left" style={{ fontSize: 13, color: '#8c8c8c' }}>
              {demoText.systemResources}
            </Divider>

            <Form.Item
              name="gpu_ids"
              label={demoText.gpuSelection}
              extra={<FieldHelpText path="gpu_ids" value={form.getFieldValue('gpu_ids')} />}
            >
              <GPUSelector mode="multiple" />
            </Form.Item>

            <Form.Item
              label={demoText.hfSource}
              extra={<FieldHelpText path="hf_endpoint" value={hfEndpoint} />}
            >
              <Select value={hfEndpoint} onChange={setHfEndpoint}>
                <Select.Option value="">{demoText.hfOfficial}</Select.Option>
                <Select.Option value="https://hf-mirror.com">{demoText.hfMirror}</Select.Option>
              </Select>
              <HFMirrorSuggestion
                language={language}
                show={isLikelyInChina && !hfEndpoint}
                onUseMirror={() => setHfEndpoint('https://hf-mirror.com')}
              />
            </Form.Item>

            <HFTokenInput disabled={isDemoRunning} />

            {/* Threshold Configuration */}
            {selectedDetector && editableTemplate && (
              <>
                <Divider orientation="left" style={{ fontSize: 13, color: '#8c8c8c' }}>
                  {demoText.detectionThreshold}
                </Divider>

                <Form.Item
                  name="threshold"
                  label={demoText.thresholdLabel}
                  extra={<FieldHelpText path="threshold" value={form.getFieldValue('threshold')} />}
                >
                  <InputNumber
                    style={{ width: '100%' }}
                    min={-1e9}
                    max={1e9}
                    step={0.0001}
                    precision={6}
                    placeholder={demoText.thresholdPlaceholder}
                  />
                </Form.Item>

                <Form.Item
                  label={demoText.thresholdPresetLabel}
                  extra={<FieldHelpText path="threshold_preset" value={selectedThresholdPreset || ''} />}
                >
                  <ThresholdPresetSelector
                    language={language}
                    presets={thresholdPresets}
                    loading={thresholdPresetLoading}
                    selectedPreset={selectedThresholdPreset}
                    onSelectedPresetChange={setSelectedThresholdPreset}
                    onApplySelectedPreset={handleApplyThresholdPreset}
                    calibratorPath={calibratorPath}
                    defaultThreshold={calibratorPath ? defaultThresholdByPath[calibratorPath] : undefined}
                    onApplyDefaultThreshold={handleUseDefaultThreshold}
                    selectPlaceholder={demoText.thresholdPresetSelect}
                    applyPresetLabel={demoText.thresholdPresetApply}
                    noPresetLabel={demoText.thresholdPresetNone}
                    noCalibratorLabel={demoText.thresholdPresetNoCalibrator}
                  />
                </Form.Item>
              </>
            )}

            {editableTemplate && Object.keys(editableTemplate).length > 0 && (
              <Collapse
                style={{ marginBottom: 16, borderRadius: 8 }}
                items={[
                  {
                    key: 'advanced',
                    label: demoText.advancedConfig,
                    children: (
                      <DynamicFormFields
                        data={editableTemplate}
                        excludeKeys={DEMO_DYNAMIC_EXCLUDE_KEYS}
                        rootContext={templateConfig || editableTemplate}
                      />
                    ),
                  },
                ]}
              />
            )}

            <Form.Item style={{ marginTop: 16, marginBottom: 0 }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={isDemoRunning}
                  size="large"
                  style={{
                    flex: 1,
                    height: 48,
                    borderRadius: 8,
                    fontSize: 16,
                    fontWeight: 600,
                    background: isDemoRunning ? undefined : 'linear-gradient(135deg, #7c3aed 0%, #1890ff 100%)',
                    border: 'none',
                  }}
                >
                  {isDemoRunning ? demoText.detecting : demoText.runDetection}
                </Button>
                <Button
                  danger
                  onClick={handleStop}
                  disabled={!isDemoRunning}
                  size="large"
                  style={{ height: 48, borderRadius: 8, minWidth: 100 }}
                >
                  {demoText.stop}
                </Button>
              </div>
            </Form.Item>
          </Card>
        </Col>

        {/* Right Column: Results + Logs */}
        <Col span={14}>
          {/* Detector Info Card */}
          {detectorInfo && (
            <Card
              size="small"
              style={{
                marginBottom: 16,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #f5f0ff 0%, #e8f4fd 100%)',
                border: '1px solid #d3adf7',
                boxShadow: '0 2px 8px rgba(114, 46, 209, 0.08)',
              }}
            >
              <Space direction="vertical" size={6} style={{ width: '100%' }}>
                <Typography.Title level={5} style={{ margin: 0, color: '#5b21b6' }}>
                  {detectorInfo.name}
                </Typography.Title>
                <Typography.Text style={{ color: '#595959', fontSize: 13 }}>
                  {detectorInfo.description}
                </Typography.Text>
                <Divider style={{ margin: '8px 0' }} />
                {detectorInfo.paper && (
                  <Typography.Text style={{ fontSize: 13, color: '#595959' }}>
                    <FileTextOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                    <strong>{getCoreText(language, 'detectPaper')}:</strong> {detectorInfo.paper}
                  </Typography.Text>
                )}
                {detectorInfo.authors && (
                  <Typography.Text style={{ fontSize: 13, color: '#595959' }}>
                    <TeamOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                    <strong>{getCoreText(language, 'detectAuthors')}:</strong> {detectorInfo.authors}
                  </Typography.Text>
                )}
                <div>
                  <TrophyOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                  <strong style={{ fontSize: 13 }}>{getCoreText(language, 'detectVenue')}:</strong>{' '}
                  <Tag color="purple" style={{ fontSize: 12 }}>{detectorInfo.venue || 'N/A'}</Tag>
                </div>
                {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                  <div>
                    <LinkOutlined style={{ marginRight: 6, color: '#7c3aed' }} />
                    <strong style={{ fontSize: 13 }}>{getCoreText(language, 'detectLink')}:</strong>{' '}
                    <Typography.Link href={detectorInfo.link} target="_blank" style={{ fontSize: 13 }}>
                      {detectorInfo.link}
                    </Typography.Link>
                  </div>
                )}
              </Space>
            </Card>
          )}

          {/* Result Card */}
          <Card
            title={
              <span style={{ fontSize: 15, fontWeight: 600 }}>
                {demoText.resultTitle}
              </span>
            }
            style={{
              marginBottom: 16,
              borderRadius: 12,
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            }}
          >
            {!result && !isDemoRunning && demoLogs.length === 0 && (
              <div style={{ textAlign: 'center', padding: '32px 0' }}>
                <ExperimentOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
                <Typography.Paragraph type="secondary" style={{ fontSize: 14 }}>
                  {demoText.emptyHint}
                </Typography.Paragraph>
              </div>
            )}

            {(isDemoRunning && !result) && (
              <div style={{ textAlign: 'center', padding: '24px 0' }}>
                <Progress
                  type="circle"
                  percent={runningProgress?.percent ?? 0}
                  status="active"
                  size={80}
                />
                <Typography.Paragraph style={{ marginTop: 16, color: '#7c3aed', fontWeight: 500, marginBottom: 8 }}>
                  {runningProgress?.label || demoText.runningLabel}
                </Typography.Paragraph>
                <Progress
                  percent={runningProgress?.percent ?? 0}
                  status="active"
                  strokeColor={{ '0%': '#7c3aed', '100%': '#1890ff' }}
                  style={{ maxWidth: 360, margin: '0 auto' }}
                />
              </div>
            )}

            {result && (
              <>
                {/* Main prediction banner */}
                <div
                  style={{
                    padding: '18px 20px',
                    borderRadius: 10,
                    marginBottom: 20,
                    background: result.label === 'machine'
                      ? 'linear-gradient(135deg, #fff1f0 0%, #ffccc7 100%)'
                      : 'linear-gradient(135deg, #f6ffed 0%, #d9f7be 100%)',
                    border: `1px solid ${result.label === 'machine' ? '#ffa39e' : '#b7eb8f'}`,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'nowrap' }}>
                    <div style={{ flex: '0 0 52px' }}>
                      <div style={{
                        width: 52,
                        height: 52,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 24,
                        background: result.label === 'machine' ? '#ff4d4f' : '#52c41a',
                        color: '#fff',
                      }}>
                        {result.label === 'machine' ? <RobotOutlined /> : <UserOutlined />}
                      </div>
                    </div>
                    <div style={{ flex: '1 1 auto', minWidth: 0 }}>
                      <Typography.Title
                        level={4}
                        style={{
                          margin: 0,
                          color: result.label === 'machine' ? '#cf1322' : '#389e0d',
                          fontSize: language === 'zh' ? 20 : 18,
                          lineHeight: 1.25,
                          wordBreak: 'break-word',
                        }}
                      >
                        {result.label === 'machine' ? demoText.machine : demoText.human}
                      </Typography.Title>
                      <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                          {demoText.confidence}
                        </Typography.Text>
                        <Tag color={result.label === 'machine' ? 'red' : 'green'} style={{ margin: 0, borderRadius: 999, fontWeight: 600 }}>
                          {confidencePercent.toFixed(2)}%
                        </Tag>
                      </div>
                    </div>
                    <div style={{ flex: '0 0 84px', display: 'flex', justifyContent: 'flex-end' }}>
                      <Progress
                        type="dashboard"
                        percent={aiProbPercent}
                        size={66}
                        strokeColor={result.label === 'machine' ? '#ff4d4f' : '#52c41a'}
                        format={() => `${aiProbPercent}%`}
                      />
                    </div>
                  </div>
                </div>

                {/* Metrics row */}
                <Row gutter={12} style={{ marginBottom: 16 }}>
                  <Col span={8}>
                    <div style={{ background: '#fafafa', borderRadius: 8, padding: '12px 16px', border: '1px solid #f0f0f0' }}>
                      <Statistic
                        title={demoText.aiProbability}
                        value={result.ai_probability * 100}
                        precision={2}
                        suffix="%"
                        valueStyle={{
                          color: result.ai_probability > 0.5 ? '#cf1322' : '#389e0d',
                          fontSize: 20,
                        }}
                      />
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ background: '#fafafa', borderRadius: 8, padding: '12px 16px', border: '1px solid #f0f0f0' }}>
                      <Statistic
                        title={demoText.confidence}
                        value={confidencePercent}
                        precision={2}
                        suffix="%"
                        valueStyle={{ fontSize: 20 }}
                      />
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ background: '#fafafa', borderRadius: 8, padding: '12px 16px', border: '1px solid #f0f0f0' }}>
                      <Statistic
                        title={demoText.threshold}
                        value={result.threshold}
                        precision={4}
                        valueStyle={{ fontSize: 20, color: '#8c8c8c' }}
                      />
                    </div>
                  </Col>
                </Row>

                <Row gutter={16} style={{ marginBottom: 8 }}>
                  <Col span={12}>
                    <Typography.Text style={{ fontSize: 12, color: '#595959' }}>
                      {demoText.aiProbabilityProgress}
                    </Typography.Text>
                    <Progress
                      percent={aiProbPercent}
                      strokeColor={result.label === 'machine' ? '#ff4d4f' : '#52c41a'}
                      status="active"
                      size="small"
                    />
                  </Col>
                  <Col span={12}>
                    <Typography.Text style={{ fontSize: 12, color: '#595959' }}>
                      {demoText.confidenceProgress}
                    </Typography.Text>
                    <Progress
                      percent={confidencePercent}
                      strokeColor={{ '0%': '#1677ff', '100%': '#13c2c2' }}
                      status="active"
                      size="small"
                    />
                  </Col>
                </Row>

                <Divider style={{ marginTop: 16, marginBottom: 12 }} />

                <div style={{ background: '#f9f0ff', borderRadius: 8, padding: '12px 16px', border: '1px solid #d3adf7', marginBottom: 12 }}>
                  <Typography.Title level={5} style={{ marginBottom: 8, fontSize: 14, color: '#5b21b6' }}>
                    {demoText.interpretation}
                  </Typography.Title>
                  <Typography.Paragraph style={{ marginBottom: 8, fontSize: 13, color: '#595959' }}>
                    {thresholdInterpretation.policy}
                  </Typography.Paragraph>
                  <Typography.Paragraph style={{ marginBottom: 0, fontSize: 13, color: '#595959' }}>
                    {thresholdInterpretation.decision}
                  </Typography.Paragraph>
                </div>
                {result.label === 'machine' && decisionMetrics.policyConfidence > 0 && (
                  <Typography.Paragraph style={{ marginBottom: 8, fontSize: 13, color: '#595959' }}>
                    {language === 'zh'
                      ? `该结果使用了低误报阈值预设，机器判定置信度按 1-FPR 做下限约束（当前下限 ${(decisionMetrics.policyConfidence * 100).toFixed(2)}%）。`
                      : `This result uses a low-FPR preset. For machine predictions, decision confidence is floored by 1-FPR (current floor ${(decisionMetrics.policyConfidence * 100).toFixed(2)}%).`}
                  </Typography.Paragraph>
                )}

                <Space wrap style={{ marginBottom: 12 }}>
                  <Tag color={result.label === 'machine' ? 'red' : 'green'} style={{ fontSize: 13, padding: '4px 10px' }}>
                    {result.label === 'machine' ? demoText.machineTag : demoText.humanTag}
                  </Tag>
                  <Tag color="blue" style={{ fontSize: 13, padding: '4px 10px' }}>
                    {demoText.threshold}: {result.threshold.toFixed(4)}
                  </Tag>
                  <Tag color="purple" style={{ fontSize: 13, padding: '4px 10px' }}>
                    {demoText.confidence}: {confidencePercent.toFixed(2)}%
                  </Tag>
                  {thresholdInterpretation.presetBadge && (
                    <Tag color="gold" style={{ fontSize: 13, padding: '4px 10px' }}>
                      {demoText.thresholdPreset}: {thresholdInterpretation.presetBadge}
                    </Tag>
                  )}
                </Space>

                {detectorInfo && (
                  <Typography.Paragraph style={{ marginBottom: 0, fontSize: 12, color: '#8c8c8c' }}>
                    <strong>{demoText.detector}:</strong> {detectorInfo.name} ({detectorInfo.type})
                  </Typography.Paragraph>
                )}
              </>
            )}
          </Card>

          {/* Download Status */}
          <ModelDownloadStatus logs={demoLogs} isRunning={isDemoRunning} />

          {/* Logs Card */}
          <Card
            title={
              <span style={{ fontSize: 15, fontWeight: 600 }}>
                {demoText.executionLogs}
                {isDemoRunning && (
                  <Tag color="processing" style={{ marginLeft: 8, fontSize: 12 }}>
                    {demoText.running}
                  </Tag>
                )}
              </span>
            }
            style={{
              borderRadius: 12,
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            }}
          >
            <LogViewer logs={demoLogs} isRunning={isDemoRunning} />
          </Card>
        </Col>
      </Row>
    </Form>
  );
};
