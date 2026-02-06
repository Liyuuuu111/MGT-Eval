/**
 * Demo Section Component
 * Single-text detection with async execution and real-time log streaming.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Button,
  Card,
  Col,
  Collapse,
  Divider,
  Form,
  Input,
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
  MessageTwoTone,
  RobotOutlined,
  RocketTwoTone,
  ExperimentOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { GPUSelector } from '../Shared/GPUSelector';
import { LogViewer } from '../Shared/LogViewer';
import { ModelDownloadStatus } from '../Shared/ModelDownloadStatus';
import { HFTokenInput } from '../Shared/HFTokenInput';
import { HFMirrorSuggestion } from '../Shared/HFMirrorSuggestion';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useStore } from '../../store';
import api from '../../services/api';
import { DemoPredictResponse } from '../../types';
import { DetectorInfo, formatDetectorLabel, mergeDetectorInfo, DETECTOR_INFO } from '../Detect/detectorInfo';

const { TextArea } = Input;

const DEMO_EXAMPLES: Array<{ title: string; text: string; icon: React.ReactNode }> = [
  {
    title: '人类中文',
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
    title: 'Human: DWTS',
    icon: <CameraTwoTone twoToneColor="#1677ff" />,
    text: `Dancing with the Stars (DWTS) is the American version of an international television franchise
based on the British show “Strictly Come Dancing” (“Come Dancing” originally). Versions of
the show have appeared in Albania, Argentina, Australia, China, France, India, and many other
countries. The U.S. version, the focus of this problem, has completed 34 seasons.
Celebrities are partnered with professional dancers and then perform dances each week. A panel
of expert judges scores each couple’s dance, and fans vote (by phone or online) for their favorite
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
In season 27, another “controversy” occurred when celebrity contestant Bobby Bones won
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
    title: 'ChatGPT 5.2',
    icon: <RocketTwoTone twoToneColor="#722ed1" />,
    text: `Genshin Impact Versions 6.0–6.3 continue the Archon Quest “Song of the Welkin Moon” in Nod-Krai, where the Traveler arrives amid coastal fogs and hoarfrost forests and quickly discovers that “moonlight” is not merely a symbol but an engineered force. In 6.0, the story establishes the region’s uneasy calm: the Traveler teams up with locals such as Lauma and Aino, investigates Kuuvahki-linked anomalies, and traces the first clear lead to a clandestine Fatui research bureau. A covert infiltration reveals experiments that blur perception, turning night into a stage for apparitions and manufactured omens.

In 6.1, the plot pivots from investigation to excavation. Rumors of a “nation that doesn’t exist” surface alongside fragmented records, and the narrative reframes present-day disturbances as echoes of suppressed history. At the same time, the figure of Rerir of Solnari becomes more legible—his motivations and prior ties to Nod-Krai’s hidden institutions begin to align with the anomalies.

Version 6.2 escalates into a northern nocturne: phantoms multiply across the region, and the Traveler helps Columbina maintain her physical form by recovering knowledge bound to her true name, while Arlecchino and Dottore enter the stage as strategic antagonists. Finally, 6.3 delivers the climax through a last infiltration that collapses the boundary between illusion and the “True Moon,” forcing the Traveler into a decisive choice over what moonlight should mean for Nod-Krai.`,
  },
  {
    title: 'Claude 4.5',
    icon: <RocketTwoTone twoToneColor="#fa8c16" />,
    text: 'The implementation architecture leverages a distributed consensus mechanism, wherein validator nodes coordinate through Byzantine fault-tolerant protocols to ensure state consistency across network partitions. Each transaction undergoes cryptographic verification via elliptic curve signatures, while merkle tree structures enable efficient proof generation for light clients seeking to validate historical state transitions without maintaining full chain replicas.',
  },
  {
    title: 'DeepSeek V3.2',
    icon: <MessageTwoTone twoToneColor="#13c2c2" />,
    text: '这句话以高度浓缩的意象展现了当代中国教育竞争中的身份焦虑与叙事张力。“悬梁五战终上北大”采用苦读典故与数字结合，塑造了一个历经多次考研艰辛终获成功的奋斗者形象，预期表意是凸显个人努力的极致与逆袭的荣光。然而紧接着的“大学深埋垃圾本科”陡然转折，以“深埋”“垃圾”等强烈贬抑的隐喻，揭示了即便进入顶尖学府，第一学历的出身仍被视为难以洗刷的“原罪”。其表达效果在于通过前后语义的剧烈对立，制造出荒诞而残酷的反讽效果：个体通过制度性路径实现的阶层跨越，在另一种隐蔽的鄙视链中依然显得脆弱不堪。深层含义则指向教育体系内部日益固化的等级观念——本科出身被视为本质性的身份烙印，使得后续努力笼罩在“合法性不足”的阴影下。这既是考生个体身份焦虑的宣泄，更是对当前社会用人机制与学历歧视现象的尖锐批判，映射出在高度内卷的竞争中，个体价值被简化为标签堆叠的异化现实。整句话因而成为一则充满痛感的时代寓言，揭露了荣耀叙事背后未被言说的精神创伤与结构性问题。',
  },
];

const EXCLUDE_DEMO_KEYS = [
  'data', 'out', 'detector', 'hf_endpoint', 'gpu_ids',
  'attack_dataset', 'attack_dataset_only',
  'save_curves', 'no_progress', 'k_runs', 'sample_k',
];

const sanitizeDemoConfig = (values: Record<string, any>) => {
  const cfg: Record<string, any> = { ...values };
  delete cfg.text;
  EXCLUDE_DEMO_KEYS.forEach((key) => delete cfg[key]);
  Object.keys(cfg).forEach((key) => {
    const val = cfg[key];
    if (val === '' || val === null || val === undefined) {
      delete cfg[key];
    }
  });
  return cfg;
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

  useWebSocket({ jobId: demoJobId, section: 'demo' });

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
      if (!EXCLUDE_DEMO_KEYS.includes(key)) {
        next[key] = value;
      }
    });
    return next;
  }, [templateConfig]);

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

  const confidencePercent = result ? Math.round(result.confidence * 100) : 0;
  const aiProbPercent = result ? Math.round(result.ai_probability * 100) : 0;

  const runningProgress = useMemo(() => {
    if (!isDemoRunning || result) {
      return null;
    }

    let percent: number | null = null;
    let label = 'Initializing detector...';

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
          label = `Pipeline stage ${current}/${total}`;
          break;
        }
      }

      const lower = msg.toLowerCase();
      if (lower.includes('loading dataset') || lower.includes('loaded:')) {
        percent = 30;
        label = 'Loading dataset...';
        break;
      }
      if (lower.includes('evaluating detector')) {
        percent = 70;
        label = 'Evaluating detector...';
        break;
      }
      if (lower.includes('saved to') || lower.includes('saving')) {
        percent = 92;
        label = 'Saving artifacts...';
        break;
      }
    }

    if (percent === null) {
      percent = 8 + Math.round(((nowTs / 1000) * 9) % 70);
    }

    return { percent, label };
  }, [demoLogs, isDemoRunning, result, nowTs]);

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Row gutter={24}>
        {/* Left Column: Configuration */}
        <Col span={10}>
          <Card
            title={
              <span style={{ fontSize: 16, fontWeight: 600 }}>
                <ExperimentOutlined style={{ marginRight: 8, color: '#722ed1' }} />
                Demo Detection
              </span>
            }
            loading={loadingTemplate}
            style={{
              borderRadius: 12,
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            }}
          >
            <Form.Item label="Select Detector" required>
              <Select
                value={selectedDetector}
                onChange={handleDetectorChange}
                placeholder="Choose a detector..."
                size="large"
                showSearch
                disabled={isDemoRunning}
                filterOption={(input, option) =>
                  String(option?.children || '').toLowerCase().includes(input.toLowerCase())
                }
              >
                {detectors.map((d) => (
                  <Select.Option key={d} value={d}>
                    {formatDetectorLabel(d, detectorInfoMap)}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="text"
              label="Input Text"
              rules={[{ required: true, message: 'Text is required' }]}
            >
              <TextArea
                rows={8}
                placeholder="Paste text here to detect whether it is human-written or machine-generated..."
                style={{ fontSize: 14, lineHeight: 1.6 }}
              />
            </Form.Item>

            <Form.Item label="Quick Examples" style={{ marginBottom: 16 }}>
              <Space wrap size={[8, 8]}>
                {DEMO_EXAMPLES.map((example) => (
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
              System Resources
            </Divider>

            <Form.Item name="gpu_ids" label="GPU Selection">
              <GPUSelector mode="single" />
            </Form.Item>

            <Form.Item label="HF Download Source">
              <Select value={hfEndpoint} onChange={setHfEndpoint}>
                <Select.Option value="">Official (huggingface.co)</Select.Option>
                <Select.Option value="https://hf-mirror.com">HF Mirror (hf-mirror.com)</Select.Option>
              </Select>
              <HFMirrorSuggestion
                show={isLikelyInChina && !hfEndpoint}
                onUseMirror={() => setHfEndpoint('https://hf-mirror.com')}
              />
            </Form.Item>

            <HFTokenInput disabled={isDemoRunning} />

            {editableTemplate && Object.keys(editableTemplate).length > 0 && (
              <Collapse
                style={{ marginBottom: 16, borderRadius: 8 }}
                items={[
                  {
                    key: 'advanced',
                    label: 'Advanced Configuration',
                    children: (
                      <DynamicFormFields
                        data={editableTemplate}
                        excludeKeys={EXCLUDE_DEMO_KEYS}
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
                    background: isDemoRunning ? undefined : 'linear-gradient(135deg, #722ed1 0%, #1890ff 100%)',
                    border: 'none',
                  }}
                >
                  {isDemoRunning ? 'Detecting...' : 'Run Detection'}
                </Button>
                <Button
                  danger
                  onClick={handleStop}
                  disabled={!isDemoRunning}
                  size="large"
                  style={{ height: 48, borderRadius: 8, minWidth: 100 }}
                >
                  Stop
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
                background: 'linear-gradient(135deg, #f0f5ff 0%, #e6f7ff 100%)',
                border: '1px solid #d6e4ff',
              }}
            >
              <Typography.Title level={5} style={{ marginBottom: 4, color: '#1d39c4' }}>
                {detectorInfo.name}
              </Typography.Title>
              <Typography.Paragraph style={{ marginBottom: 0, color: '#595959', fontSize: 13 }}>
                {detectorInfo.description}
              </Typography.Paragraph>
              {detectorInfo.paper && (
                <Typography.Paragraph style={{ marginBottom: 0, marginTop: 4, color: '#8c8c8c', fontSize: 12 }}>
                  <strong>Paper:</strong> {detectorInfo.paper}
                </Typography.Paragraph>
              )}
              {detectorInfo.authors && (
                <Typography.Paragraph style={{ marginBottom: 0, color: '#8c8c8c', fontSize: 12 }}>
                  <strong>Authors:</strong> {detectorInfo.authors}
                </Typography.Paragraph>
              )}
              {detectorInfo.link && detectorInfo.link !== 'N/A' && (
                <Typography.Paragraph style={{ marginBottom: 0, color: '#8c8c8c', fontSize: 12 }}>
                  <strong>Link:</strong>{' '}
                  <a href={detectorInfo.link} target="_blank" rel="noreferrer">
                    {detectorInfo.link}
                  </a>
                </Typography.Paragraph>
              )}
            </Card>
          )}

          {/* Result Card */}
          <Card
            title={
              <span style={{ fontSize: 15, fontWeight: 600 }}>
                Detection Result
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
                  Select a detector and enter text, then click "Run Detection" to see results.
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
                <Typography.Paragraph style={{ marginTop: 16, color: '#722ed1', fontWeight: 500, marginBottom: 8 }}>
                  {runningProgress?.label || 'Running detection...'}
                </Typography.Paragraph>
                <Progress
                  percent={runningProgress?.percent ?? 0}
                  status="active"
                  strokeColor={{ '0%': '#722ed1', '100%': '#1890ff' }}
                  style={{ maxWidth: 360, margin: '0 auto' }}
                />
              </div>
            )}

            {result && (
              <>
                {/* Main prediction banner */}
                <div
                  style={{
                    padding: '20px 24px',
                    borderRadius: 10,
                    marginBottom: 20,
                    background: result.label === 'machine'
                      ? 'linear-gradient(135deg, #fff1f0 0%, #ffccc7 100%)'
                      : 'linear-gradient(135deg, #f6ffed 0%, #d9f7be 100%)',
                    border: `1px solid ${result.label === 'machine' ? '#ffa39e' : '#b7eb8f'}`,
                  }}
                >
                  <Row align="middle" gutter={16}>
                    <Col>
                      <div style={{
                        width: 56,
                        height: 56,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 28,
                        background: result.label === 'machine' ? '#ff4d4f' : '#52c41a',
                        color: '#fff',
                      }}>
                        {result.label === 'machine' ? <RobotOutlined /> : <UserOutlined />}
                      </div>
                    </Col>
                    <Col flex={1}>
                      <Typography.Title level={4} style={{
                        marginBottom: 0,
                        color: result.label === 'machine' ? '#cf1322' : '#389e0d',
                      }}>
                        {result.label === 'machine' ? 'Machine Generated' : 'Human Written'}
                      </Typography.Title>
                      <Typography.Text type="secondary">
                        Confidence: {confidencePercent}%
                      </Typography.Text>
                    </Col>
                    <Col>
                      <Progress
                        type="dashboard"
                        percent={aiProbPercent}
                        size={72}
                        strokeColor={result.label === 'machine' ? '#ff4d4f' : '#52c41a'}
                        format={() => `${aiProbPercent}%`}
                      />
                    </Col>
                  </Row>
                </div>

                {/* Metrics row */}
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col span={8}>
                    <Statistic
                      title="AI Probability"
                      value={result.ai_probability * 100}
                      precision={2}
                      suffix="%"
                      valueStyle={{
                        color: result.ai_probability > 0.5 ? '#cf1322' : '#389e0d',
                        fontSize: 20,
                      }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Confidence"
                      value={result.confidence * 100}
                      precision={2}
                      suffix="%"
                      valueStyle={{ fontSize: 20 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Threshold"
                      value={result.threshold}
                      precision={4}
                      valueStyle={{ fontSize: 20, color: '#8c8c8c' }}
                    />
                  </Col>
                </Row>

                <Row gutter={16} style={{ marginBottom: 8 }}>
                  <Col span={12}>
                    <Typography.Text style={{ fontSize: 12, color: '#595959' }}>
                      AI Probability Progress
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
                      Confidence Progress
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

                <Typography.Title level={5} style={{ marginBottom: 8, fontSize: 14 }}>
                  Interpretation
                </Typography.Title>
                <Typography.Paragraph style={{ marginBottom: 8, fontSize: 13, color: '#595959' }}>
                  {result.ai_probability >= result.threshold ? (
                    <>
                      The AI probability ({(result.ai_probability * 100).toFixed(2)}%) <strong>exceeds</strong> the threshold ({(result.threshold * 100).toFixed(2)}%),
                      indicating this text is likely <strong style={{ color: '#cf1322' }}>machine-generated</strong>.
                    </>
                  ) : (
                    <>
                      The AI probability ({(result.ai_probability * 100).toFixed(2)}%) is <strong>below</strong> the threshold ({(result.threshold * 100).toFixed(2)}%),
                      indicating this text is likely <strong style={{ color: '#389e0d' }}>human-written</strong>.
                    </>
                  )}
                </Typography.Paragraph>

                <Space wrap style={{ marginBottom: 12 }}>
                  <Tag color={result.label === 'machine' ? 'red' : 'green'} style={{ fontSize: 13, padding: '4px 10px' }}>
                    {result.label === 'machine' ? 'Machine Generated' : 'Human Written'}
                  </Tag>
                  <Tag color="blue" style={{ fontSize: 13, padding: '4px 10px' }}>
                    Threshold: {result.threshold.toFixed(4)}
                  </Tag>
                  <Tag color="purple" style={{ fontSize: 13, padding: '4px 10px' }}>
                    Confidence: {confidencePercent}%
                  </Tag>
                </Space>

                {detectorInfo && (
                  <Typography.Paragraph style={{ marginBottom: 0, fontSize: 12, color: '#8c8c8c' }}>
                    <strong>Detector:</strong> {detectorInfo.name} ({detectorInfo.type})
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
                Execution Logs
                {isDemoRunning && (
                  <Tag color="processing" style={{ marginLeft: 8, fontSize: 12 }}>
                    RUNNING
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
