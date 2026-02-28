/**
 * Main App Component
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Layout, Menu, Typography, message, Tag, Space } from 'antd';
import {
  AimOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  RocketOutlined,
  StopOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { BuildSection } from './components/Build/BuildSection';
import { AttackSection } from './components/Attack/AttackSection';
import { TrainSection } from './components/Train/TrainSection';
import { DetectSection } from './components/Detect/DetectSection';
import { DemoSection } from './components/Demo/DemoSection';
import { SystemMonitorPanel } from './components/Shared/SystemMonitorPanel';
import { LanguageSwitcher } from './components/Shared/LanguageSwitcher';
import { useUILanguage } from './hooks/useUILanguage';
import { useStore } from './store';
import api from './services/api';
import { dispatchStartSection } from './constants/jobControls';
import { Section } from './types';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

type SectionKey = Section;

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<SectionKey>('demo');
  const { t, language } = useUILanguage();
  const [backendStatus, setBackendStatus] = useState<{
    connected: boolean | null;
    latencyMs: number | null;
  }>({ connected: null, latencyMs: null });
  const [isStopping, setIsStopping] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const startFlashTimerRef = useRef<number | null>(null);
  const {
    buildJobId,
    isBuildRunning,
    stopBuild,
    addBuildLog,
    attackJobId,
    isAttackRunning,
    stopAttack,
    addAttackLog,
    trainJobId,
    isTrainRunning,
    stopTrain,
    addTrainLog,
    detectJobId,
    isDetectRunning,
    stopDetect,
    addDetectLog,
    demoJobId,
    isDemoRunning,
    stopDemo,
    addDemoLog,
  } = useStore((state) => ({
    buildJobId: state.buildJobId,
    isBuildRunning: state.isBuildRunning,
    stopBuild: state.stopBuild,
    addBuildLog: state.addBuildLog,
    attackJobId: state.attackJobId,
    isAttackRunning: state.isAttackRunning,
    stopAttack: state.stopAttack,
    addAttackLog: state.addAttackLog,
    trainJobId: state.trainJobId,
    isTrainRunning: state.isTrainRunning,
    stopTrain: state.stopTrain,
    addTrainLog: state.addTrainLog,
    detectJobId: state.detectJobId,
    isDetectRunning: state.isDetectRunning,
    stopDetect: state.stopDetect,
    addDetectLog: state.addDetectLog,
    demoJobId: state.demoJobId,
    isDemoRunning: state.isDemoRunning,
    stopDemo: state.stopDemo,
    addDemoLog: state.addDemoLog,
  }));

  const runningJobs = [
    {
      section: 'build' as SectionKey,
      jobId: buildJobId,
      isRunning: isBuildRunning,
      stop: stopBuild,
      addLog: addBuildLog,
    },
    {
      section: 'attack' as SectionKey,
      jobId: attackJobId,
      isRunning: isAttackRunning,
      stop: stopAttack,
      addLog: addAttackLog,
    },
    {
      section: 'train' as SectionKey,
      jobId: trainJobId,
      isRunning: isTrainRunning,
      stop: stopTrain,
      addLog: addTrainLog,
    },
    {
      section: 'detect' as SectionKey,
      jobId: detectJobId,
      isRunning: isDetectRunning,
      stop: stopDetect,
      addLog: addDetectLog,
    },
    {
      section: 'demo' as SectionKey,
      jobId: demoJobId,
      isRunning: isDemoRunning,
      stop: stopDemo,
      addLog: addDemoLog,
    },
  ];

  const hasRunningJob = runningJobs.some((job) => job.isRunning && job.jobId);
  const activeSectionJob = runningJobs.find((job) => job.section === activeSection);
  const isActiveSectionRunning = Boolean(activeSectionJob?.isRunning && activeSectionJob?.jobId);

  const startButtonText = useMemo(() => {
    if (language === 'zh') {
      const zhMap: Record<SectionKey, string> = {
        demo: '开始演示检测',
        build: '开始构建数据集',
        attack: '开始攻击数据集',
        train: '开始训练检测器',
        detect: '开始评估',
      };
      return zhMap[activeSection];
    }
    const enMap: Record<SectionKey, string> = {
      demo: 'Start Demo Detection',
      build: 'Start Build',
      attack: 'Start Attack',
      train: 'Start Training',
      detect: 'Start Evaluation',
    };
    return enMap[activeSection];
  }, [activeSection, language]);

  useEffect(() => {
    let mounted = true;

    const pingBackend = async () => {
      const start = performance.now();
      try {
        await api.getHealth();
        const latency = Math.round(performance.now() - start);
        if (mounted) {
          setBackendStatus({ connected: true, latencyMs: latency });
        }
      } catch (error) {
        if (mounted) {
          setBackendStatus({ connected: false, latencyMs: null });
        }
      }
    };

    pingBackend();
    const intervalId = window.setInterval(pingBackend, 10000);
    return () => {
      mounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (startFlashTimerRef.current !== null) {
        window.clearTimeout(startFlashTimerRef.current);
      }
    };
  }, []);

  const handleStartCurrent = () => {
    if (isActiveSectionRunning) {
      return;
    }
    dispatchStartSection(activeSection);
    setIsStarting(true);
    if (startFlashTimerRef.current !== null) {
      window.clearTimeout(startFlashTimerRef.current);
    }
    startFlashTimerRef.current = window.setTimeout(() => {
      setIsStarting(false);
      startFlashTimerRef.current = null;
    }, 700);
  };

  const handleStopRunning = async () => {
    setIsStopping(true);
    const activeJob = runningJobs.find(
      (job) => job.section === activeSection && job.isRunning && job.jobId
    );
    const jobsToCancel = activeJob
      ? [activeJob]
      : runningJobs.filter((job) => job.isRunning && job.jobId);
    if (jobsToCancel.length === 0) {
      setIsStopping(false);
      return;
    }

    const results = await Promise.allSettled(
      jobsToCancel.map((job) => api.cancelJob(job.jobId as string))
    );

    let failed = 0;
    results.forEach((result, idx) => {
      const job = jobsToCancel[idx];
      if (result.status === 'fulfilled') {
        job.addLog({
          level: 'warning',
          message: 'Cancellation requested',
          timestamp: new Date().toISOString(),
        });
        job.stop();
      } else {
        failed += 1;
      }
    });

    if (failed > 0) {
      message.error('Failed to cancel one or more jobs');
    } else {
      message.info('Cancellation requested');
    }
    setIsStopping(false);
  };

  const menuItems = [
    { key: 'demo', icon: <RocketOutlined />, label: t('menuDemo') },
    { key: 'build', icon: <DatabaseOutlined />, label: t('menuBuild') },
    { key: 'attack', icon: <ThunderboltOutlined />, label: t('menuAttack') },
    { key: 'train', icon: <ExperimentOutlined />, label: t('menuTrain') },
    { key: 'detect', icon: <AimOutlined />, label: t('menuDetect') },
  ];

  const renderSection = () => {
    switch (activeSection) {
      case 'build':
        return <BuildSection />;
      case 'attack':
        return <AttackSection />;
      case 'train':
        return <TrainSection />;
      case 'detect':
        return <DetectSection />;
      case 'demo':
        return <DemoSection />;
      default:
        return null;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
          padding: '0 24px',
          position: 'fixed',
          width: '100%',
          zIndex: 1000,
          top: 0,
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '100%' }}>
          <div style={{ minWidth: 0, lineHeight: 1.2 }}>
            <Title level={3} style={{ color: 'white', margin: 0, lineHeight: 1.2 }}>
              {t('appTitle')}
            </Title>
            <span style={{ color: 'rgba(255,255,255,0.75)', fontSize: 12, fontWeight: 400 }}>
              {t('appSubtitle')}
            </span>
          </div>
          <Space size="middle" style={{ flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Tag color={backendStatus.connected === null ? 'default' : backendStatus.connected ? 'green' : 'red'}>
                {backendStatus.connected === null
                  ? t('appChecking')
                  : backendStatus.connected
                    ? t('appBackendConnected')
                    : t('appBackendDisconnected')}
              </Tag>
              <span style={{ color: '#fff', fontSize: 12 }}>
                {t('appLatency')}: {backendStatus.connected ? `${backendStatus.latencyMs ?? '-'} ms` : '-'}
              </span>
            </div>
            <LanguageSwitcher />
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              loading={isStarting}
              disabled={isActiveSectionRunning}
              onClick={handleStartCurrent}
              style={{
                fontWeight: 700,
                borderWidth: 1,
                borderColor: '#95de64',
                background: isActiveSectionRunning
                  ? '#8c8c8c'
                  : 'linear-gradient(135deg, #52c41a 0%, #389e0d 100%)',
                color: '#fff',
                boxShadow: isStarting
                  ? '0 0 0 3px rgba(82,196,26,0.35)'
                  : '0 2px 6px rgba(0,0,0,0.22)',
              }}
            >
              {startButtonText}
            </Button>
            <Button
              danger
              type={hasRunningJob ? 'primary' : 'default'}
              icon={<StopOutlined />}
              loading={isStopping}
              disabled={!hasRunningJob}
              onClick={handleStopRunning}
              style={{
                fontWeight: 700,
                borderWidth: 1,
                background: hasRunningJob
                  ? 'linear-gradient(135deg, #ff4d4f 0%, #cf1322 100%)'
                  : undefined,
                color: hasRunningJob ? '#fff' : undefined,
                boxShadow: hasRunningJob
                  ? '0 0 0 3px rgba(255,77,79,0.28)'
                  : 'none',
              }}
            >
              {isStopping
                ? (language === 'zh' ? '停止中...' : 'Stopping...')
                : t('appStopCurrentJob')}
            </Button>
          </Space>
        </div>
      </Header>
      <Layout style={{ marginTop: '64px' }}>
        <Sider
          width={250}
          theme="light"
          style={{
            position: 'fixed',
            left: 0,
            top: '64px',
            bottom: 0,
            overflowY: 'auto',
            height: 'calc(100vh - 64px)',
            zIndex: 999
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[activeSection]}
            items={menuItems}
            onClick={({ key }) => setActiveSection(key as SectionKey)}
            style={{ height: '100%', borderRight: 0 }}
          />
        </Sider>
        <Content style={{ padding: '24px', background: '#f0f2f5', marginLeft: '250px' }}>
          {/* <SystemMonitorPanel /> */}
          {renderSection()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
