/**
 * Main App Component
 */

import React, { useEffect, useState } from 'react';
import { Button, Layout, Menu, Typography, message, Tag, Space } from 'antd';
import {
  BuildOutlined,
  BugOutlined,
  ExperimentOutlined,
  EyeOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import { BuildSection } from './components/Build/BuildSection';
import { AttackSection } from './components/Attack/AttackSection';
import { TrainSection } from './components/Train/TrainSection';
import { DetectSection } from './components/Detect/DetectSection';
import { DemoSection } from './components/Demo/DemoSection';
import { SystemMonitorPanel } from './components/Shared/SystemMonitorPanel';
import { useStore } from './store';
import api from './services/api';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

type SectionKey = 'build' | 'attack' | 'train' | 'detect' | 'demo';

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<SectionKey>('build');
  const [backendStatus, setBackendStatus] = useState<{
    connected: boolean | null;
    latencyMs: number | null;
  }>({ connected: null, latencyMs: null });
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
  ];

  const hasRunningJob = runningJobs.some((job) => job.isRunning && job.jobId);

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
    const intervalId = window.setInterval(pingBackend, 5000);
    return () => {
      mounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const handleStopRunning = async () => {
    const activeJob = runningJobs.find(
      (job) => job.section === activeSection && job.isRunning && job.jobId
    );
    const jobsToCancel = activeJob
      ? [activeJob]
      : runningJobs.filter((job) => job.isRunning && job.jobId);
    if (jobsToCancel.length === 0) {
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
  };

  const menuItems = [
    { key: 'build', icon: <BuildOutlined />, label: 'Build Dataset' },
    { key: 'attack', icon: <BugOutlined />, label: 'Attack Dataset' },
    { key: 'train', icon: <ExperimentOutlined />, label: 'Train Detector' },
    { key: 'detect', icon: <EyeOutlined />, label: 'Detect' },
    { key: 'demo', icon: <PlayCircleOutlined />, label: 'Demo' },
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
          background: '#001529',
          padding: '0 24px',
          position: 'fixed',
          width: '100%',
          zIndex: 1000,
          top: 0
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '100%' }}>
          <Title level={3} style={{ color: 'white', margin: 0 }}>
            MGT Eval - Machine Generated Text Evaluation
          </Title>
          <Space size="middle">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Tag color={backendStatus.connected === null ? 'default' : backendStatus.connected ? 'green' : 'red'}>
                {backendStatus.connected === null
                  ? 'Checking...'
                  : backendStatus.connected
                    ? 'Backend Connected'
                    : 'Backend Disconnected'}
              </Tag>
              <span style={{ color: '#fff', fontSize: 12 }}>
                Latency: {backendStatus.connected ? `${backendStatus.latencyMs ?? '-'} ms` : '-'}
              </span>
            </div>
            <Button danger disabled={!hasRunningJob} onClick={handleStopRunning}>
              Stop Current Job
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
          <SystemMonitorPanel />
          {renderSection()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
