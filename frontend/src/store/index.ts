/**
 * Zustand store for state management
 */

import { create } from 'zustand';
import { LogEntry, UILanguage } from '../types';

const getDefaultUILanguage = (): UILanguage => {
  if (typeof navigator !== 'undefined') {
    const lang = (navigator.language || '').toLowerCase();
    if (lang.startsWith('zh')) {
      return 'zh';
    }
  }
  return 'en';
};

interface AppState {
  // Build state
  buildLogs: LogEntry[];
  buildJobId: string | null;
  isBuildRunning: boolean;

  // Attack state
  attackLogs: LogEntry[];
  attackJobId: string | null;
  isAttackRunning: boolean;

  // Train state
  trainLogs: LogEntry[];
  trainJobId: string | null;
  isTrainRunning: boolean;
  trainResult: any | null;

  // Detect state
  detectLogs: LogEntry[];
  detectJobId: string | null;
  isDetectRunning: boolean;
  detectResult: any | null;

  // Demo state
  demoLogs: LogEntry[];
  demoJobId: string | null;
  isDemoRunning: boolean;
  demoResult: any | null;

  // Shared settings state
  hfToken: string;
  uiLanguage: UILanguage;

  // Actions
  addBuildLog: (log: LogEntry) => void;
  clearBuildLogs: () => void;
  startBuild: (jobId: string) => void;
  stopBuild: () => void;

  addAttackLog: (log: LogEntry) => void;
  clearAttackLogs: () => void;
  startAttack: (jobId: string) => void;
  stopAttack: () => void;

  addTrainLog: (log: LogEntry) => void;
  clearTrainLogs: () => void;
  startTrain: (jobId: string) => void;
  stopTrain: () => void;
  setTrainResult: (result: any | null) => void;

  addDetectLog: (log: LogEntry) => void;
  clearDetectLogs: () => void;
  startDetect: (jobId: string) => void;
  stopDetect: () => void;
  setDetectResult: (result: any | null) => void;

  addDemoLog: (log: LogEntry) => void;
  clearDemoLogs: () => void;
  startDemo: (jobId: string) => void;
  stopDemo: () => void;
  setDemoResult: (result: any | null) => void;

  setHfToken: (token: string) => void;
  clearHfToken: () => void;
  setUiLanguage: (lang: UILanguage) => void;
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  buildLogs: [],
  buildJobId: null,
  isBuildRunning: false,

  attackLogs: [],
  attackJobId: null,
  isAttackRunning: false,

  trainLogs: [],
  trainJobId: null,
  isTrainRunning: false,
  trainResult: null,

  detectLogs: [],
  detectJobId: null,
  isDetectRunning: false,
  detectResult: null,

  demoLogs: [],
  demoJobId: null,
  isDemoRunning: false,
  demoResult: null,

  hfToken: '',
  uiLanguage: getDefaultUILanguage(),

  // Build actions
  addBuildLog: (log) => set((state) => ({
    buildLogs: [...state.buildLogs, log],
  })),
  clearBuildLogs: () => set({ buildLogs: [] }),
  startBuild: (jobId) => set({ buildJobId: jobId, isBuildRunning: true }),
  stopBuild: () => set({ isBuildRunning: false }),

  // Attack actions
  addAttackLog: (log) => set((state) => ({
    attackLogs: [...state.attackLogs, log],
  })),
  clearAttackLogs: () => set({ attackLogs: [] }),
  startAttack: (jobId) => set({ attackJobId: jobId, isAttackRunning: true }),
  stopAttack: () => set({ isAttackRunning: false }),

  // Train actions
  addTrainLog: (log) => set((state) => ({
    trainLogs: [...state.trainLogs, log],
  })),
  clearTrainLogs: () => set({ trainLogs: [] }),
  startTrain: (jobId) => set({ trainJobId: jobId, isTrainRunning: true, trainResult: null }),
  stopTrain: () => set({ isTrainRunning: false }),
  setTrainResult: (result) => set({ trainResult: result }),

  // Detect actions
  addDetectLog: (log) => set((state) => ({
    detectLogs: [...state.detectLogs, log],
  })),
  clearDetectLogs: () => set({ detectLogs: [] }),
  startDetect: (jobId) => set({ detectJobId: jobId, isDetectRunning: true, detectResult: null }),
  stopDetect: () => set({ isDetectRunning: false }),
  setDetectResult: (result) => set({ detectResult: result }),

  // Demo actions
  addDemoLog: (log) => set((state) => ({
    demoLogs: [...state.demoLogs, log],
  })),
  clearDemoLogs: () => set({ demoLogs: [] }),
  startDemo: (jobId) => set({ demoJobId: jobId, isDemoRunning: true, demoResult: null }),
  stopDemo: () => set({ isDemoRunning: false }),
  setDemoResult: (result) => set({ demoResult: result }),

  setHfToken: (token) => set({ hfToken: token }),
  clearHfToken: () => set({ hfToken: '' }),
  setUiLanguage: (lang) => set({ uiLanguage: lang }),
}));
