/**
 * Attack Comparison Radar Chart
 * Multi-dimensional method-level comparison.
 */

import React, { useMemo } from 'react';
import { Radar } from '@ant-design/plots';
import type { UILanguage } from '../../types';
import { ATTACK_COLOR_MAP, ATTACK_METHOD_METRICS, formatAttackLabel } from './attackLabels';

interface AttackComparisonChartProps {
  language: UILanguage;
  selectedAttackTypes?: string[];
}

const DIMENSIONS = {
  en: {
    computeCost: { label: 'Compute Cost', tip: 'Compute and runtime overhead of the attack method.' },
    semanticPreservation: { label: 'Semantic Preservation', tip: 'How well original meaning is retained.' },
    fluency: { label: 'Fluency', tip: 'How natural the attacked text reads.' },
    stealth: { label: 'Stealth', tip: 'How hard the perturbation is to notice.' },
    attackPower: { label: 'Attack Power', tip: 'Expected impact on detector robustness.' },
  },
  zh: {
    computeCost: { label: '计算成本', tip: '方法带来的计算开销与运行成本。' },
    semanticPreservation: { label: '语义保持', tip: '攻击后文本保持原语义的程度。' },
    fluency: { label: '流畅度', tip: '攻击后文本的人类可读性与自然度。' },
    stealth: { label: '隐蔽性', tip: '扰动被人工或规则发现的难易程度。' },
    attackPower: { label: '攻击强度', tip: '对检测器造成性能下降的潜在能力。' },
  },
} as const;

const ORDERED_ATTACK_TYPES = [
  'typo',
  'inse',
  'dele',
  'subs',
  'tran',
  'homo',
  'form',
  'syno',
  'span',
  'para',
  'back_trans',
  'humanize',
];

export const AttackComparisonChart: React.FC<AttackComparisonChartProps> = ({
  language,
  selectedAttackTypes = [],
}) => {
  const { data, colorMap } = useMemo(() => {
    const dimMap = DIMENSIONS[language];
    const selectedSet = new Set(selectedAttackTypes.map((t) => t.toLowerCase()));
    const attackTypes =
      selectedSet.size > 0
        ? ORDERED_ATTACK_TYPES.filter((type) => selectedSet.has(type))
        : ORDERED_ATTACK_TYPES;

    const radarData: Array<{
      attack: string;
      attackType: string;
      dimension: string;
      dimensionTip: string;
      score: number;
    }> = [];

    const localizedColorMap: Record<string, string> = {};
    for (const attackType of attackTypes) {
      const metrics = ATTACK_METHOD_METRICS[attackType];
      if (!metrics) {
        continue;
      }
      const label = formatAttackLabel(attackType, undefined, language);
      localizedColorMap[label] = ATTACK_COLOR_MAP[attackType] || '#1677ff';
      for (const key of Object.keys(dimMap) as Array<keyof typeof dimMap>) {
        radarData.push({
          attack: label,
          attackType,
          dimension: dimMap[key].label,
          dimensionTip: dimMap[key].tip,
          score: metrics.scores[key],
        });
      }
    }

    return { data: radarData, colorMap: localizedColorMap };
  }, [language, selectedAttackTypes]);

  const config = {
    data,
    xField: 'dimension',
    yField: 'score',
    seriesField: 'attack',
    color: ({ attack }: { attack: string }) => colorMap[attack] || '#1677ff',
    meta: {
      score: {
        alias: language === 'zh' ? '得分' : 'Score',
        min: 0,
        max: 10,
      },
    },
    xAxis: {
      line: null,
      tickLine: null,
      grid: {
        line: {
          style: {
            lineDash: null,
          },
        },
      },
      label: {
        style: {
          fontSize: 13,
          fontWeight: 500,
        },
      },
    },
    yAxis: {
      line: null,
      tickLine: null,
      grid: {
        line: {
          type: 'line',
          style: {
            lineDash: null,
            stroke: '#d9d9d9',
          },
        },
      },
    },
    tooltip: {
      fields: ['attack', 'dimension', 'score', 'dimensionTip'],
      formatter: (datum: any) => {
        return {
          name: `${datum.attack} · ${datum.dimension}`,
          value: `${datum.score}/10${language === 'zh' ? `（${datum.dimensionTip}）` : ` (${datum.dimensionTip})`}`,
        };
      },
    },
    point: {
      size: 3.5,
      shape: 'circle',
      style: ({ attack }: { attack: string }) => ({
        fill: colorMap[attack] || '#1677ff',
        stroke: '#fff',
        lineWidth: 1.5,
      }),
    },
    lineStyle: {
      lineWidth: 2.5,
    },
    area: {
      style: {
        fillOpacity: 0.15,
      },
    },
    legend: {
      position: 'bottom' as const,
      offsetY: 8,
      itemName: {
        style: {
          fontSize: 13,
          fontWeight: 500,
        },
      },
      marker: {
        symbol: 'circle',
        style: {
          r: 5,
        },
      },
    },
    height: 480,
  };

  return <Radar {...config} />;
};

