/**
 * Attack Cost Bar Chart
 * Cost and method type comparison across selected attack methods.
 */

import React, { useMemo, useState } from 'react';
import { Bar } from '@ant-design/plots';
import { Select, Space, Typography } from 'antd';
import type { UILanguage } from '../../types';
import { getCoreText } from '../../i18n/coreText';
import { ATTACK_COLOR_MAP, ATTACK_METHOD_METRICS, formatAttackLabel } from './attackLabels';

interface AttackCostChartProps {
  language: UILanguage;
  selectedAttackTypes?: string[];
}

type SortMode = 'cost' | 'name';

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

export const AttackCostChart: React.FC<AttackCostChartProps> = ({
  language,
  selectedAttackTypes = [],
}) => {
  const [sortMode, setSortMode] = useState<SortMode>('cost');

  const data = useMemo(() => {
    const selectedSet = new Set(selectedAttackTypes.map((t) => t.toLowerCase()));
    const attackTypes =
      selectedSet.size > 0
        ? ORDERED_ATTACK_TYPES.filter((type) => selectedSet.has(type))
        : ORDERED_ATTACK_TYPES;

    const rows = attackTypes
      .map((attackType) => {
        const metric = ATTACK_METHOD_METRICS[attackType];
        if (!metric) {
          return null;
        }
        const tierText =
          language === 'zh'
            ? metric.costTier === 'low'
              ? '低成本'
              : metric.costTier === 'medium'
                ? '中等成本'
                : '高成本'
            : metric.costTier === 'low'
              ? 'Low Cost'
              : metric.costTier === 'medium'
                ? 'Medium Cost'
                : 'High Cost';
        const methodTypeText =
          language === 'zh'
            ? metric.methodType === 'rule'
              ? '规则法'
              : metric.methodType === 'model'
                ? '模型法'
                : metric.methodType === 'api'
                  ? 'API法'
                  : '混合法'
            : metric.methodType === 'rule'
              ? 'Rule-based'
              : metric.methodType === 'model'
                ? 'Model-based'
                : metric.methodType === 'api'
                  ? 'API-based'
                  : 'Hybrid';

        return {
          attack: formatAttackLabel(attackType, undefined, language),
          attackType,
          cost: metric.costLevel,
          tierText,
          methodTypeText,
          legend: `${tierText} · ${methodTypeText}`,
        };
      })
      .filter(Boolean) as Array<{
      attack: string;
      attackType: string;
      cost: number;
      tierText: string;
      methodTypeText: string;
      legend: string;
    }>;

    if (sortMode === 'cost') {
      rows.sort((a, b) => a.cost - b.cost || a.attack.localeCompare(b.attack));
    } else {
      rows.sort((a, b) => a.attack.localeCompare(b.attack));
    }
    return rows;
  }, [language, selectedAttackTypes, sortMode]);

  const config = {
    data,
    xField: 'cost',
    yField: 'attack',
    seriesField: 'legend',
    color: ({ attackType }: { attackType: string }) => ATTACK_COLOR_MAP[attackType] || '#1677ff',
    barStyle: {
      radius: [0, 6, 6, 0] as [number, number, number, number],
    },
    label: {
      position: 'right' as const,
      content: (item: any) => {
        const cost = item.cost ?? item.data?.cost ?? 0;
        return `${cost}/10`;
      },
      style: {
        fill: '#262626',
        fontSize: 13,
        fontWeight: 600,
      },
    },
    xAxis: {
      min: 0,
      max: 10,
      label: {
        formatter: (v: string) => (language === 'zh' ? `${v}级` : `Level ${v}`),
        style: {
          fontSize: 12,
        },
      },
      grid: {
        line: {
          style: {
            stroke: '#f0f0f0',
            lineWidth: 1,
          },
        },
      },
    },
    yAxis: {
      label: {
        style: {
          fontSize: 13,
          fontWeight: 500,
        },
      },
    },
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: `${datum.attack}`,
          value: `${datum.legend} · ${language === 'zh' ? '评分' : 'Score'} ${datum.cost}/10`,
        };
      },
    },
    legend: {
      position: 'top' as const,
      offsetY: -5,
      itemName: {
        style: {
          fontSize: 13,
          fontWeight: 500,
        },
      },
      marker: {
        symbol: 'square',
        style: {
          r: 4,
        },
      },
    },
    height: 420,
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={8}>
      <Space align="center">
        <Typography.Text type="secondary">{getCoreText(language, 'attackCostSortLabel')}:</Typography.Text>
        <Select<SortMode>
          value={sortMode}
          onChange={setSortMode}
          style={{ width: 140 }}
          options={[
            { value: 'cost', label: getCoreText(language, 'attackCostSortCost') },
            { value: 'name', label: getCoreText(language, 'attackCostSortName') },
          ]}
        />
      </Space>
      <Bar {...config} />
    </Space>
  );
};

