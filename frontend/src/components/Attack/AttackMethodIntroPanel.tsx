import React from 'react';
import { Card, Collapse, Typography, Tag } from 'antd';
import { UILanguage } from '../../types';
import { getCoreText } from '../../i18n/coreText';
import { formatAttackLabel, getAttackMethodInfo } from './attackLabels';
import { AttackComparisonChart } from './AttackComparisonChart';
import { AttackExampleViewer } from './AttackExampleViewer';

interface AttackMethodIntroPanelProps {
  selectedAttacks: string[];
  attackMap: Map<string, any>;
  language: UILanguage;
}

export const AttackMethodIntroPanel: React.FC<AttackMethodIntroPanelProps> = ({
  selectedAttacks,
  attackMap,
  language,
}) => {
  const selectedAttackTypes = selectedAttacks
    .map((attackKey) => attackMap.get(attackKey)?.type)
    .filter((value): value is string => Boolean(value));

  const items = selectedAttacks
    .map((attackKey) => {
      const attack = attackMap.get(attackKey);
      if (!attack) {
        return null;
      }
      const info = getAttackMethodInfo(attack.type);
      const label = formatAttackLabel(attack.type, attack.backend, language);
      return {
        key: attackKey,
        label,
        attackType: attack.type, // Store attack type for example viewer
        children: (
          <div style={{ display: 'grid', gap: 8 }}>
            <Typography.Text>
              <strong>{getCoreText(language, 'attackIntroWhat')}:</strong>{' '}
              {info.whatItDoes[language]}
            </Typography.Text>
            <Typography.Text>
              <strong>{getCoreText(language, 'attackIntroWhen')}:</strong>{' '}
              {info.whenToUse[language]}
            </Typography.Text>
            <Typography.Text>
              <strong>{getCoreText(language, 'attackIntroRisk')}:</strong>{' '}
              {info.riskTradeoff[language]}
            </Typography.Text>
            <Typography.Text>
              <strong>{getCoreText(language, 'attackIntroCost')}:</strong>{' '}
              {info.computeCostHint[language]}
            </Typography.Text>
            {/* Add attack example viewer */}
            <AttackExampleViewer attackType={attack.type} language={language} />
          </div>
        ),
      };
    })
    .filter(Boolean) as Array<{ key: string; label: string; children: React.ReactNode }>;

  return (
    <Card
      size="small"
      title={getCoreText(language, 'attackIntroTitle')}
      style={{ marginBottom: 16 }}
    >
      <div style={{ marginBottom: 12 }}>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 8 }}>
          {getCoreText(language, 'attackCompareHint')}
        </Typography.Paragraph>
        <Tag color={selectedAttackTypes.length > 0 ? 'blue' : 'default'}>
          {selectedAttackTypes.length > 0
            ? getCoreText(language, 'attackCompareFilterSelected')
            : getCoreText(language, 'attackCompareFilterAll')}
        </Tag>
      </div>

      {/* Visual comparison charts */}
      <Collapse
        items={[
          {
            key: 'charts',
            label: `📊 ${getCoreText(language, 'attackCompareTitle')}`,
            children: (
              <div>
                <Typography.Title level={5}>
                  {getCoreText(language, 'attackCompareRadarTitle')}
                </Typography.Title>
                <AttackComparisonChart
                  language={language}
                  selectedAttackTypes={selectedAttackTypes}
                />
              </div>
            ),
          },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Attack method details */}
      {items.length > 0 ? (
        <Collapse items={items} />
      ) : (
        <Typography.Text type="secondary">{getCoreText(language, 'attackIntroEmpty')}</Typography.Text>
      )}
    </Card>
  );
};
