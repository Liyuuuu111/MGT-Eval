/**
 * Attack Configuration Editor
 * Displays configuration fields for selected attacks
 */

import React from 'react';
import { Collapse, Divider } from 'antd';
import { DynamicFormFields } from '../Shared/DynamicFormFields';
import { formatAttackLabel } from './attackLabels';

const { Panel } = Collapse;

interface AttackConfigEditorProps {
  selectedAttacks: string[]; // attack keys (type or type-backend)
  allAttacks: any;
  attackMap: Map<string, any>; // Map from key to attack config
}

export const AttackConfigEditor: React.FC<AttackConfigEditorProps> = ({
  selectedAttacks,
  allAttacks,
  attackMap
}) => {
  if (!selectedAttacks || selectedAttacks.length === 0) {
    return null;
  }

  if (!allAttacks?.text_attacks) {
    return null;
  }

  return (
    <div>
      <Divider orientation="left">Attack Parameters</Divider>

      <Collapse defaultActiveKey={selectedAttacks}>
        {selectedAttacks.map((attackKey) => {
          const attack = attackMap.get(attackKey);

          if (!attack) {
            return null;
          }

          // Extract attack config (exclude 'type' and 'backend')
          const { type, backend, ...attackConfig } = attack;

          // Generate display name
          const displayName = formatAttackLabel(type, backend);

          return (
            <Panel
              header={`${displayName} Configuration`}
              key={attackKey}
            >
              <DynamicFormFields
                data={attackConfig}
                prefix={['attack_configs', attackKey]}
                excludeKeys={['type', 'backend']}
              />
            </Panel>
          );
        })}
      </Collapse>
    </div>
  );
};
