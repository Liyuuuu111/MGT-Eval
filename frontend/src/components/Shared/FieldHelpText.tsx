import React, { useMemo } from 'react';
import { Typography } from 'antd';
import { getCoreText } from '../../i18n/coreText';
import { useUILanguage } from '../../hooks/useUILanguage';
import { getFieldHelp, shouldShowHigherLower } from '../../utils/fieldHelp';

interface FieldHelpTextProps {
  path: string;
  value: unknown;
  context?: Record<string, unknown>;
}

export const FieldHelpText: React.FC<FieldHelpTextProps> = ({ path, value, context }) => {
  const { language } = useUILanguage();
  const help = useMemo(() => getFieldHelp(path, value, context, language), [path, value, context, language]);
  const showHigherLower = useMemo(() => shouldShowHigherLower(path, value), [path, value]);

  return (
    <div style={{ marginTop: 2, lineHeight: 1.35 }}>
      <Typography.Text type="secondary" style={{ fontSize: 11, display: 'block' }}>
        {getCoreText(language, 'fieldHelpPurpose')}: {help.purpose}
      </Typography.Text>
      {showHigherLower && (
        <>
          <Typography.Text type="secondary" style={{ fontSize: 11, display: 'block' }}>
            {getCoreText(language, 'fieldHelpHigher')}: {help.higher}
          </Typography.Text>
          <Typography.Text type="secondary" style={{ fontSize: 11, display: 'block' }}>
            {getCoreText(language, 'fieldHelpLower')}: {help.lower}
          </Typography.Text>
        </>
      )}
    </div>
  );
};
