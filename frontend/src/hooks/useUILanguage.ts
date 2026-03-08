import { useMemo } from 'react';
import { useStore } from '../store';
import { CoreTextKey, getCoreText } from '../i18n/coreText';

export const useUILanguage = () => {
  const language = useStore((state) => state.uiLanguage);
  const setLanguage = useStore((state) => state.setUiLanguage);

  const t = useMemo(() => {
    return (key: CoreTextKey) => getCoreText(language, key);
  }, [language]);

  return { language, setLanguage, t };
};

