import { Section } from '../types';

export const START_SECTION_EVENT = 'mgt-eval:start-section';

export interface StartSectionEventDetail {
  section: Section;
}

export const dispatchStartSection = (section: Section): void => {
  window.dispatchEvent(
    new CustomEvent<StartSectionEventDetail>(START_SECTION_EVENT, {
      detail: { section },
    }),
  );
};

