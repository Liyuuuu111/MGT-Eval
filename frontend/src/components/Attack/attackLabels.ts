export const ATTACK_TYPE_LABELS: Record<string, string> = {
  span: 'Span Perturbation',
  para: 'Paraphrasing',
  typo: 'Typo Mixed',
  inse: 'Typo Insertion',
  dele: 'Typo Deletion',
  subs: 'Typo Substitution',
  tran: 'Typo Transposition',
  homo: 'Homoglyph Alteration',
  form: 'Format Character Editing',
  syno: 'Synonym Substitution',
  back_trans: 'Back Translation',
  humanize: 'Humanize',
};

export const formatAttackLabel = (attackType?: string, backend?: string): string => {
  const key = (attackType || '').toLowerCase();
  const base = ATTACK_TYPE_LABELS[key] || (attackType || '').toUpperCase();
  if (backend) {
    return `${base} (${backend})`;
  }
  return base;
};
