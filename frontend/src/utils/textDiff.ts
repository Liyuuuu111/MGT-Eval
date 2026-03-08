export type DiffMode = 'token' | 'char' | 'auto' | 'invisible';

export interface DiffSegment {
  type: 'equal' | 'add' | 'del';
  text: string;
}

const INVISIBLE_CHAR_LABELS: Record<string, string> = {
  '\u200b': '⟪ZWSP⟫',
  '\u200c': '⟪ZWNJ⟫',
  '\u200d': '⟪ZWJ⟫',
  '\ufeff': '⟪BOM⟫',
  '\u2060': '⟪WJ⟫',
};

const hasHan = (text: string): boolean => /[\u4e00-\u9fff]/.test(text);

const visualizeInvisibleChars = (text: string): string => {
  let result = '';
  for (const ch of Array.from(text)) {
    result += INVISIBLE_CHAR_LABELS[ch] ?? ch;
  }
  return result;
};

const tokenize = (text: string, mode: Exclude<DiffMode, 'auto'>): string[] => {
  if (!text) {
    return [];
  }
  if (mode === 'char' || mode === 'invisible') {
    return Array.from(text);
  }
  return text.match(/\s+|[^\s]+/g) ?? [];
};

const resolveMode = (original: string, attacked: string, mode: DiffMode): Exclude<DiffMode, 'auto'> => {
  if (mode !== 'auto') {
    return mode;
  }
  return hasHan(original) || hasHan(attacked) ? 'char' : 'token';
};

const buildLcsTable = (a: string[], b: string[]): number[][] => {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    Array.from({ length: b.length + 1 }, () => 0),
  );
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      if (a[i] === b[j]) {
        dp[i][j] = dp[i + 1][j + 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }
  return dp;
};

const pushSegment = (segments: DiffSegment[], type: DiffSegment['type'], text: string): void => {
  if (!text) {
    return;
  }
  const last = segments[segments.length - 1];
  if (last && last.type === type) {
    last.text += text;
    return;
  }
  segments.push({ type, text });
};

export const buildDiffSegments = (
  originalInput: string,
  attackedInput: string,
  mode: DiffMode = 'auto',
): DiffSegment[] => {
  const effectiveMode = resolveMode(originalInput, attackedInput, mode);
  const original = effectiveMode === 'invisible' ? visualizeInvisibleChars(originalInput) : originalInput;
  const attacked = effectiveMode === 'invisible' ? visualizeInvisibleChars(attackedInput) : attackedInput;

  const a = tokenize(original, effectiveMode);
  const b = tokenize(attacked, effectiveMode);
  const dp = buildLcsTable(a, b);

  const segments: DiffSegment[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      pushSegment(segments, 'equal', a[i]);
      i += 1;
      j += 1;
      continue;
    }
    if (dp[i + 1][j] >= dp[i][j + 1]) {
      pushSegment(segments, 'del', a[i]);
      i += 1;
    } else {
      pushSegment(segments, 'add', b[j]);
      j += 1;
    }
  }

  while (i < a.length) {
    pushSegment(segments, 'del', a[i]);
    i += 1;
  }
  while (j < b.length) {
    pushSegment(segments, 'add', b[j]);
    j += 1;
  }

  return segments;
};

