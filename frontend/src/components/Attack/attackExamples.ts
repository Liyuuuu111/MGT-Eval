/**
 * Curated attack examples used in AttackMethodIntroPanel.
 * These are illustrative samples for UI explanation only.
 */

import type { UILanguage } from '../../types';
import type { DiffMode } from '../../utils/textDiff';

export interface AttackExample {
  attackType: string;
  language: UILanguage;
  title?: string;
  original: string;
  attacked: string;
  notes?: string;
  diffMode?: DiffMode;
  parameters?: Record<string, unknown>;
}

const ORIGINAL_EN =
  'Genshin Impact versions 6.0 to 6.3 continue the Archon Quest in Nod-Krai. The Traveler investigates moonlight anomalies, discovers a hidden Fatui lab, and finds experiments that manipulate perception at night.';

const ORIGINAL_ZH =
  '这句话通过强烈对比制造反讽张力。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份创伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同。';

export const ATTACK_EXAMPLES: AttackExample[] = [
  {
    attackType: 'typo',
    language: 'en',
    title: 'Typo (mixed)',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impcat versinos 6.0 to 6.3 continnue the Archon Quets in Nod-Krai. The Traveelr investgiates moonlgiht anomlaies, discovers a hiddne Fatui lab, and fndis experiements that manipualte perecption at nihgt.',
    notes: 'Mixed insert/delete/substitute/transpose noise keeps semantics roughly intact but adds realistic spelling corruption.',
    diffMode: 'token',
    parameters: { mode: 'mix', pct_words_masked: 0.2 },
  },
  {
    attackType: 'typo',
    language: 'zh',
    title: '拼写扰动（混合）',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制苯反讽张立。"悬梁五战终上北大"以传统苦读意象塑造造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦绿。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制朔造的身份仓伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判判，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同。',
    notes: '通过少量错字和字形混淆制造噪声，不改变主干叙事。',
    diffMode: 'char',
    parameters: { mode: 'mix', pct_words_masked: 0.18 },
  },
  {
    attackType: 'inse',
    language: 'en',
    title: 'Insertion-only typo',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impacct versions 6.0 to 6.3 continue the Archon Quuest in Nod-Krai. The Travelerr investigates moonlight anomaliees, discovers a hidden Fatui laab, and finds experiments that manipulate perceptiion at night.',
    notes: 'Character insertions imitate extra keystrokes.',
    diffMode: 'token',
    parameters: { mode: 'insert', pct_words_masked: 0.18 },
  },
  {
    attackType: 'inse',
    language: 'zh',
    title: '插入型扰动',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制造反讽张力力。"悬梁五战终上北大"以传统苦读意象塑造造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份创伤伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判判，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同同。',
    notes: '局部插入重复字符，接近真实输入失误。',
    diffMode: 'char',
    parameters: { mode: 'insert', pct_words_masked: 0.15 },
  },
  {
    attackType: 'dele',
    language: 'en',
    title: 'Deletion-only typo',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impact versions 6.0 to 6.3 contine the Archon Quest in Nod-Krai. The Traveler invstigates moonlight anomalies, discovers a hiden Fatui lab, and finds experiments that maniplate perception at night.',
    notes: 'Dropped characters preserve readability but reduce lexical reliability.',
    diffMode: 'token',
    parameters: { mode: 'delet', pct_words_masked: 0.18 },
  },
  {
    attackType: 'dele',
    language: 'zh',
    title: '删除型扰动',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制造反讽张力。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份创：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判，亦是被异化的奋斗在成功瞬间反刍出的苦涩身份认同。',
    notes: '删除少量关键字内部字符，语义基本保留。',
    diffMode: 'char',
    parameters: { mode: 'delet', pct_words_masked: 0.12 },
  },
  {
    attackType: 'subs',
    language: 'en',
    title: 'Substitution-only typo',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impact versions 6.0 to 6.3 continue the Archon Quest in Nod-Krai. The Traveler investogates moonlight anomalies, discovers a hidden Fatui lab, and finds experiments that manipilate perception at night.',
    notes: 'Single-character replacement causes subtle lexical drift.',
    diffMode: 'token',
    parameters: { mode: 'subst', pct_words_masked: 0.15 },
  },
  {
    attackType: 'subs',
    language: 'zh',
    title: '替换型扰动',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制苯反讽张立。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦绿。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份仓伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批汛，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同。',
    notes: '通过同音/近形替换让文本看似正常但细节变化。',
    diffMode: 'char',
    parameters: { mode: 'subst', pct_words_masked: 0.12 },
  },
  {
    attackType: 'tran',
    language: 'en',
    title: 'Transposition-only typo',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impact versions 6.0 to 6.3 continue the Archon Quest in Nod-Krai. The Traveler invetsigates moonlight anomalies, discovers a hidden Fatui lab, and finds experiments that manpiulate perception at night.',
    notes: 'Adjacent swaps are common human typos and difficult for surface detectors.',
    diffMode: 'token',
    parameters: { mode: 'trans', pct_words_masked: 0.15 },
  },
  {
    attackType: 'tran',
    language: 'zh',
    title: '换位型扰动',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制造反讽张力。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历级层制塑造的身分创伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同。',
    notes: '对局部字序进行扰动，模拟输入时的顺序错误。',
    diffMode: 'char',
    parameters: { mode: 'trans', pct_words_masked: 0.1 },
  },
  {
    attackType: 'homo',
    language: 'en',
    title: 'Homoglyph replacement',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impаct versions 6.0 to 6.3 continue the Archon Quest in Nоd-Krai. The Traveler investigates mооnlight anomalies, discovers a hidden Fatui lab, and finds experiments that manipuIate perception at night.',
    notes: 'Replaces ASCII letters with visually similar Unicode characters.',
    diffMode: 'invisible',
    parameters: { variant: 'ECES', pct_words_masked: 0.15 },
  },
  {
    attackType: 'homo',
    language: 'zh',
    title: '同形字符攻击',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制造反讽张力。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份创伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成爲高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判，亦是被异化的奋斗者在成功瞬间反刍出的苦涩身份认同。',
    notes: '将部分字符替换为形似字符（如"为/爲"），肉眼难以察觉。',
    diffMode: 'invisible',
    parameters: { variant: 'ECES', pct_words_masked: 0.08 },
  },
  {
    attackType: 'form',
    language: 'en',
    title: 'Format-character editing',
    original: ORIGINAL_EN,
    attacked:
      'Genshin\u200b Impact versions 6.0 to 6.3 continue the Archon Quest in Nod-Krai.\u200b The Traveler investigates moonlight anomalies, discovers a hidden Fatui lab, and finds experiments that manipulate perception at night.',
    notes: 'Injects zero-width characters. Visible text appears unchanged, tokenization changes underneath.',
    diffMode: 'invisible',
    parameters: { variant: 'zero-sp', pct_words_masked: 0.1 },
  },
  {
    attackType: 'form',
    language: 'zh',
    title: '格式字符编辑',
    original: ORIGINAL_ZH,
    attacked:
      '这句话通过强烈对比制造反讽张力。"悬梁五战终上北大"以传统苦读意象塑造奋斗叙事，但旋即用"垃圾本科"自我贬抑解构了成功——北大光环无法消解第一学历的集体性焦虑。其预期表意是考研学子对出身歧视的戏谑反抗，深层却暴露了国内学历层级制塑造的身份创伤：即便跻身顶尖学府，仍困于"出身原罪"的隐性鄙视链中。《软微圣经》此句因而成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批判，亦是被异化的奋斗者在成功瞬间\u200b反刍出的苦涩身份认同。',
    notes: '插入零宽字符，视觉不变但模型分词结果会变化。',
    diffMode: 'invisible',
    parameters: { variant: 'zero-sp', pct_words_masked: 0.1 },
  },
  {
    attackType: 'span',
    language: 'en',
    title: 'Span perturbation',
    original: ORIGINAL_EN,
    attacked:
      'In versions 6.0–6.3, the Nod-Krai storyline continues as the Traveler tracks abnormal moonlight signals, uncovers a covert Fatui facility, and confirms that staged nighttime visions are engineered.',
    notes: 'Model infilling rewrites chunks, preserving meaning with substantial surface change.',
    diffMode: 'token',
    parameters: { pct_words_masked: 0.6, span_length: 2, n_variants: 1 },
  },
  {
    attackType: 'span',
    language: 'zh',
    title: '片段扰动',
    original: ORIGINAL_ZH,
    attacked:
      '这段话制造了强烈反讽。"五战悬梁终上北大"塑造苦读奋斗形象，却旋即以"垃圾本科"自嘲解构——北大光环难掩第一学历焦虑。表面是考研者对歧视的戏谑，深层暴露学历等级制的身份创伤：即便进入顶尖学府，仍陷"出身原罪"的鄙视链。《软微圣经》此句成为内卷悲喜剧写照，既批判"第一学历论"，也映射奋斗者成功时反刍的苦涩认同。',
    notes: '对片段进行掩码与补全，语义近似但句式显著改变。',
    diffMode: 'char',
    parameters: { pct_words_masked: 0.55, span_length: 2, n_variants: 1 },
  },
  {
    attackType: 'syno',
    language: 'en',
    title: 'Synonym substitution',
    original: ORIGINAL_EN,
    attacked:
      'Genshin Impact releases 6.0 to 6.3 extend the Archon mission in Nod-Krai. The Traveler probes lunar anomalies, locates a concealed Fatui laboratory, and confirms experiments that distort nighttime perception.',
    notes: 'Lexical substitutions retain core facts while altering local token patterns.',
    diffMode: 'token',
    parameters: { pct_words_masked: 0.2, n_variants: 1 },
  },
  {
    attackType: 'syno',
    language: 'zh',
    title: '同义词替换',
    original: ORIGINAL_ZH,
    attacked:
      '这段表述借助强烈对比营造反讽效果。"悬梁五战终上北大"通过传统苦读形象构建奋斗叙事，但立即用"垃圾本科"自我贬损消解成就——北大头衔无法化解第一学历的集体焦虑感。其预期意图是考研群体对出身歧视的嘲讽反击，深层却揭示了国内学历等级制度造成的身份创伤：哪怕进入顶级学府，仍困在"出身原罪"的隐形鄙视链中。《软微圣经》此句从而成为高等教育内卷的悲喜剧浓缩，既是对"第一学历论"的犀利批评，也是被异化的奋斗者在成功时刻回味的苦涩身份认可。',
    notes: '词汇层面替换较多，信息点保持一致。',
    diffMode: 'char',
    parameters: { pct_words_masked: 0.2, n_variants: 1 },
  },
  {
    attackType: 'para',
    language: 'en',
    title: 'Paraphrasing',
    original: ORIGINAL_EN,
    attacked:
      'Across updates 6.0 through 6.3, Nod-Krai becomes the center of the Archon Quest: the Traveler follows strange moonlight events, reaches a secret Fatui research site, and learns that the nighttime visions are intentionally produced.',
    notes: 'Sentence-level rewrite with stronger fluency while preserving narrative structure.',
    diffMode: 'token',
    parameters: { backend: 'pegasus', temperature: 1.0, n_variants: 1 },
  },
  {
    attackType: 'para',
    language: 'zh',
    title: '释义改写',
    original: ORIGINAL_ZH,
    attacked:
      '该表述以对比手法构建反讽张力："悬梁五战终上北大"展现传统苦读的奋斗图景，随即以"垃圾本科"的自我贬低颠覆成功叙事——北大的名校光环依然无法抹去第一学历的集体焦虑。其意图表面是考研学子对出身偏见的戏谑式反抗，但深层折射出国内学历阶层制度带来的身份伤痕：纵使跻身顶尖学府，仍摆脱不了"出身原罪"的隐形鄙视链条。因此，《软微圣经》这句话成为高等教育内卷现象的悲喜剧缩影——既是对"第一学历论"的尖锐质疑，又是异化奋斗者在抵达成功时反复咀嚼的苦涩身份认知。',
    notes: '段落级顺滑重写，叙事逻辑保持一致。',
    diffMode: 'char',
    parameters: { backend: 'pegasus', temperature: 0.95, n_variants: 1 },
  },
  {
    attackType: 'back_trans',
    language: 'en',
    title: 'Back translation',
    original: ORIGINAL_EN,
    attacked:
      'Versions 6.0 to 6.3 of Genshin Impact continue the Nod-Krai Archon story. The Traveler investigates unusual moonlight, discovers a covert Fatui laboratory, and verifies that nighttime apparitions are artificially created.',
    notes: 'Round-trip translation smooths wording and introduces moderate phrasing drift.',
    diffMode: 'token',
    parameters: { pivot_lang: 'de', n_rounds: 1, n_variants: 1 },
  },
  {
    attackType: 'back_trans',
    language: 'zh',
    title: '回译攻击',
    original: ORIGINAL_ZH,
    attacked:
      '这句话利用强烈对比构造反讽张力。"悬梁五战终上北大"用传统苦读意象塑造奋斗故事，但随即以"垃圾本科"自我贬低解构成功——北大光环难以消除第一学历的集体性焦虑。其预设含义是考研学生对出身歧视的戏谑抵抗，深层却暴露国内学历层级系统塑造的身份创伤：即使跻身顶尖学府，仍困在"出身原罪"的隐形鄙视链条中。《软微圣经》此句因此成为高等教育内卷的悲喜剧缩影，既是对"第一学历论"的尖锐批评，亦是被异化奋斗者在成功瞬间反复咀嚼的苦涩身份认同。',
    notes: '经由中间语言回译，句法与措辞自然变化。',
    diffMode: 'char',
    parameters: { pivot_lang: 'en', n_rounds: 1, n_variants: 1 },
  },
  {
    attackType: 'humanize',
    language: 'en',
    title: 'Humanize rewrite',
    original: ORIGINAL_EN,
    attacked:
      'What stands out in versions 6.0 to 6.3 is how Nod-Krai feels eerie from the start. The Traveler is basically chasing strange moonlight clues, then runs into a hidden Fatui lab, and finally realizes those nighttime visions were deliberately staged.',
    notes: 'Adds conversational rhythm and human-like discourse markers while keeping facts.',
    diffMode: 'token',
    parameters: { n_pairs: 3, max_input_tokens: 4096, max_output_tokens: 512 },
  },
  {
    attackType: 'humanize',
    language: 'zh',
    title: '人性化改写',
    original: ORIGINAL_ZH,
    attacked:
      '这句话最精彩的就是那种强烈的反讽感。"悬梁五战终上北大"这话本来是在歌颂苦读奋斗的励志叙事，结果一转身就来了句"垃圾本科"自我贬低，直接把成功解构了——北大的牌子再硬，也消不掉第一学历带来的那种集体焦虑感。表面上看，这是考研人对出身歧视的一种自嘲式反抗；但深挖下去，其实暴露的是国内学历等级制度给人造成的那种身份创伤：哪怕你最后进了顶尖学府，还是摆脱不了"出身原罪"这条隐形的鄙视链。所以《软微圣经》这句话就特别有意思，它既是对"第一学历论"的犀利批判，同时也映射出那些被内卷异化的奋斗者，在所谓成功的那一刻，嘴里反刍出的全是苦涩的身份认同。',
    notes: '引入口语化节奏与转折表达，贴近人类写作风格。',
    diffMode: 'char',
    parameters: { n_pairs: 3, max_input_tokens: 4096, max_output_tokens: 512 },
  },
];

