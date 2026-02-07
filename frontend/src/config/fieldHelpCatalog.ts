import { FieldHelpEntry } from '../types';

export const EXACT_FIELD_HELP: Record<string, FieldHelpEntry> = {
  data: {
    purpose: 'Input dataset path used by this pipeline.',
    higher: 'Not directly applicable. Choose a larger file only if you want more samples.',
    lower: 'Not directly applicable. Smaller files run faster but cover less data.',
  },
  out: {
    purpose: 'Output file path for generated predictions or transformed dataset.',
    higher: 'Not directly applicable. The path itself does not improve quality.',
    lower: 'Not directly applicable. Keep it unique to avoid accidental overwrite.',
  },
  dataset_train: {
    purpose: 'Training dataset path.',
    higher: 'Not directly applicable. More rows usually improve robustness but increase time.',
    lower: 'Not directly applicable. Fewer rows train faster but may underfit.',
  },
  dataset_valid: {
    purpose: 'Validation dataset path used during training.',
    higher: 'Not directly applicable. More validation samples produce stabler metrics.',
    lower: 'Not directly applicable. Too small validation sets can make metrics noisy.',
  },
  dataset_test: {
    purpose: 'Test dataset path for final evaluation.',
    higher: 'Not directly applicable. Larger test sets provide more reliable final metrics.',
    lower: 'Not directly applicable. Smaller sets are faster but less statistically stable.',
  },
  gpu_ids: {
    purpose: 'GPU devices assigned to this job.',
    higher: 'Using more GPUs can increase throughput, but communication overhead may increase.',
    lower: 'Using fewer GPUs reduces memory/compute parallelism and can slow execution.',
  },
  hf_endpoint: {
    purpose: 'Hugging Face Hub endpoint used for model/tokenizer downloads.',
    higher: 'Not directly applicable. This is a source selection, not a numeric control.',
    lower: 'Not directly applicable. Choose official or mirror based on network quality.',
  },
  hf_token: {
    purpose: 'Authentication token for gated/private Hugging Face models.',
    higher: 'Not directly applicable. Token length does not affect model quality.',
    lower: 'Not directly applicable. Empty token may fail on restricted model downloads.',
  },
  prompt_from_label: {
    purpose:
      'Label ID used to choose prompt source examples (commonly 0=human, 1=machine, depending on dataset labels). If only_human_prompts=1, this value is overridden to 0.',
    higher: 'Higher label IDs select a different source class only if that label exists in the dataset.',
    lower: 'Lower label IDs select a different source class only if that label exists in the dataset.',
  },
  only_human_prompts: {
    purpose:
      'Boolean-like switch (0/1). When set to 1, prompt_from_label is forced to 0, so prompts always come from human-labeled samples.',
    higher: '1 enables strict human-prompt mode and ignores prompt_from_label.',
    lower: '0 disables the override and uses prompt_from_label as configured.',
  },
  prompt_template: {
    purpose:
      'Template used to build the final generation prompt. Supported placeholders: {prefix} (truncated source text), {text} (full original text), {id} (sample id), {lang} (language), {source} (dataset/source field), {label} (label value), {meta_json} (JSON metadata). Example: "You are writing in {lang}. Continue:\\n{prefix}".',
    higher: 'Not directly applicable. This is a template design choice, not a numeric value.',
    lower: 'Not directly applicable. Simpler templates are faster to iterate but may provide weaker instruction context.',
  },
  prompt_template_file: {
    purpose: 'Optional file path for loading prompt_template from an external text file. If provided, file content overrides inline prompt_template.',
    higher: 'Not directly applicable. This is a file/path selection, not a scalar.',
    lower: 'Not directly applicable. Invalid paths prevent template loading.',
  },
  machine_text_mode: {
    purpose: 'Controls what is stored as machine text. Options: prompt_plus (store prompt + completion), completion_only (store completion only).',
    higher: 'Not directly applicable. This is a mode selection, not a scalar.',
    lower: 'Not directly applicable. Choose based on whether you want prompt context included in stored machine text.',
  },
  detector: {
    purpose: 'Detector backend family used for evaluation or training.',
    higher: 'Not directly applicable. This is a model choice, not a scalar.',
    lower: 'Not directly applicable. Different detectors trade off speed, robustness, and accuracy.',
  },
  'detector_kwargs.threshold': {
    purpose:
      'Decision boundary for classifying machine-generated text. Scores above this threshold are classified as machine-generated. Default values vary by detector (typically 0.5 for probability-based detectors, detector-specific for metric-based ones like DetectGPT).',
    higher:
      'Higher threshold (e.g., 0.7-0.9) makes machine prediction stricter, reducing false positives (fewer human texts misclassified as machine) but potentially missing some machine-generated content (lower recall). Useful when precision is more important than recall.',
    lower:
      'Lower threshold (e.g., 0.1-0.3) makes machine prediction easier, increasing recall (catching more machine-generated text) but raising false positive rate (more human texts flagged incorrectly). Useful when catching all machine text is critical.',
  },
  'detector_kwargs.batch_size': {
    purpose:
      'Per-step inference/evaluation batch size inside detector execution. Typical values: 16-32 for small models on consumer GPUs, 64-128 for larger GPUs or smaller models. Affects memory usage and speed.',
    higher:
      'Higher batch size (e.g., 64-128) improves throughput by utilizing GPU parallelism, but increases VRAM usage significantly. Risk of Out-Of-Memory (OOM) errors especially with large models or long sequences. Monitor GPU memory usage carefully.',
    lower:
      'Lower batch size (e.g., 1-8) is safer for limited memory but increases total runtime due to reduced GPU utilization. Recommended for debugging, large models, or when running multiple jobs simultaneously.',
  },
  'detector_kwargs.max_length': {
    purpose:
      'Maximum token length processed per sample. Texts longer than this are truncated. Typical values: 512 for most BERT-based detectors, 1024-2048 for longer-context models, 128-256 for fast screening. Model-specific limits apply.',
    higher:
      'Higher length (e.g., 1024-2048) retains more context and may improve detection quality for long documents, but significantly increases memory usage and latency. May hit model maximum sequence length.',
    lower:
      'Lower length (e.g., 128-512) speeds up inference and reduces memory, but truncates long texts and may miss important patterns in later parts of the document. Suitable for short texts or initial screening.',
  },
  'detector_kwargs.top_p': {
    purpose: 'Nucleus sampling cutoff used when detector includes text generation.',
    higher: 'Higher top_p keeps more candidate tokens and increases diversity.',
    lower: 'Lower top_p narrows sampling and makes outputs more conservative.',
  },
  'detector_kwargs.top_k': {
    purpose: 'Top-k sampling cap for token candidates.',
    higher: 'Higher top_k allows more token variety and can add randomness.',
    lower: 'Lower top_k constrains generation and tends to be more deterministic.',
  },
  'detector_kwargs.temperature': {
    purpose: 'Sampling temperature for generative components.',
    higher: 'Higher temperature increases randomness and diversity.',
    lower: 'Lower temperature reduces randomness and makes outputs more deterministic.',
  },
  'detector_kwargs.model1': {
    purpose: 'Primary model used by model-based detector methods.',
    higher: 'Not directly applicable. This is a model choice.',
    lower: 'Not directly applicable. Smaller models are usually faster but may be less accurate.',
  },
  'detector_kwargs.model2': {
    purpose: 'Secondary/reference model used by dual-model detectors.',
    higher: 'Not directly applicable. This is a model choice.',
    lower: 'Not directly applicable. Ensure compatibility with model1 for fair comparison.',
  },
  calibrator_path: {
    purpose: 'Path to saved calibrator artifact used by metric detectors.',
    higher: 'Not directly applicable. This is a file/path selection.',
    lower: 'Not directly applicable. Wrong path prevents calibrated inference.',
  },
  attack_dataset_only: {
    purpose: 'Switch between dataset-level attack generation and full pipeline behavior.',
    higher: 'Enabled focuses on producing attacked dataset artifacts only.',
    lower: 'Disabled may run additional evaluation/build steps depending on config.',
  },
  save_attack_outputs: {
    purpose: 'Whether to keep additional per-attack output artifacts.',
    higher: 'Enabled keeps richer debugging/analysis files at the cost of storage.',
    lower: 'Disabled keeps output minimal and usually leaves only primary target files.',
  },
  keep_attack_aux_files: {
    purpose: 'Whether temporary attack config/intermediate files are retained.',
    higher: 'Enabled retains intermediates for debugging and reproducibility checks.',
    lower: 'Disabled removes auxiliary files to keep output directory clean.',
  },
  text: {
    purpose: 'Input text to classify in demo detection.',
    higher: 'Longer text usually gives detector more signal but costs more compute.',
    lower: 'Shorter text is faster but may reduce confidence/reliability.',
  },
};

export const LEAF_FIELD_HELP: Record<string, FieldHelpEntry> = {
  batch_size: {
    purpose: 'Number of samples processed per forward step.',
    higher: 'Higher value improves throughput but increases memory usage.',
    lower: 'Lower value is safer for memory and slower overall.',
  },
  eval_batch_size: {
    purpose: 'Batch size for validation/test evaluation.',
    higher: 'Higher value speeds evaluation but can trigger VRAM OOM.',
    lower: 'Lower value is more stable but slower.',
  },
  epochs: {
    purpose: 'Total passes over training data.',
    higher: 'More epochs can improve fit but may overfit and increase training time.',
    lower: 'Fewer epochs are faster but can underfit.',
  },
  lr: {
    purpose: 'Learning rate for optimizer updates.',
    higher: 'Higher learning rate converges faster initially but can become unstable.',
    lower: 'Lower learning rate is stable but may train too slowly.',
  },
  learning_rate: {
    purpose: 'Learning rate controlling update step size.',
    higher: 'Higher values risk divergence or oscillation.',
    lower: 'Lower values are safer but may need more epochs.',
  },
  weight_decay: {
    purpose: 'Regularization strength applied to model weights.',
    higher: 'Higher decay reduces overfitting risk but may underfit.',
    lower: 'Lower decay allows tighter fit but may overfit.',
  },
  warmup_steps: {
    purpose: 'Number of initial optimizer warmup steps.',
    higher: 'Longer warmup can stabilize early training at the cost of slower ramp-up.',
    lower: 'Short warmup ramps quickly but may be less stable.',
  },
  max_steps: {
    purpose: 'Hard cap on training update steps.',
    higher: 'More steps allow longer training and higher compute cost.',
    lower: 'Fewer steps finish faster and may stop before convergence.',
  },
  gradient_accumulation_steps: {
    purpose: 'Micro-batches accumulated before one optimizer step.',
    higher: 'Higher values simulate larger effective batch size with slower step cadence.',
    lower: 'Lower values update more frequently with smaller effective batch.',
  },
  threshold: {
    purpose: 'Decision cutoff for machine/human classification.',
    higher: 'Higher threshold is stricter for machine label predictions.',
    lower: 'Lower threshold is more permissive for machine label predictions.',
  },
  pct_words_masked: {
    purpose: 'Fraction of words selected for perturbation in token-level attacks.',
    higher:
      'Higher masking ratio increases attack strength and variation, but can damage readability and semantic fidelity.',
    lower:
      'Lower masking ratio keeps text closer to the original and more natural, but often weakens attack effectiveness.',
  },
  n_variants: {
    purpose: 'Number of attacked variants generated for each original sample.',
    higher:
      'More variants improve coverage and robustness evaluation, but multiply runtime and storage cost.',
    lower:
      'Fewer variants reduce compute/storage overhead, but may under-sample attack diversity.',
  },
  n_pairs: {
    purpose: 'Number of style or rewrite pairs used for humanization/retrieval-driven attack construction.',
    higher:
      'More pairs usually improve rewrite quality/stability by adding richer references, but increase latency and token usage.',
    lower:
      'Fewer pairs are faster and cheaper, but outputs may be less stable and stylistically weaker.',
  },
  max_input_tokens: {
    purpose: 'Maximum input token budget consumed before rewrite/inference for an attack step.',
    higher:
      'Higher limits preserve more context and can improve quality for long inputs, but raise memory/time and API cost.',
    lower:
      'Lower limits reduce compute cost and latency, but may truncate important context.',
  },
  max_output_tokens: {
    purpose: 'Maximum generated token budget for each attacked output.',
    higher:
      'Higher values allow richer/longer rewrites but increase generation cost and completion time.',
    lower:
      'Lower values keep outputs concise and fast, but can cut off important content.',
  },
  max_nodes_num: {
    purpose: 'Upper bound on graph nodes kept for graph-based modeling features.',
    higher:
      'Higher node limits preserve more structural information but significantly increase graph memory and compute complexity.',
    lower:
      'Lower node limits simplify computation and improve speed, but may discard useful relational structure.',
  },
  with_relation: {
    purpose: 'Enable relation-aware graph edges/features in graph neural modules.',
    higher:
      'Not directly applicable. Enabled mode introduces relational signals that can improve structure modeling with extra complexity.',
    lower:
      'Not directly applicable. Disabled mode is simpler and faster but may lose relational cues.',
  },
  gcn_layers: {
    purpose: 'Number of Graph Convolution layers used for message passing.',
    higher:
      'More layers capture wider-hop graph context but can over-smooth node representations and raise compute cost.',
    lower:
      'Fewer layers are faster and reduce over-smoothing risk, but may miss long-range structure.',
  },
  span_length: {
    purpose: 'Average length of contiguous masked spans in span-perturbation attacks.',
    higher:
      'Longer spans create stronger paraphrastic changes but can harm local coherence if too aggressive.',
    lower:
      'Shorter spans preserve local fluency and meaning better, but attack perturbation becomes milder.',
  },
  mask_top_p: {
    purpose: 'Top-p sampling cutoff used while filling masked spans.',
    higher:
      'Higher top-p increases lexical diversity and attack variability, but may introduce noisy generations.',
    lower:
      'Lower top-p constrains generation to safer candidates, improving consistency but reducing diversity.',
  },
  num_replacement_retry: {
    purpose: 'Maximum retry attempts when synonym/model-based replacement fails quality checks.',
    higher:
      'More retries can improve successful replacement rate and quality, but increase runtime.',
    lower:
      'Fewer retries run faster, but may leave more tokens unchanged or low-quality replacements.',
  },
  vllm_batch_size: {
    purpose: 'Batch size used by vLLM-backed generation within attacks.',
    higher:
      'Higher batch size improves throughput on sufficient VRAM, but increases OOM risk and tail latency.',
    lower:
      'Lower batch size is safer for memory and mixed workloads, but slower overall.',
  },
  vllm_tensor_parallel_size: {
    purpose: 'Number of GPUs used for tensor-parallel execution in vLLM.',
    higher:
      'Higher parallel size can enable larger models and higher throughput, but adds communication overhead and scheduling constraints.',
    lower:
      'Lower parallel size is easier to schedule and simpler to debug, but may limit model scale and speed.',
  },
  vllm_gpu_memory_utilization: {
    purpose: 'Target fraction of each GPU memory reserved by vLLM allocator.',
    higher:
      'Higher utilization improves memory usage efficiency for large batches/models but increases OOM sensitivity.',
    lower:
      'Lower utilization leaves safer memory headroom for stability, but may reduce maximum feasible workload.',
  },
  temperature: {
    purpose: 'Sampling randomness control.',
    higher: 'Higher temperature increases diversity/randomness.',
    lower: 'Lower temperature yields more deterministic outputs.',
  },
  top_p: {
    purpose: 'Nucleus sampling probability mass retained.',
    higher: 'Higher top_p broadens candidate token pool.',
    lower: 'Lower top_p narrows candidates and increases determinism.',
  },
  top_k: {
    purpose: 'Maximum number of candidate tokens at each step.',
    higher: 'Higher top_k allows more diversity.',
    lower: 'Lower top_k makes generation more conservative.',
  },
  max_length: {
    purpose: 'Maximum token length considered/generated.',
    higher: 'Higher max length uses more context and more memory/time.',
    lower: 'Lower max length is faster but can truncate context.',
  },
  max_new_tokens: {
    purpose: 'Maximum number of newly generated tokens.',
    higher: 'Higher values allow longer generations and slower runtime.',
    lower: 'Lower values shorten outputs and reduce latency.',
  },
  min_length: {
    purpose: 'Minimum output length constraint.',
    higher: 'Higher minimum length enforces longer outputs.',
    lower: 'Lower minimum length allows shorter outputs.',
  },
  num_beams: {
    purpose: 'Beam search width during generation.',
    higher: 'More beams can improve search quality but increase latency.',
    lower: 'Fewer beams are faster but explore less candidate space.',
  },
  repetition_penalty: {
    purpose: 'Penalty factor reducing repeated token sequences.',
    higher: 'Stronger penalty reduces repetition but can hurt fluency.',
    lower: 'Weaker penalty allows repetition and can improve coherence for some tasks.',
  },
  length_penalty: {
    purpose: 'Length bias during beam search scoring.',
    higher: 'Higher values favor longer outputs.',
    lower: 'Lower values favor shorter outputs.',
  },
  k_runs: {
    purpose: 'Number of repeated runs for stable metrics.',
    higher: 'More runs improve statistical stability but cost more time.',
    lower: 'Fewer runs are faster with higher variance.',
  },
  sample_k: {
    purpose: 'Number of sampled items or perturbations.',
    higher: 'More samples improve robustness but increase runtime.',
    lower: 'Fewer samples are faster but noisier.',
  },
  num_workers: {
    purpose: 'Parallel worker processes for data loading/preprocessing.',
    higher: 'More workers can improve throughput but raise CPU/memory pressure.',
    lower: 'Fewer workers reduce overhead but can bottleneck data pipeline.',
  },
  seed: {
    purpose: 'Random seed controlling reproducibility.',
    higher: 'Not directly quality-related. Different seeds change random draws.',
    lower: 'Not directly quality-related. Keep fixed for reproducible comparisons.',
  },
  dropout: {
    purpose: 'Dropout probability for regularization.',
    higher: 'Higher dropout regularizes more and may underfit.',
    lower: 'Lower dropout fits training data tighter and may overfit.',
  },
  alpha: {
    purpose: 'Algorithm-specific mixing/weight coefficient.',
    higher: 'Higher alpha increases the contribution of its paired term.',
    lower: 'Lower alpha reduces the contribution of its paired term.',
  },
  beta: {
    purpose: 'Algorithm-specific balancing coefficient.',
    higher: 'Higher beta strengthens its controlled effect.',
    lower: 'Lower beta weakens its controlled effect.',
  },
  gamma: {
    purpose: 'Algorithm-specific scaling coefficient.',
    higher: 'Higher gamma amplifies the corresponding term.',
    lower: 'Lower gamma reduces the corresponding term.',
  },
  model: {
    purpose: 'Model identifier/path used by this module.',
    higher: 'Not directly applicable. Larger models can be stronger but slower.',
    lower: 'Not directly applicable. Smaller models are faster but may lose quality.',
  },
  model1: {
    purpose: 'Primary model in dual-model setups.',
    higher: 'Not directly applicable. This is a model selection.',
    lower: 'Not directly applicable. Match tokenizer and architecture expectations.',
  },
  model2: {
    purpose: 'Secondary/reference model in dual-model setups.',
    higher: 'Not directly applicable. This is a model selection.',
    lower: 'Not directly applicable. Keep semantic role consistent with method design.',
  },
  model3: {
    purpose: 'Optional third model in extended setups.',
    higher: 'Not directly applicable. This is a model selection.',
    lower: 'Not directly applicable. Remove if method does not require it.',
  },
  device: {
    purpose: 'Computation device target for execution.',
    higher: 'Not directly applicable. GPU is faster; CPU is more compatible.',
    lower: 'Not directly applicable. Use cpu when GPU memory is insufficient.',
  },
  dtype: {
    purpose: 'Numeric precision for tensor computation.',
    higher: 'Not directly applicable. Higher precision improves stability but costs memory.',
    lower: 'Not directly applicable. Lower precision is faster and lighter but may lose precision.',
  },
  backend: {
    purpose: 'Backend engine/provider used by the selected attack or detector.',
    higher: 'Not directly applicable. Backend choice affects quality, speed, and dependencies.',
    lower: 'Not directly applicable. Choose based on availability and runtime constraints.',
  },
  calibrator_path: {
    purpose: 'Calibrator file/directory path used for probability calibration.',
    higher: 'Not directly applicable. Path correctness is what matters.',
    lower: 'Not directly applicable. Invalid path disables calibrated output.',
  },
  data: {
    purpose: 'Path to input dataset file.',
    higher: 'Not directly applicable. Larger datasets increase runtime and coverage.',
    lower: 'Not directly applicable. Smaller datasets run quickly but give fewer signals.',
  },
  out: {
    purpose: 'Path for main output artifact.',
    higher: 'Not directly applicable. Use clear naming to keep experiment traces clean.',
    lower: 'Not directly applicable. Reusing existing files may overwrite prior outputs.',
  },
};

export const EXACT_FIELD_HELP_ZH: Partial<Record<string, FieldHelpEntry>> = {
  data: {
    purpose: '当前任务使用的输入数据集路径。',
    higher: '不直接适用。更大的数据通常覆盖更充分，但耗时更长。',
    lower: '不直接适用。更小的数据运行更快，但统计稳定性更弱。',
  },
  out: {
    purpose: '当前任务主输出文件路径。',
    higher: '不直接适用。路径本身不是数值控制项。',
    lower: '不直接适用。建议使用唯一文件名避免覆盖历史结果。',
  },
  gpu_ids: {
    purpose: '指定任务可使用的 GPU 设备编号列表。',
    higher: '使用更多 GPU 通常能提升吞吐，但会增加通信和调度开销。',
    lower: '使用更少 GPU 资源更省，但运行可能更慢。',
  },
  hf_endpoint: {
    purpose: 'HuggingFace 模型下载源地址（官方或镜像）。',
    higher: '不直接适用。这是来源选择，不是数值大小。',
    lower: '不直接适用。应根据网络条件选择官方或镜像源。',
  },
  hf_token: {
    purpose: '用于访问受限/私有 HuggingFace 模型的认证 Token。',
    higher: '不直接适用。长度不影响效果，关键是 Token 是否有效。',
    lower: '不直接适用。为空或无效时可能无法下载 gated 模型。',
  },
  prompt_from_label: {
    purpose: '选择哪一类标签样本作为 prompt 来源（常见 0=human, 1=machine）。',
    higher: '更高标签值会切换到另一类样本（前提是该标签在数据中存在）。',
    lower: '更低标签值会切换到另一类样本（前提是该标签在数据中存在）。',
  },
  only_human_prompts: {
    purpose: '是否强制只用 human 样本作为 prompt（1=强制，0=按 prompt_from_label）。',
    higher: '取 1 时会覆盖 prompt_from_label，固定使用标签 0 作为 prompt 来源。',
    lower: '取 0 时不覆盖，按 prompt_from_label 选择 prompt 来源。',
  },
  prompt_template: {
    purpose:
      '构造最终生成提示词的模板。可用占位符：{prefix}、{text}、{id}、{lang}、{source}、{label}、{meta_json}。',
    higher: '不直接适用。模板复杂度提升通常能提供更多控制信息，但也更难调参。',
    lower: '不直接适用。模板更简洁可读性更好，但控制力度可能下降。',
  },
  machine_text_mode: {
    purpose: '控制保存的 machine text 形式：prompt_plus（prompt+续写）或 completion_only（仅续写）。',
    higher: '不直接适用。这是模式选择，不是数值大小。',
    lower: '不直接适用。应根据你是否需要保留 prompt 上下文来选择。',
  },
  detector: {
    purpose: '当前任务使用的检测器类型。',
    higher: '不直接适用。检测器是类别选择，不是数值调节。',
    lower: '不直接适用。不同检测器在速度、鲁棒性和精度上各有取舍。',
  },
  'detector_kwargs.threshold': {
    purpose:
      '判定机器生成文本的决策边界。高于此阈值的分数被分类为机器生成。默认值因检测器而异（基于概率的检测器通常为 0.5，基于度量的检测器如 DetectGPT 则取决于具体实现）。',
    higher:
      '更高的阈值（例如 0.7-0.9）使机器生成判定更严格，减少假阳性（更少人类文本被误判为机器生成），但可能遗漏部分机器生成内容（召回率下降）。适用于对精确率要求高于召回率的场景。',
    lower:
      '更低的阈值（例如 0.1-0.3）使机器生成判定更宽松，提高召回率（捕获更多机器生成文本），但会提高假阳性率（更多人类文本被错误标记）。适用于需要尽可能捕获所有机器文本的场景。',
  },
  'detector_kwargs.batch_size': {
    purpose:
      '检测器执行时每步的推理批大小。典型值：小模型在消费级GPU上为16-32，大GPU或小模型为64-128。影响内存使用和速度。',
    higher:
      '更大的批大小（例如 64-128）通过利用 GPU 并行性提高吞吐量，但显著增加显存占用。特别是在大模型或长序列场景下容易出现显存溢出（OOM）错误。需要仔细监控 GPU 内存使用。',
    lower:
      '更小的批大小（例如 1-8）对有限内存更安全，但由于 GPU 利用率降低而增加总运行时间。适用于调试、大模型或同时运行多个作业的情况。',
  },
  'detector_kwargs.max_length': {
    purpose:
      '每个样本处理的最大令牌长度。超过此长度的文本会被截断。典型值：大多数 BERT 检测器为 512，长上下文模型为 1024-2048，快速筛选为 128-256。受模型特定限制约束。',
    higher:
      '更高的长度（例如 1024-2048）保留更多上下文，可能提高长文档的检测质量，但显著增加内存使用和延迟。可能达到模型的最大序列长度限制。',
    lower:
      '更低的长度（例如 128-512）加快推理速度并减少内存，但会截断长文本，可能错过文档后半部分的重要模式。适用于短文本或初步筛选。',
  },
  'detector_kwargs.temperature': {
    purpose: '生成组件的采样温度。仅在生成型检测器中有效（如 DetectGPT 的扰动生成）。',
    higher: '更高的温度（例如 1.2-1.5）增加随机性和多样性，生成的扰动文本变化更大。',
    lower: '更低的温度（例如 0.5-0.8）减少随机性，使输出更确定性和保守。接近 0 时几乎完全确定性。',
  },
  'detector_kwargs.top_p': {
    purpose: 'Nucleus 采样截断值。仅在包含文本生成的检测器中使用（如扰动生成）。',
    higher: '更高的 top_p（例如 0.95-1.0）保留更多候选令牌，增加生成多样性。',
    lower: '更低的 top_p（例如 0.6-0.8）缩小采样范围，使输出更保守和可预测。',
  },
  'detector_kwargs.top_k': {
    purpose: 'Top-k 采样的候选令牌数量上限。仅在生成型检测器中有效。',
    higher: '更高的 top_k（例如 100-200）允许更多令牌变化，可能增加随机性。',
    lower: '更低的 top_k（例如 10-50）约束生成，倾向于更确定性的输出。',
  },
  'detector_kwargs.model1': {
    purpose: '基于模型的检测器使用的主要模型。不同检测器对应不同作用（如 Binoculars 的观察者模型，DetectGPT 的评分模型）。',
    higher: '不直接适用。这是模型选择，不是数值。',
    lower: '不直接适用。更小的模型通常更快但可能精度较低，更大的模型精度更高但资源消耗更大。',
  },
  'detector_kwargs.model2': {
    purpose: '双模型检测器使用的辅助/参考模型（如 Binoculars 的执行者模型）。',
    higher: '不直接适用。这是模型选择，不是数值。',
    lower: '不直接适用。确保与 model1 的兼容性以进行公平比较。',
  },
  'detector_kwargs.model3': {
    purpose: '部分检测器使用的第三个模型（如 GREATER 的多模型集成）。',
    higher: '不直接适用。这是模型选择，不是数值。',
    lower: '不直接适用。需根据具体检测器文档选择合适的模型。',
  },
  calibrator_path: {
    purpose: 'Metric 检测器使用的校准器文件/目录路径。',
    higher: '不直接适用。路径正确性比“大小”更重要。',
    lower: '不直接适用。路径错误会导致无法启用校准推理。',
  },
  text: {
    purpose: 'Demo 检测输入文本。',
    higher: '文本越长，模型可利用信号通常越多，但计算开销更高。',
    lower: '文本越短，计算更快，但置信度和稳定性可能下降。',
  },
};

export const LEAF_FIELD_HELP_ZH: Partial<Record<string, FieldHelpEntry>> = {
  batch_size: {
    purpose: '每一步并行处理的样本数。',
    higher: '更大通常吞吐更高，但显存占用更高，OOM 风险增加。',
    lower: '更小更稳妥，但整体运行更慢。',
  },
  eval_batch_size: {
    purpose: '验证/测试阶段批大小。',
    higher: '更大可加快评估，但更容易触发显存不足。',
    lower: '更小更稳定，但评估更慢。',
  },
  epochs: {
    purpose: '训练轮数（完整遍历训练集的次数）。',
    higher: '轮数更多可能提升拟合效果，但过拟合风险和耗时上升。',
    lower: '轮数更少更快，但可能训练不足。',
  },
  lr: {
    purpose: '优化器学习率。',
    higher: '学习率更高收敛更快，但训练不稳定风险更高。',
    lower: '学习率更低更稳定，但收敛速度可能偏慢。',
  },
  learning_rate: {
    purpose: '优化更新步长控制项。',
    higher: '值更高可能振荡或发散。',
    lower: '值更低更稳但通常需要更多训练步。',
  },
  threshold: {
    purpose: '人类/机器分类阈值。',
    higher: '阈值更高时，判为机器通常更严格，误报可能降低。',
    lower: '阈值更低时，判为机器更宽松，召回通常更高。',
  },
  pct_words_masked: {
    purpose: '攻击时参与扰动的词比例。',
    higher: '比例更高会增强扰动强度与多样性，但可能降低可读性并破坏语义保持。',
    lower: '比例更低更接近原文、可读性更好，但攻击效果通常变弱。',
  },
  n_variants: {
    purpose: '每条原始样本生成的攻击变体数量。',
    higher: '变体更多可提高覆盖度与鲁棒性评估稳定性，但耗时和存储成本线性增加。',
    lower: '变体更少运行更快、成本更低，但攻击覆盖范围变窄。',
  },
  n_pairs: {
    purpose: '人性化/检索式重写中使用的参考对数量。',
    higher: '对数更多通常提升改写稳定性与风格质量，但延迟和 token 成本上升。',
    lower: '对数更少更快更省，但风格迁移能力与稳定性可能下降。',
  },
  max_input_tokens: {
    purpose: '攻击阶段单次输入可消费的最大 token 上限。',
    higher: '上限更高可保留更多上下文，长文本效果更稳，但显存、时延与费用上升。',
    lower: '上限更低可降低成本和延迟，但可能截断关键信息。',
  },
  max_output_tokens: {
    purpose: '攻击生成阶段单次输出的最大 token 数。',
    higher: '上限更高允许更完整改写，但生成时间与成本增加。',
    lower: '上限更低更快更省，但可能出现输出被截断。',
  },
  max_nodes_num: {
    purpose: '图结构特征中允许保留的最大节点数。',
    higher: '节点更多可保留更丰富结构信息，但图计算和内存开销明显增加。',
    lower: '节点更少计算更轻，但可能丢失关键关系结构。',
  },
  with_relation: {
    purpose: '是否启用图模型中的关系边/关系特征。',
    higher: '不直接适用。启用后可利用关系信息，但复杂度更高。',
    lower: '不直接适用。禁用后更轻量，但可能损失关系建模能力。',
  },
  gcn_layers: {
    purpose: '图卷积网络的层数（消息传递深度）。',
    higher: '层数更多能建模更远距离关系，但更容易过平滑且计算更重。',
    lower: '层数更少更快更稳，但可能无法覆盖长程依赖。',
  },
  span_length: {
    purpose: '片段扰动中每段掩码 span 的平均长度。',
    higher: 'span 更长可产生更强改写效果，但可能牺牲局部连贯性。',
    lower: 'span 更短更易保持语义和流畅，但攻击扰动更弱。',
  },
  mask_top_p: {
    purpose: '片段补全时的 top-p 采样截断阈值。',
    higher: 'top-p 更高保留更多候选，改写更丰富但噪声风险上升。',
    lower: 'top-p 更低输出更稳更可控，但多样性下降。',
  },
  num_replacement_retry: {
    purpose: '替换失败时允许的最大重试次数。',
    higher: '重试次数更多可提升替换成功率和质量，但运行更慢。',
    lower: '重试次数更少速度更快，但可能保留更多未替换词。',
  },
  vllm_batch_size: {
    purpose: '攻击中使用 vLLM 推理时的批大小。',
    higher: '批更大吞吐更高，但显存压力更大、OOM 风险上升。',
    lower: '批更小更稳更安全，但总体运行更慢。',
  },
  vllm_tensor_parallel_size: {
    purpose: 'vLLM 张量并行使用的 GPU 数。',
    higher: '并行度更高可支持更大模型/更高吞吐，但通信和调度开销更高。',
    lower: '并行度更低更易部署和排障，但可扩展性与速度受限。',
  },
  vllm_gpu_memory_utilization: {
    purpose: 'vLLM 目标显存占用比例。',
    higher: '比例更高能提高显存利用率，但更容易触发显存边界问题。',
    lower: '比例更低留出更大安全余量，稳定性更好但最大负载能力下降。',
  },
  temperature: {
    purpose: '采样随机性温度参数。',
    higher: '更高会增加随机性与多样性。',
    lower: '更低会让输出更稳定、更确定。',
  },
  top_p: {
    purpose: '核采样概率质量截断。',
    higher: '更高会保留更多候选 token，输出更发散。',
    lower: '更低会收窄候选，输出更保守。',
  },
  top_k: {
    purpose: '每步采样候选 token 上限。',
    higher: '更高允许更多候选，输出更灵活。',
    lower: '更低约束更强，输出更稳定。',
  },
  max_length: {
    purpose: '最大处理/生成 token 长度。',
    higher: '更大可保留更多上下文，但耗时和显存开销增加。',
    lower: '更小运行更快，但可能截断关键信息。',
  },
  max_new_tokens: {
    purpose: '最大新生成 token 数。',
    higher: '更大允许更长输出，但延迟更高。',
    lower: '更小可缩短输出并降低延迟。',
  },
  model: {
    purpose: '任务使用的模型名称或本地路径。',
    higher: '不直接适用。大模型通常更强但更慢、更耗资源。',
    lower: '不直接适用。小模型更快但可能损失效果。',
  },
  model1: {
    purpose: '检测器中的第一角色模型（具体语义由检测器定义）。',
    higher: '不直接适用。这是模型选择，不是数值调节。',
    lower: '不直接适用。应根据该检测器的模型角色说明选择。',
  },
  model2: {
    purpose: '检测器中的第二角色模型（通常作为对照/参考）。',
    higher: '不直接适用。这是模型选择，不是数值调节。',
    lower: '不直接适用。建议与 model1 的语义角色保持匹配。',
  },
  model3: {
    purpose: '检测器中的第三角色模型（可选）。',
    higher: '不直接适用。这是模型选择，不是数值调节。',
    lower: '不直接适用。不需要时可留空或移除。',
  },
  device: {
    purpose: '计算设备选择（如 cuda/cpu）。',
    higher: '不直接适用。GPU 通常更快，CPU 通常更兼容。',
    lower: '不直接适用。当显存不足时可切换 CPU。',
  },
  dtype: {
    purpose: '张量计算精度类型。',
    higher: '不直接适用。高精度更稳但更耗显存与算力。',
    lower: '不直接适用。低精度更快更省显存但可能损失精度。',
  },
  calibrator_path: {
    purpose: '校准器路径配置项。',
    higher: '不直接适用。关键是路径是否有效可读。',
    lower: '不直接适用。路径无效会导致校准失效。',
  },
};
