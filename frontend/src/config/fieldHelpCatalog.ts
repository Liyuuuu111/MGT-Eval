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
  enable_dataset_split: {
    purpose: 'Enable optional post-build split of the output dataset into train/dev/test files.',
    higher: 'Not directly applicable. Enabled writes extra split files next to the main output.',
    lower: 'Not directly applicable. Disabled keeps only the main output file.',
  },
  split_train_ratio: {
    purpose: 'Weight for assigning groups into the train split (range 0-10).',
    higher: 'Higher values route more groups to train, increasing training volume.',
    lower: 'Lower values reduce train coverage; 0 disables train split output.',
  },
  split_dev_ratio: {
    purpose: 'Weight for assigning groups into the dev/validation split (range 0-10).',
    higher: 'Higher values allocate more groups to dev, improving validation stability.',
    lower: 'Lower values allocate fewer groups to dev; 0 disables dev split output.',
  },
  split_test_ratio: {
    purpose: 'Weight for assigning groups into the test split (range 0-10).',
    higher: 'Higher values allocate more groups to test, improving final evaluation reliability.',
    lower: 'Lower values allocate fewer groups to test; 0 disables test split output.',
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
    purpose:
      'Build mode switch: when enabled, the pipeline only generates attacked dataset files and skips normal text generation/evaluation quality metrics.',
    higher:
      'Set to 1 when your goal is to produce an attack dataset for downstream detection robustness testing.',
    lower:
      'Set to 0 for the full build pipeline (generation + optional quality metrics + optional attack stage).',
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
  calibrator_name: {
    purpose:
      'Calibration algorithm name used when fitting/reading detector calibration (for example `platt_lr` or other runner-supported calibrators). This choice controls how raw detector scores are transformed into probabilities.',
    higher:
      'Not directly applicable. This is an algorithm selection, not a scalar. Different calibrators change probability mapping behavior and compatibility with saved calibrator JSON.',
    lower:
      'Not directly applicable. Prefer a calibrator that matches your detector score shape (single-score vs multi-feature).',
  },
  auto_calibrate: {
    purpose:
      'If enabled, the detector will try to automatically locate and load a matching calibrator file when `calibrator_path` is not explicitly provided. Matching is based on detector/model naming heuristics and packaged/user calibration directories.',
    higher:
      'Not directly applicable. Enabled mode improves convenience but can load a stale or unintended calibrator if multiple similar files exist.',
    lower:
      'Not directly applicable. Disabled mode avoids accidental auto-loading and requires explicit calibrator selection.',
  },
  force_runner_calibration: {
    purpose:
      'Forces calibration in the evaluator/runner layer even when detector internals may already produce probabilities. Use this for controlled experiments where all detectors share one external calibration policy.',
    higher:
      'Not directly applicable. Enabled mode enforces runner-side mapping, improving consistency across runs but potentially overriding detector-native calibration behavior.',
    lower:
      'Not directly applicable. Disabled mode keeps detector-native probability behavior when available.',
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
    purpose:
      'Subsample size for quick experiments. Uses only the first/selected K records instead of the full dataset.',
    higher:
      'Larger K makes metrics closer to full-dataset behavior, but runtime and GPU/CPU cost increase.',
    lower:
      'Smaller K is faster for debugging, but metrics become noisier and less reliable for final conclusions.',
  },
  api_model: {
    purpose:
      'Model ID sent to the OpenAI-compatible API backend (for example `gpt-4o-mini` or your deployed model name).',
    higher:
      'Not directly applicable. Pick a model that matches your quality, latency, and cost target.',
    lower:
      'Not directly applicable. Smaller/cheaper API models are faster but may reduce generation quality.',
  },
  api_key: {
    purpose:
      'API credential used to authenticate requests to the model provider. Keep it secret and never share logs with raw keys.',
    higher:
      'Not directly applicable. Key length is irrelevant as long as the token is valid.',
    lower:
      'Not directly applicable. Empty/invalid keys will cause authorization failures.',
  },
  api_base: {
    purpose:
      'Base URL for OpenAI-compatible services. Leave empty for official OpenAI; set it when using Azure/vLLM/other compatible gateways.',
    higher:
      'Not directly applicable. Correct endpoint compatibility matters more than URL form.',
    lower:
      'Not directly applicable. Wrong base URL often causes 404/connection errors.',
  },
  api_endpoint: {
    purpose:
      'API route style: `chat` for chat-completions format, `completions` for legacy prompt-completion format.',
    higher:
      'Not directly applicable. Use the endpoint that your provider/model actually supports.',
    lower:
      'Not directly applicable. Mismatched endpoint type usually causes request schema errors.',
  },
  api_timeout: {
    purpose:
      'Maximum seconds to wait for each API response before aborting the request.',
    higher:
      'Higher timeout reduces premature timeout failures on slow networks/large outputs, but failed requests take longer to return.',
    lower:
      'Lower timeout fails faster and improves responsiveness, but may interrupt valid long-running requests.',
  },
  metric_ppl: {
    purpose:
      'Enable perplexity-based fluency scoring (`0` off, `1` on). Useful to measure whether generated/attacked text remains language-model plausible.',
    higher:
      'Not directly applicable. Enabled mode adds an extra scoring pass and more compute time.',
    lower:
      'Not directly applicable. Disabled mode is faster but removes this quality signal.',
  },
  ppl_model: {
    purpose:
      'Language model used to compute perplexity (for example `gpt2`). This model is only for scoring quality, not for generation.',
    higher:
      'Not directly applicable. Larger scorer models can be more faithful but are heavier and slower.',
    lower:
      'Not directly applicable. Smaller scorer models are cheap but may be less sensitive to subtle fluency issues.',
  },
  ppl_device: {
    purpose:
      'Execution device for perplexity scoring (for example `cuda:0` or `cpu`).',
    higher:
      'Not directly applicable. GPU is usually faster; CPU is safer when GPU memory is tight.',
    lower:
      'Not directly applicable. Using CPU avoids VRAM contention but significantly increases latency.',
  },
  ppl_dtype: {
    purpose:
      'Numeric precision for perplexity scoring model (`auto`, `float16`, `bfloat16`, `float32`).',
    higher:
      'Not directly applicable. Higher precision (float32) is safer numerically but slower and heavier.',
    lower:
      'Not directly applicable. Lower precision is faster and lighter but may slightly shift scores.',
  },
  ppl_stride: {
    purpose:
      'Sliding-window stride for perplexity over long texts. Controls how much context windows overlap.',
    higher:
      'Larger stride runs faster (fewer windows) but can reduce scoring fidelity near window boundaries.',
    lower:
      'Smaller stride increases overlap and score stability, but costs more compute time.',
  },
  ppl_max_length: {
    purpose:
      'Maximum token length per perplexity window. Longer texts are chunked according to this limit.',
    higher:
      'Higher values preserve more context in each window, but increase memory usage and latency.',
    lower:
      'Lower values are lighter and safer, but may lose long-range context information.',
  },
  metric_readability: {
    purpose:
      'Enable readability metrics (`0` off, `1` on) to estimate how easy generated text is to read.',
    higher:
      'Not directly applicable. Enabled mode adds extra post-processing time.',
    lower:
      'Not directly applicable. Disabled mode is faster but provides no readability feedback.',
  },
  metric_bertscore: {
    purpose:
      'Enable BERTScore semantic similarity metric (`0` off, `1` on), often used to compare rewritten text against the source.',
    higher:
      'Not directly applicable. Enabled mode improves semantic quality visibility but is compute-heavy.',
    lower:
      'Not directly applicable. Disabled mode is faster but loses semantic similarity tracking.',
  },
  bertscore_model: {
    purpose:
      'Encoder model used for BERTScore (for example `roberta-large`).',
    higher:
      'Not directly applicable. Larger models can provide richer semantic signals but are slower.',
    lower:
      'Not directly applicable. Smaller models are cheaper but may miss nuanced semantic changes.',
  },
  bertscore_device: {
    purpose:
      'Execution device for BERTScore computation.',
    higher:
      'Not directly applicable. GPU greatly speeds up large-batch semantic scoring.',
    lower:
      'Not directly applicable. CPU scoring is slower but useful when GPU resources are occupied.',
  },
  bertscore_lang: {
    purpose:
      'Language code used by BERTScore tokenization defaults (for example `en`, `zh`).',
    higher:
      'Not directly applicable. Use the language matching your data to avoid tokenization mismatch.',
    lower:
      'Not directly applicable. Wrong language settings can bias semantic scores.',
  },
  bertscore_batch_size: {
    purpose:
      'Batch size for BERTScore embedding inference.',
    higher:
      'Higher batch size improves throughput, but quickly increases VRAM usage.',
    lower:
      'Lower batch size is memory-safe, but overall semantic scoring is slower.',
  },
  bertscore_rescale: {
    purpose:
      'Whether to apply baseline rescaling to BERTScore output (`0` off, `1` on).',
    higher:
      'Set to 1 when you want scores normalized to be more comparable across examples.',
    lower:
      'Set to 0 to keep raw BERTScore values; useful when you prefer unadjusted outputs.',
  },
  only_attack_machine: {
    purpose:
      'When enabled, attacks are applied only to records whose label equals `machine_label`.',
    higher:
      'Set to 1 to focus perturbations on machine-generated samples and reduce unnecessary attack cost.',
    lower:
      'Set to 0 to attack all samples, which is broader but may dilute machine-focused robustness analysis.',
  },
  machine_label: {
    purpose:
      'Numeric label value treated as machine-generated class when filtering attack targets (commonly `1`).',
    higher:
      'Higher values only help if your dataset actually uses that label for machine text.',
    lower:
      'Lower values only help if your dataset encodes machine text with a lower label (for example `0`).',
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
  enable_dataset_split: {
    purpose: '是否启用构建完成后的 train/dev/test 拆分输出。',
    higher: '不直接适用。启用后会在主输出旁新增拆分文件。',
    lower: '不直接适用。关闭后仅保留主输出文件。',
  },
  split_train_ratio: {
    purpose: 'train 划分权重（范围 0-10）。',
    higher: '值更高会让更多样本组分配到 train，训练数据更充足。',
    lower: '值更低会减少 train 覆盖；设为 0 表示不生成 train 划分文件。',
  },
  split_dev_ratio: {
    purpose: 'dev/验证集划分权重（范围 0-10）。',
    higher: '值更高会分配更多样本组到 dev，验证指标更稳定。',
    lower: '值更低会减少 dev 覆盖；设为 0 表示不生成 dev 划分文件。',
  },
  split_test_ratio: {
    purpose: 'test 划分权重（范围 0-10）。',
    higher: '值更高会分配更多样本组到 test，最终评估统计更稳定。',
    lower: '值更低会减少 test 覆盖；设为 0 表示不生成 test 划分文件。',
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
  seed: {
    purpose:
      '随机种子，用于控制可复现实验中的随机过程（如采样、打乱、初始化）。通常固定为同一个值以保证多次运行可对齐比较。',
    higher: '不直接适用。种子不是“越大越好”的调参项，不同数值只代表不同随机序列。',
    lower: '不直接适用。关键是是否固定一致，而不是数值大小。',
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
  sample_k: {
    purpose:
      '快速实验子采样数量。只使用 K 条样本而非全量数据，常用于调参和冒烟测试。',
    higher:
      'K 更大时结果更接近全量评估，但运行耗时和资源占用上升。',
    lower:
      'K 更小时调试更快，但指标波动会更大，不适合最终结论。',
  },
  api_model: {
    purpose:
      '发送给 OpenAI 兼容接口的模型名称（例如 `gpt-4o-mini` 或你部署服务的模型名）。',
    higher:
      '不直接适用。应按“效果/速度/成本”选择模型，而不是看字符串大小。',
    lower:
      '不直接适用。更小或更便宜的模型通常更快，但生成质量可能下降。',
  },
  api_key: {
    purpose:
      '调用 API 所需的鉴权密钥。请妥善保管，不要在日志或截图中暴露明文。',
    higher:
      '不直接适用。密钥长度不重要，关键是该密钥是否有效且权限足够。',
    lower:
      '不直接适用。为空或无效会导致 401/403 鉴权失败。',
  },
  api_base: {
    purpose:
      'OpenAI 兼容服务的基础地址。官方 OpenAI 通常可留空；使用 Azure/vLLM/代理网关时需填写对应地址。',
    higher:
      '不直接适用。关键是地址是否与服务协议兼容。',
    lower:
      '不直接适用。地址错误通常会触发连接失败或 404。',
  },
  api_endpoint: {
    purpose:
      '接口类型选择：`chat` 使用聊天格式，`completions` 使用传统补全格式。',
    higher:
      '不直接适用。请选择服务端实际支持的接口类型。',
    lower:
      '不直接适用。接口类型与服务不匹配时，常见报错是请求字段格式不兼容。',
  },
  api_timeout: {
    purpose:
      '每次 API 请求允许等待的最长秒数，超时后该请求会被终止。',
    higher:
      '超时更长可减少“误超时”，适合慢网络或长输出任务，但失败请求返回更慢。',
    lower:
      '超时更短可更快发现异常并返回，但长请求更容易被提前中断。',
  },
  metric_ppl: {
    purpose:
      '是否启用困惑度（PPL）质量指标（0=关闭，1=开启），用于评估文本语言流畅性。',
    higher:
      '不直接适用。开启后会增加一次额外模型打分，耗时上升。',
    lower:
      '不直接适用。关闭后运行更快，但缺少流畅性指标。',
  },
  ppl_model: {
    purpose:
      '用于计算 PPL 的评分模型（如 `gpt2`）。该模型只用于质量评估，不参与生成。',
    higher:
      '不直接适用。更大评分模型通常更敏感，但计算更重。',
    lower:
      '不直接适用。更小评分模型更快更省，但对细微质量差异不够敏感。',
  },
  ppl_device: {
    purpose:
      'PPL 评分使用的设备（如 `cuda:0` 或 `cpu`）。',
    higher:
      '不直接适用。GPU 通常更快，CPU 在显存紧张时更稳妥。',
    lower:
      '不直接适用。切到 CPU 会明显变慢，但可避免 GPU 资源争抢。',
  },
  ppl_dtype: {
    purpose:
      'PPL 评分模型的计算精度（`auto`、`float16`、`bfloat16`、`float32`）。',
    higher:
      '不直接适用。高精度（如 float32）更稳，但更慢、更占显存。',
    lower:
      '不直接适用。低精度更快更省，但分数可能有轻微偏移。',
  },
  ppl_stride: {
    purpose:
      '长文本 PPL 计算时滑窗步长。用于控制窗口重叠程度。',
    higher:
      '步长更大时窗口更少，速度更快，但边界位置分数可能不够稳定。',
    lower:
      '步长更小时窗口重叠更多，分数更稳，但计算开销更高。',
  },
  ppl_max_length: {
    purpose:
      '每个 PPL 窗口允许的最大 token 长度。',
    higher:
      '上限更高可保留更多上下文，但显存与时延都会上升。',
    lower:
      '上限更低更轻量，但可能损失长程上下文信息。',
  },
  metric_readability: {
    purpose:
      '是否启用可读性指标（0=关闭，1=开启），用于衡量文本易读程度。',
    higher:
      '不直接适用。开启后可读性分析更完整，但会增加后处理耗时。',
    lower:
      '不直接适用。关闭后更快，但无法得到可读性反馈。',
  },
  metric_bertscore: {
    purpose:
      '是否启用 BERTScore 语义相似度指标（0=关闭，1=开启），常用于比较改写前后语义保持。',
    higher:
      '不直接适用。开启后语义评估更全面，但计算成本较高。',
    lower:
      '不直接适用。关闭后运行更快，但缺少语义一致性指标。',
  },
  bertscore_model: {
    purpose:
      'BERTScore 使用的编码模型（如 `roberta-large`）。',
    higher:
      '不直接适用。更大模型语义表达更强，但速度更慢、资源占用更高。',
    lower:
      '不直接适用。更小模型更快，但语义分辨率可能下降。',
  },
  bertscore_device: {
    purpose:
      'BERTScore 计算设备。',
    higher:
      '不直接适用。GPU 能明显加速大批量语义评分。',
    lower:
      '不直接适用。CPU 更慢，但在 GPU 忙时更容易落地。',
  },
  bertscore_lang: {
    purpose:
      'BERTScore 语言代码（如 `en`、`zh`），用于匹配分词和语言设置。',
    higher:
      '不直接适用。应选择与文本语言一致的代码，避免评分偏差。',
    lower:
      '不直接适用。语言设置错误会导致相似度分数失真。',
  },
  bertscore_batch_size: {
    purpose:
      'BERTScore 推理批大小。',
    higher:
      '更大批大小可提升吞吐，但显存压力显著增加。',
    lower:
      '更小批大小更安全，但计算总时长更长。',
  },
  bertscore_rescale: {
    purpose:
      '是否对 BERTScore 结果做 baseline 重标定（0=关闭，1=开启）。',
    higher:
      '取 1 时分数更便于跨样本比较，适合做横向分析。',
    lower:
      '取 0 时保留原始分数，适合需要原值对比的场景。',
  },
  calibrator_name: {
    purpose:
      '校准算法名称（例如 `platt_lr` 或其他 runner 支持的校准器）。该选项决定如何把检测器原始分数映射为概率，并影响与已保存校准器 JSON 的兼容性。',
    higher:
      '不直接适用。这是算法类型选择，不是数值大小。不同算法会改变概率映射形状与稳定性。',
    lower:
      '不直接适用。应根据分数形态选择（单分数映射或多特征映射）。',
  },
  auto_calibrate: {
    purpose:
      '启用后，在未显式填写 `calibrator_path` 时，系统会按检测器/模型命名规则自动搜索并加载最匹配的校准器文件（含内置与用户目录）。',
    higher:
      '不直接适用。开启后配置更省事，但当目录中存在多个相近文件时可能加载到非预期校准器。',
    lower:
      '不直接适用。关闭后需要手动指定校准器路径，但结果来源更可控、可复现性更强。',
  },
  force_runner_calibration: {
    purpose:
      '强制在 evaluator/runner 层执行校准映射，即使检测器自身已输出概率也仍按 runner 逻辑重映射。适用于跨检测器对齐实验设置。',
    higher:
      '不直接适用。开启后可提高不同检测器之间的概率口径一致性，但可能覆盖检测器内置校准行为。',
    lower:
      '不直接适用。关闭后优先使用检测器原生概率输出与内部校准路径。',
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
  attack_dataset_only: {
    purpose:
      '构建模式开关：开启后只产出攻击数据集，不执行常规生成与质量评估流程。',
    higher:
      '设为 1 适合专门准备攻击数据集做鲁棒性评测，流程更聚焦。',
    lower:
      '设为 0 则运行完整 Build 流程（生成 + 可选质量指标 + 可选攻击）。',
  },
  only_attack_machine: {
    purpose:
      '是否只对机器标签样本执行攻击（1=只攻击机器样本，0=全部样本都可攻击）。',
    higher:
      '设为 1 可聚焦机器文本攻击，成本更低，也更贴合 ASR 场景。',
    lower:
      '设为 0 攻击覆盖更全面，但可能引入与目标无关的扰动样本。',
  },
  machine_label: {
    purpose:
      '当 `only_attack_machine=1` 时，指定“机器文本”的标签值（常见为 1）。',
    higher:
      '只有当数据集确实用更高标签表示机器文本时才应升高该值。',
    lower:
      '只有当数据集用更低标签表示机器文本时才应降低该值。',
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
