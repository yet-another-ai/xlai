import type { Grammar, TransformersTokenizer } from 'transformers-llguidance';
import type {
  TransformersEmbedRequest,
  TransformersEmbedResponse,
  TransformersGenerateRequest,
  TransformersGenerateResponse,
  TransformersJsAdapterOptions,
} from './types';

export type XlaiTransformersJsAdapter = {
  generate: (
    request: TransformersGenerateRequest,
  ) => Promise<TransformersGenerateResponse>;
  embed: (
    request: TransformersEmbedRequest,
  ) => Promise<TransformersEmbedResponse>;
};

/**
 * Maps the JSON payload produced by `xlai-backend-transformersjs` to a `transformers-llguidance` grammar.
 * Exported for unit tests.
 */
export function rustPayloadToGrammar(
  grammar: unknown | undefined,
  toolSchema: unknown | undefined,
): Grammar | undefined {
  if (toolSchema !== undefined && toolSchema !== null) {
    return {
      type: 'json_schema',
      schema: toolSchema as Record<string, unknown>,
    };
  }
  if (grammar === undefined || grammar === null) {
    return undefined;
  }
  const g = grammar as Record<string, unknown>;
  const ty = g.type;
  if (ty === 'json_schema') {
    return {
      type: 'json_schema',
      schema: g.schema as Record<string, unknown>,
    };
  }
  if (ty === 'lark') {
    return {
      type: 'lark',
      grammar: String(g.grammar),
      startSymbol: (g.startSymbol as string | undefined) ?? 'start',
    };
  }
  throw new Error(`unsupported grammar payload: ${String(ty)}`);
}

/**
 * Browser adapter: loads `@huggingface/transformers` and `transformers-llguidance`, runs
 * `text-generation` with optional constrained decoding (per request).
 */
export async function createXlaiTransformersJsAdapter(
  options: TransformersJsAdapterOptions = {},
): Promise<XlaiTransformersJsAdapter> {
  const [tlg, tf] = await Promise.all([
    import('transformers-llguidance'),
    import('@huggingface/transformers'),
  ]);

  const { GuidanceParser, GuidanceLogitsProcessor, extractTokenizerData } = tlg;
  const { LogitsProcessor, StoppingCriteria, StoppingCriteriaList, Tensor } =
    tf;
  const pipelineFactory = tf.pipeline as (
    task: string,
    modelId: string,
  ) => Promise<unknown>;

  /** Text-generation pipeline (typed loosely — `pipeline()` overload union is huge). */
  type TextGenPipeline = {
    tokenizer: { encode: (text: string, options?: unknown) => unknown };
    (prompt: string, options?: Record<string, unknown>): Promise<unknown>;
  };
  type FeatureExtractionPipeline = (
    input: string | string[],
    options?: Record<string, unknown>,
  ) => Promise<{
    tolist: () => number[] | number[][] | number[][][];
  }>;

  const pipelines = new Map<string, TextGenPipeline>();
  const embeddingPipelines = new Map<string, FeatureExtractionPipeline>();

  async function getPipeline(modelId: string): Promise<TextGenPipeline> {
    let p = pipelines.get(modelId);
    if (!p) {
      p = (await pipelineFactory(
        'text-generation',
        modelId,
      )) as TextGenPipeline;
      pipelines.set(modelId, p);
    }
    return p;
  }

  async function getEmbeddingPipeline(
    modelId: string,
  ): Promise<FeatureExtractionPipeline> {
    let p = embeddingPipelines.get(modelId);
    if (!p) {
      p = (await pipelineFactory(
        'feature-extraction',
        modelId,
      )) as FeatureExtractionPipeline;
      embeddingPipelines.set(modelId, p);
    }
    return p;
  }

  class XlaiGuidanceLogitsProcessor extends LogitsProcessor {
    private lastCommitted = 0;
    private generatedIds: number[] = [];
    private sawPrefill = false;
    readonly guidance: InstanceType<typeof GuidanceLogitsProcessor>;

    constructor(guidance: InstanceType<typeof GuidanceLogitsProcessor>) {
      super();
      this.guidance = guidance;
    }

    _call(
      inputIds: bigint[][],
      logits: InstanceType<typeof Tensor>,
    ): InstanceType<typeof Tensor> {
      const ids = inputIds[0]!.map((x) => Number(x));
      if (!this.sawPrefill) {
        this.lastCommitted = ids.length;
        this.sawPrefill = true;
      }
      while (this.lastCommitted < ids.length) {
        const tid = ids[this.lastCommitted]!;
        this.guidance.onToken(tid);
        this.generatedIds.push(tid);
        this.lastCommitted++;
      }

      if (this.guidance.canStop()) {
        return logits;
      }

      const logitsData = logits.data as Float32Array;
      const vocabSize = logits.dims[logits.dims.length - 1]!;
      const batchLogits = logitsData.slice(0, vocabSize);
      const processed = this.guidance.process(this.generatedIds, batchLogits);

      const out = new Float32Array(logitsData.length);
      out.set(processed.subarray(0, vocabSize), 0);
      if (logitsData.length > vocabSize) {
        out.set(logitsData.subarray(vocabSize), vocabSize);
      }
      return new Tensor(logits.type, out, logits.dims);
    }
  }

  class XlaiStoppingCriteria extends StoppingCriteria {
    readonly guidance: InstanceType<typeof GuidanceLogitsProcessor>;

    constructor(guidance: InstanceType<typeof GuidanceLogitsProcessor>) {
      super();
      this.guidance = guidance;
    }

    _call(inputIds: number[][]): boolean[] {
      return inputIds.map(() => this.guidance.canStop());
    }
  }

  return {
    async generate(
      request: TransformersGenerateRequest,
    ): Promise<TransformersGenerateResponse> {
      const grammar = rustPayloadToGrammar(request.grammar, request.toolSchema);
      const pipeline = await getPipeline(request.model);

      const genOpts: Record<string, unknown> = {
        max_new_tokens: request.maxNewTokens,
        temperature: request.temperature,
        do_sample: request.temperature > 0,
      };

      let logitsProcessor: XlaiGuidanceLogitsProcessor | undefined;
      if (grammar) {
        const tokenizerData = extractTokenizerData(
          pipeline.tokenizer as TransformersTokenizer,
        );
        const parser = await GuidanceParser.create(grammar, tokenizerData);
        const guidanceCore = new GuidanceLogitsProcessor(parser, {
          speculationDepth: options.speculationDepth ?? 5,
        });
        logitsProcessor = new XlaiGuidanceLogitsProcessor(guidanceCore);
        genOpts.logits_processor = [logitsProcessor];
        const stoppingCriteria = new StoppingCriteriaList();
        stoppingCriteria.push(new XlaiStoppingCriteria(guidanceCore));
        genOpts.stopping_criteria = stoppingCriteria;
      }

      const raw = await pipeline(request.prompt, genOpts);
      const row = (Array.isArray(raw) ? raw[0] : raw) as {
        generated_text?: string;
      };
      const full = row.generated_text ?? '';
      const continuation = full.startsWith(request.prompt)
        ? full.slice(request.prompt.length)
        : full;

      let finishReason = 'completed';
      if (logitsProcessor?.guidance.canStop()) {
        finishReason = 'stopped';
      }

      return {
        text: continuation.replace(/^\s+/, ''),
        finishReason,
      };
    },
    async embed(
      request: TransformersEmbedRequest,
    ): Promise<TransformersEmbedResponse> {
      const pipeline = await getEmbeddingPipeline(request.model);
      const raw = await pipeline(request.inputs, {
        pooling: 'mean',
        normalize: true,
      });
      const list = raw.tolist();
      const vectors = Array.isArray(list[0])
        ? (list as number[][])
        : [list as number[]];
      return { vectors };
    },
  };
}
