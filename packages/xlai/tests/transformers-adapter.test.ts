import { afterEach, describe, expect, it, vi } from 'vitest';

const mockState = vi.hoisted(() => {
  class MockTensor {
    type: string;
    data: Float32Array;
    dims: number[];

    constructor(type: string, data: Float32Array, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }

  class MockLogitsProcessor {}
  class MockStoppingCriteria {}
  class MockStoppingCriteriaList {
    criteria: unknown[] = [];
    push(item: unknown) {
      this.criteria.push(item);
    }
  }

  const tokenizer = {
    encode: vi.fn(() => ({
      input_ids: { dims: [1, 3], data: [10, 11, 12] },
    })),
  };

  const pipelineCalls: Array<{
    prompt: string;
    options?: Record<string, unknown>;
  }> = [];
  const pipelineFactory = vi.fn(async (_task: string, modelId: string) => {
    const runner = vi.fn(
      async (prompt: string, options?: Record<string, unknown>) => {
        pipelineCalls.push({ prompt, options });
        const logitsProcessors = options?.logits_processor as
          | Array<{
              _call: (
                inputIds: bigint[][],
                logits: InstanceType<typeof MockTensor>,
              ) => InstanceType<typeof MockTensor>;
            }>
          | undefined;
        if (logitsProcessors?.[0]) {
          const logits = new Float32Array([0, 1, 2, 3]);
          logitsProcessors[0]._call(
            [[10n, 11n, 12n]],
            new MockTensor('float32', logits, [1, 4]),
          );
          logitsProcessors[0]._call(
            [[10n, 11n, 12n, 99n]],
            new MockTensor('float32', logits, [1, 4]),
          );
          logitsProcessors[0]._call(
            [[10n, 11n, 12n, 99n]],
            new MockTensor('float32', logits, [1, 4]),
          );
        }

        if (modelId === 'model-a') {
          return [{ generated_text: `${prompt}  completion` }];
        }

        return [{ generated_text: `${prompt}{"answer":1}` }];
      },
    );

    return Object.assign(runner, { tokenizer });
  });

  const extractTokenizerData = vi.fn(() => ({
    vocab: { '{': 1, a: 2, '0': 3 },
    merges: [],
  }));
  const parserCreate = vi.fn(
    async (grammar: unknown, tokenizerData: unknown) => ({
      grammar,
      tokenizerData,
      advance: vi.fn(),
    }),
  );
  const guidanceInstances: Array<{
    process: ReturnType<typeof vi.fn>;
    onToken: ReturnType<typeof vi.fn>;
    canStop: ReturnType<typeof vi.fn>;
  }> = [];

  const GuidanceLogitsProcessor = vi.fn(function MockGuidanceLogitsProcessor() {
    const instance = {
      process: vi.fn((_ids: number[], logits: Float32Array) => logits),
      onToken: vi.fn(),
      canStop: vi
        .fn()
        .mockReturnValueOnce(false)
        .mockReturnValueOnce(false)
        .mockReturnValue(true)
        .mockReturnValue(true),
    };
    guidanceInstances.push(instance);
    return instance;
  });

  return {
    MockTensor,
    MockLogitsProcessor,
    MockStoppingCriteria,
    MockStoppingCriteriaList,
    tokenizer,
    pipelineCalls,
    pipelineFactory,
    extractTokenizerData,
    parserCreate,
    guidanceInstances,
    GuidanceLogitsProcessor,
  };
});

vi.mock('@huggingface/transformers', () => ({
  LogitsProcessor: mockState.MockLogitsProcessor,
  StoppingCriteria: mockState.MockStoppingCriteria,
  StoppingCriteriaList: mockState.MockStoppingCriteriaList,
  Tensor: mockState.MockTensor,
  pipeline: mockState.pipelineFactory,
}));

vi.mock('transformers-llguidance', () => ({
  GuidanceParser: {
    create: mockState.parserCreate,
  },
  GuidanceLogitsProcessor: mockState.GuidanceLogitsProcessor,
  extractTokenizerData: mockState.extractTokenizerData,
}));

import {
  createXlaiTransformersJsAdapter,
  rustPayloadToGrammar,
} from '../src/transformers';

describe('transformers adapter', () => {
  afterEach(() => {
    mockState.pipelineCalls.length = 0;
    mockState.guidanceInstances.length = 0;
    mockState.pipelineFactory.mockClear();
    mockState.tokenizer.encode.mockClear();
    mockState.extractTokenizerData.mockClear();
    mockState.parserCreate.mockClear();
    mockState.GuidanceLogitsProcessor.mockClear();
    vi.clearAllMocks();
  });

  it('generates plain text without constrained decoding and caches pipelines per model', async () => {
    const adapter = await createXlaiTransformersJsAdapter();

    await expect(
      adapter.generate({
        prompt: 'Hello',
        model: 'model-a',
        temperature: 0.25,
        maxNewTokens: 16,
      }),
    ).resolves.toEqual({
      text: 'completion',
      finishReason: 'completed',
    });

    await adapter.generate({
      prompt: 'Again',
      model: 'model-a',
      temperature: 0,
      maxNewTokens: 8,
    });

    expect(mockState.pipelineFactory).toHaveBeenCalledTimes(1);
    expect(mockState.extractTokenizerData).not.toHaveBeenCalled();
    expect(mockState.parserCreate).not.toHaveBeenCalled();
    expect(mockState.tokenizer.encode).not.toHaveBeenCalled();
    expect(mockState.pipelineCalls[0]?.options).toMatchObject({
      max_new_tokens: 16,
      temperature: 0.25,
      do_sample: true,
    });
    expect(mockState.pipelineCalls[1]?.options).toMatchObject({
      do_sample: false,
    });
  });

  it('builds constrained decoding with llguidance for grammar requests', async () => {
    const adapter = await createXlaiTransformersJsAdapter({
      speculationDepth: 7,
    });

    const grammar = rustPayloadToGrammar(
      {
        type: 'lark',
        grammar: 'start: "a"',
        startSymbol: 'start',
      },
      undefined,
    );

    await expect(
      adapter.generate({
        prompt: 'Return JSON:',
        model: 'model-b',
        temperature: 0.4,
        maxNewTokens: 32,
        grammar,
      }),
    ).resolves.toEqual({
      text: '{"answer":1}',
      finishReason: 'stopped',
    });

    expect(mockState.extractTokenizerData).toHaveBeenCalledWith(
      mockState.tokenizer,
    );
    expect(mockState.parserCreate).toHaveBeenCalledWith(grammar, {
      vocab: { '{': 1, a: 2, '0': 3 },
      merges: [],
    });
    expect(mockState.GuidanceLogitsProcessor).toHaveBeenCalledWith(
      expect.any(Object),
      { speculationDepth: 7 },
    );
    expect(mockState.pipelineCalls[0]?.options?.logits_processor).toHaveLength(
      1,
    );
    expect(
      mockState.pipelineCalls[0]?.options?.stopping_criteria,
    ).toBeDefined();

    const instance = mockState.guidanceInstances[0];
    expect(instance).toBeDefined();
    expect(instance?.onToken).toHaveBeenCalledWith(99);
    expect(instance?.process).toHaveBeenCalledWith(
      [99],
      expect.any(Float32Array),
    );
    expect(instance?.canStop).toHaveBeenCalled();
    const parser = await mockState.parserCreate.mock.results[0]?.value;
    expect(parser.advance).not.toHaveBeenCalled();
  });
});
