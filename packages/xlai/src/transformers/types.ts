import type { FileSystemApi, MemoryFileSystem } from '../filesystem';
import type { ChatContent, ChatResponse } from '../types';

/** Options forwarded to {@link import('./adapter').createXlaiTransformersJsAdapter}. */
export type TransformersJsAdapterOptions = {
  /** Passed to `transformers-llguidance` logits processor (default 5). */
  speculationDepth?: number;
};

export type TransformersChatSessionOptions = {
  modelId: string;
  systemPrompt?: string;
  temperature?: number;
  maxOutputTokens?: number;
  fileSystem?: MemoryFileSystem | FileSystemApi;
  adapterOptions?: TransformersJsAdapterOptions;
};

export type TransformersAgentSessionOptions = TransformersChatSessionOptions;

/** Request shape sent from Rust (`xlai-backend-transformersjs`) to the JS adapter. */
export type TransformersGenerateRequest = {
  prompt: string;
  model: string;
  temperature: number;
  maxNewTokens: number;
  grammar?: unknown;
  toolSchema?: unknown;
};

export type TransformersGenerateResponse = {
  text: string;
  finishReason?: string;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
  };
};

export type TransformersChatFunction = (
  options: TransformersGenerateRequest,
) => Promise<TransformersGenerateResponse>;

export type TransformersWasmModule = {
  createTransformersChatSession: (options: unknown) => unknown;
  createTransformersChatSessionWithMemoryFileSystem: (
    options: unknown,
    fs: unknown,
  ) => unknown;
  createTransformersChatSessionWithFileSystem: (
    options: unknown,
    fs: unknown,
  ) => unknown;
  createTransformersAgentSession: (options: unknown) => unknown;
  createTransformersAgentSessionWithMemoryFileSystem: (
    options: unknown,
    fs: unknown,
  ) => unknown;
  createTransformersAgentSessionWithFileSystem: (
    options: unknown,
    fs: unknown,
  ) => unknown;
};

export type { ChatContent, ChatResponse };
