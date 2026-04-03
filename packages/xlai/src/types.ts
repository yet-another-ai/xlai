import type { FileSystemApi } from './filesystem';

export type ImageDetail = 'auto' | 'low' | 'high';

export type MediaSource =
  | {
      kind: 'url';
      url: string;
    }
  | {
      kind: 'inline_data';
      mime_type: string;
      data_base64: string;
    };

export type ContentPart =
  | {
      type: 'text';
      text: string;
    }
  | {
      type: 'image';
      source: MediaSource;
      mime_type?: string;
      detail?: ImageDetail | null;
    }
  | {
      type: 'audio';
      source: MediaSource;
      mime_type?: string;
    }
  | {
      type: 'file';
      source: MediaSource;
      mime_type?: string;
      filename?: string;
    };

export type ChatContent =
  | string
  | {
      parts: ContentPart[];
    };

export interface ChatOptions {
  prompt?: string;
  content?: ChatContent;
  systemPrompt?: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
}

export type AgentOptions = ChatOptions;

export interface ChatUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export type ChatFinishReason =
  | 'completed'
  | 'tool_calls'
  | 'length'
  | 'stopped';

export interface ChatResponse {
  message: {
    role: 'assistant';
    content: ChatContent;
  };
  finishReason: ChatFinishReason;
  usage?: ChatUsage;
}

export type ToolParameterType =
  | 'string'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'array'
  | 'object';

export interface ToolParameter {
  name: string;
  description: string;
  kind: ToolParameterType;
  required: boolean;
}

/** Matches `xlai_core::ToolCallExecutionMode` JSON (`Concurrent` | `Sequential`). */
export type ToolCallExecutionMode = 'concurrent' | 'sequential';

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: ToolParameter[];
  /**
   * When any tool in a model turn is `sequential`, all tool calls in that turn
   * run one after another in model order (no overlap).
   */
  executionMode?: ToolCallExecutionMode;
}

export interface ToolResult {
  toolName?: string;
  content: string;
  isError?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ChatSessionOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  maxOutputTokens?: number;
  fileSystem?: FileSystemApi;
}

export type AgentSessionOptions = ChatSessionOptions;

/** Matches `xlai_core::VoiceReferenceSample` JSON. */
export interface VoiceReferenceSample {
  audio: MediaSource;
  mime_type?: string | null;
  transcript?: string | null;
  weight?: number | null;
  metadata?: Record<string, unknown>;
}

/** Matches `xlai_core::VoiceSpec` JSON (`kind` + snake_case variants). */
export type VoiceSpec =
  | { kind: 'preset'; name: string }
  | { kind: 'provider_ref'; id: string; provider?: string | null }
  | { kind: 'clone'; references: VoiceReferenceSample[] };

/** Matches `xlai_core::TtsAudioFormat` JSON. */
export type TtsAudioFormatWire =
  | 'mp3'
  | 'opus'
  | 'aac'
  | 'flac'
  | 'wav'
  | 'pcm';

export interface TtsOptions {
  input: string;
  voice: VoiceSpec;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  ttsModel?: string;
  responseFormat?: TtsAudioFormatWire;
  speed?: number;
  instructions?: string;
  /** When omitted, unary `tts` defaults to `unary`; `ttsStream` defaults to `stream`. */
  delivery?: 'unary' | 'stream';
}

/** Matches `xlai_core::TtsResponse` JSON. */
export interface TtsResponse {
  audio: MediaSource;
  mime_type: string;
  metadata?: Record<string, unknown>;
}

/** Matches `xlai_core::TtsChunk` JSON (`type` tag, snake_case). */
export type TtsChunk =
  | { type: 'started'; mime_type: string; metadata?: Record<string, unknown> }
  | { type: 'audio_delta'; data_base64: string }
  | { type: 'finished'; response: TtsResponse };
