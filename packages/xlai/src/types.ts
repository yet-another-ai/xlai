import type { FileSystemApi } from './filesystem';

export type ImageDetail = 'auto' | 'low' | 'high';

export type MediaSource =
  | {
      kind: 'url';
      url: string;
    }
  | ({
      kind: 'inline_data';
      mime_type: string;
    } & (
      | {
          data: string;
          /** @deprecated Use `data`; retained for request compatibility. */
          data_base64?: string;
        }
      | {
          /** @deprecated Use `data`; retained for request compatibility. */
          data_base64: string;
          data?: string;
        }
    ));

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

export interface AgentOptions extends ChatOptions {
  /**
   * Applies to **streaming** agent APIs when exposed by the WASM build. When `true` or omitted,
   * the stream can run multiple model rounds with tool execution until the last response has no
   * tool calls. When `false`, the stream stops after one model turn (no tool execution).
   * Unary `agent()` / `prompt` use a single model call regardless of this flag.
   */
  agentLoop?: boolean;
}

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

/** Wire shape from Rust `QtsModelFileEntry`. */
export interface QtsModelFileEntry {
  logical_name: string;
  filename: string;
  sha256?: string;
  size_bytes?: number;
  url?: string;
}

/** Wire shape from Rust `QtsModelManifest`. */
export interface QtsModelManifest {
  schema_version: number;
  model_id: string;
  revision: string;
  files: QtsModelFileEntry[];
}

/** Optional local QTS on chat/agent sessions (passed to WASM `RuntimeBuilder`). */
export interface QtsSessionConfig {
  manifest?: QtsModelManifest;
}

export interface ChatSessionOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  maxOutputTokens?: number;
  fileSystem?: FileSystemApi;
  /** When set, the WASM runtime behind the session includes local QTS. */
  qts?: QtsSessionConfig;
}

export interface AgentSessionOptions extends ChatSessionOptions {
  /**
   * Same semantics as {@link AgentOptions.agentLoop} for sessions (streaming only when supported).
   */
  agentLoop?: boolean;
}

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
  | ({
      type: 'audio_delta';
    } & (
      | {
          data: string;
          /** @deprecated Use `data`; retained for request compatibility. */
          data_base64?: string;
        }
      | {
          /** @deprecated Use `data`; retained for request compatibility. */
          data_base64: string;
          data?: string;
        }
    ))
  | { type: 'finished'; response: TtsResponse };
