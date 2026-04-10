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

/** Matches Rust `ChatRetryPolicy` JSON (camelCase via serde on WASM). */
export interface ChatRetryPolicy {
  enabled?: boolean;
  maxRetries?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
}

export type ReasoningEffort = 'low' | 'medium' | 'high';

export interface ChatOptions {
  prompt?: string;
  content?: ChatContent;
  systemPrompt?: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
  reasoningEffort?: ReasoningEffort;
  retryPolicy?: ChatRetryPolicy;
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
    metadata?: Record<string, unknown>;
  };
  finishReason: ChatFinishReason;
  usage?: ChatUsage;
  metadata?: Record<string, unknown>;
}

/** Matches Rust `ChatMessage` JSON (snake_case fields from serde). */
export type ChatMessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface ChatMessage {
  role: ChatMessageRole;
  content: ChatContent;
  tool_name?: string | null;
  tool_call_id?: string | null;
  metadata?: Record<string, unknown>;
}

/**
 * One event from `AgentSession.streamPrompt` / `streamPromptWithContent` (WASM JSON).
 * Intermediate agent-loop assistant rounds are surfaced as `thinking` so consumers can render
 * them separately from the terminal assistant reply.
 */
export type ChatExecutionEvent =
  | { kind: 'model'; data: unknown }
  | { kind: 'thinking'; data: ChatResponse }
  | { kind: 'toolCall'; data: unknown }
  | { kind: 'toolResult'; data: unknown };

/** Async hook before each streamed agent-loop model call (see README). */
export type AgentContextCompressor = (
  messages: ChatMessage[],
  estimatedInputTokens: number | null,
) => Promise<ChatMessage[]>;

/**
 * Async hook before every agent model call; returned text is merged into the ephemeral system
 * reminder. `messages` is the user transcript only (internal reminder rows are never passed in).
 */
export type AgentSystemReminder = (
  messages: ChatMessage[],
) => Promise<string> | string;

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

export type ToolSchema =
  | {
      type: 'string';
      description?: string;
    }
  | {
      type: 'number';
      description?: string;
    }
  | {
      type: 'integer';
      description?: string;
    }
  | {
      type: 'boolean';
      description?: string;
    }
  | {
      type: 'array';
      items?: ToolSchema;
      description?: string;
    }
  | {
      type: 'object';
      properties?: Record<string, ToolSchema>;
      required?: string[];
      additionalProperties?: boolean;
      description?: string;
    };

/** Matches `xlai_core::ToolCallExecutionMode` JSON (`Concurrent` | `Sequential`). */
export type ToolCallExecutionMode = 'concurrent' | 'sequential';

export interface ToolDefinition {
  name: string;
  description: string;
  /**
   * Preferred recursive schema shape for tool arguments.
   * Legacy callers may continue using `parameters` during the migration window.
   */
  inputSchema?: ToolSchema;
  /** Legacy flat top-level tool parameters. */
  parameters?: ToolParameter[];
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
  reasoningEffort?: ReasoningEffort;
  retryPolicy?: ChatRetryPolicy;
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

export type ImageGenerationQuality = 'low' | 'medium' | 'high';

export type ImageGenerationBackground = 'transparent' | 'opaque';

export type ImageGenerationOutputFormat = 'png' | 'jpeg' | 'webp';

/** Matches `xlai_core::GeneratedImage` JSON. */
export interface GeneratedImage {
  image: MediaSource;
  mime_type?: string | null;
  revised_prompt?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ImageGenerationOptions {
  prompt: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  imageModel?: string;
  size?: string;
  quality?: ImageGenerationQuality;
  background?: ImageGenerationBackground;
  outputFormat?: ImageGenerationOutputFormat;
  count?: number;
}

/** Matches `xlai_core::ImageGenerationResponse` JSON. */
export interface ImageGenerationResponse {
  images: GeneratedImage[];
  metadata?: Record<string, unknown>;
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
