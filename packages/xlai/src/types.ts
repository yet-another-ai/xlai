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
