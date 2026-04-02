import { AgentSession } from '../agent';
import { ChatSession } from '../chat';
import type { FileSystemApi } from '../filesystem';
import { MemoryFileSystem } from '../filesystem';
import { toWasmFileSystemBridge } from '../filesystem/bridge';
import type { WasmToolSessionInstance } from '../shared';
import { getWasmModule, initXlai } from '../wasm';
import { createXlaiTransformersJsAdapter } from './adapter';
import type {
  TransformersAgentSessionOptions,
  TransformersChatSessionOptions,
  TransformersWasmModule,
} from './types';

function transformersWasm(): TransformersWasmModule {
  return getWasmModule() as ReturnType<typeof getWasmModule> &
    TransformersWasmModule;
}

function buildWasmOptions(
  options: TransformersChatSessionOptions,
  adapter: object,
) {
  return {
    modelId: options.modelId,
    adapter,
    systemPrompt: options.systemPrompt,
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
  };
}

export async function createTransformersChatSession(
  options: TransformersChatSessionOptions,
): Promise<ChatSession> {
  await initXlai();
  const adapter = await createXlaiTransformersJsAdapter(options.adapterOptions);
  const wasm = transformersWasm();
  const payload = buildWasmOptions(options, adapter);

  const inner: WasmToolSessionInstance =
    options.fileSystem === undefined
      ? (wasm.createTransformersChatSession(payload) as WasmToolSessionInstance)
      : options.fileSystem instanceof MemoryFileSystem
        ? (wasm.createTransformersChatSessionWithMemoryFileSystem(
            payload,
            options.fileSystem.toWasmFileSystem(),
          ) as WasmToolSessionInstance)
        : (wasm.createTransformersChatSessionWithFileSystem(
            payload,
            toWasmFileSystemBridge(options.fileSystem as FileSystemApi),
          ) as WasmToolSessionInstance);

  return ChatSession.fromWasm(inner);
}

export async function createTransformersAgentSession(
  options: TransformersAgentSessionOptions,
): Promise<AgentSession> {
  await initXlai();
  const adapter = await createXlaiTransformersJsAdapter(options.adapterOptions);
  const wasm = transformersWasm();
  const payload = buildWasmOptions(options, adapter);

  const inner: WasmToolSessionInstance =
    options.fileSystem === undefined
      ? (wasm.createTransformersAgentSession(
          payload,
        ) as WasmToolSessionInstance)
      : options.fileSystem instanceof MemoryFileSystem
        ? (wasm.createTransformersAgentSessionWithMemoryFileSystem(
            payload,
            options.fileSystem.toWasmFileSystem(),
          ) as WasmToolSessionInstance)
        : (wasm.createTransformersAgentSessionWithFileSystem(
            payload,
            toWasmFileSystemBridge(options.fileSystem as FileSystemApi),
          ) as WasmToolSessionInstance);

  return AgentSession.fromWasm(inner);
}
