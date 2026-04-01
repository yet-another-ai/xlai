import { getWasmModule, initXlai } from '../wasm';

import {
  type FileSystemEntry,
  type FileSystemApi,
  type WasmMemoryFileSystemConstructor,
  type WasmMemoryFileSystemInstance,
  toUint8Array,
} from './types';

function wasmMemoryFileSystemConstructor(): WasmMemoryFileSystemConstructor {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      MemoryFileSystem: WasmMemoryFileSystemConstructor;
    }
  ).MemoryFileSystem;
}

export class MemoryFileSystem implements FileSystemApi {
  private constructor(private readonly inner: WasmMemoryFileSystemInstance) {}

  static async create(): Promise<MemoryFileSystem> {
    await initXlai();
    return new MemoryFileSystem(new (wasmMemoryFileSystemConstructor())());
  }

  async read(path: string): Promise<Uint8Array> {
    return this.inner.read(path);
  }

  async exists(path: string): Promise<boolean> {
    return this.inner.exists(path);
  }

  async write(
    path: string,
    data: Uint8Array | ArrayBuffer | ArrayBufferView,
  ): Promise<void> {
    return this.inner.write(path, toUint8Array(data));
  }

  async createDirAll(path: string): Promise<void> {
    return this.inner.createDirAll(path);
  }

  async list(path: string): Promise<FileSystemEntry[]> {
    return this.inner.list(path);
  }

  async deletePath(path: string): Promise<void> {
    return this.inner.deletePath(path);
  }

  toWasmFileSystem(): WasmMemoryFileSystemInstance {
    return this.inner;
  }
}

export async function createMemoryFileSystem(): Promise<MemoryFileSystem> {
  return MemoryFileSystem.create();
}
