import type { FileSystemApi, WasmFileSystemBridge } from './types';

export function toWasmFileSystemBridge(
  fileSystem: FileSystemApi,
): WasmFileSystemBridge {
  return {
    read: async (path) => fileSystem.read(path),
    exists: async (path) => fileSystem.exists(path),
    write: async (path, data) => fileSystem.write(path, data),
    createDirAll: async (path) => fileSystem.createDirAll(path),
    list: async (path) => fileSystem.list(path),
    deletePath: async (path) => fileSystem.deletePath(path),
  };
}
