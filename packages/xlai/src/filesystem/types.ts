export type FileSystemEntryKind = 'file' | 'directory';

export interface FileSystemEntry {
  path: string;
  kind: FileSystemEntryKind;
}

export interface FileSystemApi {
  read(path: string): Promise<Uint8Array>;
  exists(path: string): Promise<boolean>;
  write(
    path: string,
    data: Uint8Array | ArrayBuffer | ArrayBufferView,
  ): Promise<void>;
  createDirAll(path: string): Promise<void>;
  list(path: string): Promise<FileSystemEntry[]>;
  deletePath(path: string): Promise<void>;
}

export type WasmMemoryFileSystemInstance = {
  read: (path: string) => Promise<Uint8Array>;
  exists: (path: string) => Promise<boolean>;
  write: (path: string, data: Uint8Array) => Promise<void>;
  createDirAll: (path: string) => Promise<void>;
  list: (path: string) => Promise<FileSystemEntry[]>;
  deletePath: (path: string) => Promise<void>;
};

export type WasmMemoryFileSystemConstructor =
  new () => WasmMemoryFileSystemInstance;

export type WasmFileSystemBridge = {
  read: (path: string) => Promise<Uint8Array>;
  exists: (path: string) => Promise<boolean>;
  write: (path: string, data: Uint8Array) => Promise<void>;
  createDirAll: (path: string) => Promise<void>;
  list: (path: string) => Promise<FileSystemEntry[]>;
  deletePath: (path: string) => Promise<void>;
};

export function toUint8Array(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
): Uint8Array {
  if (data instanceof Uint8Array) {
    return data;
  }

  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }

  return new Uint8Array(data);
}
