import type { FileSystemApi, FileSystemEntry } from './types';
import { toUint8Array } from './types';

function normalizeFileSystemPath(path: string): string {
  if (!path.startsWith('/')) {
    throw new Error(`filesystem paths must start with '/': ${path}`);
  }

  const segments: string[] = [];
  for (const segment of path.split('/')) {
    if (segment === '' || segment === '.') {
      continue;
    }

    if (segment === '..') {
      throw new Error(`filesystem paths must not contain '..': ${path}`);
    }

    segments.push(segment);
  }

  return segments.length === 0 ? '/' : `/${segments.join('/')}`;
}

function splitFileSystemPath(path: string): string[] {
  const normalizedPath = normalizeFileSystemPath(path);
  return normalizedPath === '/'
    ? []
    : normalizedPath
        .slice(1)
        .split('/')
        .filter((segment) => segment.length > 0);
}

async function getOpfsRootDirectory(): Promise<FileSystemDirectoryHandle> {
  const storage = globalThis.navigator?.storage;
  if (storage === undefined || typeof storage.getDirectory !== 'function') {
    throw new Error('OPFS is not available in this environment');
  }

  return storage.getDirectory();
}

async function resolveOpfsDirectory(
  root: FileSystemDirectoryHandle,
  path: string,
  options: {
    create?: boolean;
  } = {},
): Promise<FileSystemDirectoryHandle> {
  let current = root;
  for (const segment of splitFileSystemPath(path)) {
    current = await current.getDirectoryHandle(segment, {
      create: options.create ?? false,
    });
  }
  return current;
}

async function resolveOpfsParentDirectory(
  root: FileSystemDirectoryHandle,
  path: string,
): Promise<{
  parent: FileSystemDirectoryHandle;
  name: string | null;
}> {
  const segments = splitFileSystemPath(path);
  const name = segments.pop() ?? null;
  let current = root;
  for (const segment of segments) {
    current = await current.getDirectoryHandle(segment, { create: false });
  }

  return {
    parent: current,
    name,
  };
}

function iterateDirectoryHandles(
  directory: FileSystemDirectoryHandle,
): AsyncIterable<[string, FileSystemHandle]> {
  return (
    directory as FileSystemDirectoryHandle & {
      entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
    }
  ).entries();
}

export class OpfsFileSystem implements FileSystemApi {
  private constructor(private readonly root: FileSystemDirectoryHandle) {}

  static async create(): Promise<OpfsFileSystem> {
    return new OpfsFileSystem(await getOpfsRootDirectory());
  }

  async read(path: string): Promise<Uint8Array> {
    const normalizedPath = normalizeFileSystemPath(path);
    const { parent, name } = await resolveOpfsParentDirectory(
      this.root,
      normalizedPath,
    );
    if (name === null) {
      throw new Error('cannot read the filesystem root');
    }

    const handle = await parent.getFileHandle(name, { create: false });
    const file = await handle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  }

  async exists(path: string): Promise<boolean> {
    const normalizedPath = normalizeFileSystemPath(path);
    if (normalizedPath === '/') {
      return true;
    }

    let parent: FileSystemDirectoryHandle;
    let name: string | null;
    try {
      const resolved = await resolveOpfsParentDirectory(
        this.root,
        normalizedPath,
      );
      parent = resolved.parent;
      name = resolved.name;
    } catch {
      return false;
    }

    if (name === null) {
      return true;
    }

    try {
      await parent.getFileHandle(name, { create: false });
      return true;
    } catch {
      try {
        await parent.getDirectoryHandle(name, { create: false });
        return true;
      } catch {
        return false;
      }
    }
  }

  async write(
    path: string,
    data: Uint8Array | ArrayBuffer | ArrayBufferView,
  ): Promise<void> {
    const normalizedPath = normalizeFileSystemPath(path);
    const { parent, name } = await resolveOpfsParentDirectory(
      this.root,
      normalizedPath,
    );
    if (name === null) {
      throw new Error('cannot write data to the filesystem root');
    }

    const handle = await parent.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    try {
      const bytes = toUint8Array(data);
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      await writable.write(copy);
    } finally {
      await writable.close();
    }
  }

  async createDirAll(path: string): Promise<void> {
    await resolveOpfsDirectory(this.root, path, { create: true });
  }

  async list(path: string): Promise<FileSystemEntry[]> {
    const normalizedPath = normalizeFileSystemPath(path);
    const directory = await resolveOpfsDirectory(this.root, normalizedPath, {
      create: false,
    });

    const entries: FileSystemEntry[] = [];
    for await (const [name, handle] of iterateDirectoryHandles(directory)) {
      const entryPath =
        normalizedPath === '/' ? `/${name}` : `${normalizedPath}/${name}`;
      entries.push({
        path: entryPath,
        kind: handle.kind === 'directory' ? 'directory' : 'file',
      });
    }

    return entries.sort((left, right) => left.path.localeCompare(right.path));
  }

  async deletePath(path: string): Promise<void> {
    const normalizedPath = normalizeFileSystemPath(path);
    if (normalizedPath === '/') {
      for await (const [name] of iterateDirectoryHandles(this.root)) {
        await this.root.removeEntry(name, { recursive: true });
      }
      return;
    }

    const { parent, name } = await resolveOpfsParentDirectory(
      this.root,
      normalizedPath,
    );
    if (name === null) {
      return;
    }

    await parent.removeEntry(name, { recursive: true });
  }
}

export async function createOpfsFileSystem(): Promise<OpfsFileSystem> {
  return OpfsFileSystem.create();
}
