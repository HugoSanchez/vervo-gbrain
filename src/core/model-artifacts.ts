import { copyFile, mkdir, rename, rm, stat } from 'fs/promises';
import { existsSync, statSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { configDir } from './config.ts';

export const EMBEDDING_MODEL_NAME = 'bge-m3';
export const EMBEDDING_MODEL_FILENAME = 'bge-m3-f16.gguf';

type ModelPathSource = 'override' | 'env' | 'default';

export interface EmbeddingModelStatus {
  path: string;
  installed: boolean;
  configuredBy: ModelPathSource;
  sizeBytes: number | null;
  installSource: string | null;
}

export interface InstallEmbeddingModelOptions {
  destinationPath?: string | null;
  force?: boolean;
  onProgress?: (downloadedBytes: number, totalBytes: number | null) => void;
}

export interface InstallEmbeddingModelResult {
  path: string;
  replaced: boolean;
  bytesWritten: number;
  source: string;
}

export function defaultEmbeddingModelDir(): string {
  return join(configDir(), 'models');
}

export function defaultEmbeddingModelPath(): string {
  return join(defaultEmbeddingModelDir(), EMBEDDING_MODEL_FILENAME);
}

export function getEmbeddingModelInstallSource(): string | null {
  const source = process.env.GBRAIN_EMBEDDING_MODEL_SOURCE?.trim();
  return source ? source : null;
}

export function resolveEmbeddingModelPath(overridePath: string | null = null): string {
  if (overridePath && overridePath.trim()) return overridePath.trim();
  const envPath = process.env.GBRAIN_EMBEDDING_MODEL?.trim();
  if (envPath) return envPath;
  return defaultEmbeddingModelPath();
}

function resolveConfiguredBy(overridePath: string | null = null): ModelPathSource {
  if (overridePath && overridePath.trim()) return 'override';
  return process.env.GBRAIN_EMBEDDING_MODEL?.trim() ? 'env' : 'default';
}

export function getEmbeddingModelStatus(overridePath: string | null = null): EmbeddingModelStatus {
  const path = resolveEmbeddingModelPath(overridePath);
  let sizeBytes: number | null = null;
  if (existsSync(path)) {
    try {
      sizeBytes = Number(statSync(path).size);
    } catch {
      sizeBytes = null;
    }
  }

  return {
    path,
    installed: existsSync(path),
    configuredBy: resolveConfiguredBy(overridePath),
    sizeBytes,
    installSource: getEmbeddingModelInstallSource(),
  };
}

export function formatEmbeddingModelInstallHint(status: EmbeddingModelStatus): string {
  const installCommand = status.installSource
    ? 'gbrain models install'
    : 'gbrain models install --source <path-or-url>';
  return `Run: ${installCommand} or set GBRAIN_EMBEDDING_MODEL to an existing ${EMBEDDING_MODEL_FILENAME} path.`;
}

function isHttpSource(source: string): boolean {
  try {
    const url = new URL(source);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

function isFileUrl(source: string): boolean {
  try {
    return new URL(source).protocol === 'file:';
  } catch {
    return false;
  }
}

function resolveLocalSourcePath(source: string): string {
  if (isFileUrl(source)) return fileURLToPath(source);
  return resolve(source);
}

async function copyLocalSource(
  sourcePath: string,
  tempPath: string,
  onProgress?: (downloadedBytes: number, totalBytes: number | null) => void,
): Promise<number> {
  const info = await stat(sourcePath);
  await copyFile(sourcePath, tempPath);
  onProgress?.(info.size, info.size);
  return info.size;
}

async function downloadRemoteSource(
  source: string,
  tempPath: string,
  onProgress?: (downloadedBytes: number, totalBytes: number | null) => void,
): Promise<number> {
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Embedding model download failed: HTTP ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('Embedding model download failed: empty response body');
  }

  const totalHeader = response.headers.get('content-length');
  const totalBytes = totalHeader ? Number(totalHeader) : null;
  const writer = Bun.file(tempPath).writer();
  const reader = response.body.getReader();
  let downloaded = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      await writer.write(value);
      downloaded += value.byteLength;
      onProgress?.(downloaded, totalBytes);
    }
  } finally {
    await writer.end();
  }

  return downloaded;
}

export async function installEmbeddingModel(
  source: string,
  options: InstallEmbeddingModelOptions = {},
): Promise<InstallEmbeddingModelResult> {
  const trimmedSource = source.trim();
  if (!trimmedSource) {
    throw new Error(
      `Embedding model install requires a source. Pass --source <path-or-url> or set GBRAIN_EMBEDDING_MODEL_SOURCE.`,
    );
  }

  const destinationPath = resolveEmbeddingModelPath(options.destinationPath ?? null);
  const replaced = existsSync(destinationPath);
  if (replaced && !options.force) {
    throw new Error(`Embedding model already exists at ${destinationPath}. Re-run with --force to replace it.`);
  }

  const destinationDir = dirname(destinationPath);
  await mkdir(destinationDir, { recursive: true });
  const tempPath = `${destinationPath}.download-${process.pid}-${Date.now()}`;

  try {
    let bytesWritten = 0;
    if (isHttpSource(trimmedSource)) {
      bytesWritten = await downloadRemoteSource(trimmedSource, tempPath, options.onProgress);
    } else {
      const localPath = resolveLocalSourcePath(trimmedSource);
      if (resolve(localPath) === resolve(destinationPath) && existsSync(destinationPath)) {
        const info = await stat(destinationPath);
        options.onProgress?.(info.size, info.size);
        return {
          path: destinationPath,
          replaced,
          bytesWritten: info.size,
          source: trimmedSource,
        };
      }
      bytesWritten = await copyLocalSource(localPath, tempPath, options.onProgress);
    }

    await rename(tempPath, destinationPath);
    return {
      path: destinationPath,
      replaced,
      bytesWritten,
      source: trimmedSource,
    };
  } catch (error) {
    await rm(tempPath, { force: true });
    throw error;
  }
}
