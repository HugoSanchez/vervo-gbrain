/**
 * Local Embedding Service
 *
 * Uses node-llama-cpp to run bge-m3 locally via GGUF.
 * Model is lazy-loaded on first call and unloaded after an idle timeout.
 */

import { existsSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

const MODEL = 'bge-m3';
const DIMENSIONS = 1024;
const MAX_CHARS = 8000;
const BATCH_SIZE = 100;
const IDLE_TIMEOUT_MS = 5 * 60 * 1000;

const DEFAULT_MODEL_DIR = join(homedir(), '.gbrain', 'models');
const MODEL_FILENAME = 'bge-m3-f16.gguf';

let llama: any = null;
let model: any = null;
let context: any = null;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
let loading: Promise<void> | null = null;
let modelPathOverride: string | null = null;

function getModelPath(): string {
  if (modelPathOverride) return modelPathOverride;
  const envPath = process.env.GBRAIN_EMBEDDING_MODEL;
  if (envPath) return envPath;
  return join(DEFAULT_MODEL_DIR, MODEL_FILENAME);
}

function resetIdleTimer(): void {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    dispose().catch(() => {});
  }, IDLE_TIMEOUT_MS);
}

async function ensureLoaded(): Promise<void> {
  if (context) {
    resetIdleTimer();
    return;
  }

  if (loading) {
    await loading;
    return;
  }

  loading = (async () => {
    const modelPath = getModelPath();
    if (!existsSync(modelPath)) {
      throw new Error(
        `Embedding model not found at ${modelPath}. ` +
        `Download ${MODEL_FILENAME} to ${DEFAULT_MODEL_DIR}/ ` +
        `or set GBRAIN_EMBEDDING_MODEL to the full path.`,
      );
    }

    const { getLlama } = await import('node-llama-cpp');
    llama = await getLlama();
    model = await llama.loadModel({ modelPath });
    context = await model.createEmbeddingContext();
    resetIdleTimer();
  })();

  try {
    await loading;
  } finally {
    loading = null;
  }
}

export function setModelPath(path: string | null): void {
  modelPathOverride = path;
}

export function isAvailable(): boolean {
  return existsSync(getModelPath());
}

export function isLoaded(): boolean {
  return context != null;
}

export async function dispose(): Promise<void> {
  if (idleTimer) {
    clearTimeout(idleTimer);
    idleTimer = null;
  }

  if (context) {
    await context.dispose();
    context = null;
  }

  if (model) {
    await model.dispose();
    model = null;
  }

  llama = null;
}

export async function embed(text: string): Promise<Float32Array> {
  const result = await embedBatch([text]);
  return result[0];
}

export interface EmbedBatchOptions {
  /**
   * Optional callback fired after each 100-item sub-batch completes.
   * CLI wrappers tick a reporter; Minion handlers can call
   * job.updateProgress here instead of hooking the per-page callback.
   */
  onBatchComplete?: (done: number, total: number) => void;
}

export async function embedBatch(
  texts: string[],
  options: EmbedBatchOptions = {},
): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  if (truncated.length === 0) return results;

  await ensureLoaded();

  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await Promise.all(
      batch.map(async text => {
        const embedding = await context.getEmbeddingFor(text);
        return new Float32Array(embedding.vector);
      }),
    );
    results.push(...batchResults);
    options.onBatchComplete?.(results.length, truncated.length);
  }

  return results;
}

export { MODEL as EMBEDDING_MODEL, DIMENSIONS as EMBEDDING_DIMENSIONS };

/**
 * Local embeddings do not incur a per-request API bill, so cost previews
 * currently report zero direct model spend.
 */
export const EMBEDDING_COST_PER_1K_TOKENS = 0;

/** Compute USD cost estimate for embedding `tokens` at current model rate. */
export function estimateEmbeddingCostUsd(tokens: number): number {
  return (tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS;
}
