import { describe, test, expect } from 'bun:test';
import { mkdtempSync, readFileSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  EMBEDDING_MODEL_FILENAME,
  getEmbeddingModelStatus,
  installEmbeddingModel,
  resolveEmbeddingModelPath,
} from '../src/core/model-artifacts.ts';

const CLI_CWD = new URL('..', import.meta.url).pathname;

function withEnv<T>(vars: Record<string, string | undefined>, fn: () => T): T {
  const original = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(vars)) {
    original.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of original.entries()) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

async function withEnvAsync<T>(vars: Record<string, string | undefined>, fn: () => Promise<T>): Promise<T> {
  const original = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(vars)) {
    original.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return await fn();
  } finally {
    for (const [key, value] of original.entries()) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

describe('model artifacts', () => {
  test('resolveEmbeddingModelPath honors GBRAIN_HOME for the default path', () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-model-home-'));
    const resolved = withEnv({
      GBRAIN_HOME: home,
      GBRAIN_EMBEDDING_MODEL: undefined,
    }, () => resolveEmbeddingModelPath());
    expect(resolved).toBe(join(home, '.gbrain', 'models', EMBEDDING_MODEL_FILENAME));
  });

  test('getEmbeddingModelStatus reflects env overrides and install source', () => {
    const status = withEnv({
      GBRAIN_EMBEDDING_MODEL: '/tmp/custom-model.gguf',
      GBRAIN_EMBEDDING_MODEL_SOURCE: 'https://example.com/model.gguf',
    }, () => getEmbeddingModelStatus());
    expect(status.path).toBe('/tmp/custom-model.gguf');
    expect(status.configuredBy).toBe('env');
    expect(status.installSource).toBe('https://example.com/model.gguf');
    expect(status.installed).toBe(false);
  });

  test('installEmbeddingModel copies a local artifact into the managed path', async () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-model-install-'));
    const sourceDir = mkdtempSync(join(tmpdir(), 'gbrain-model-source-'));
    const sourcePath = join(sourceDir, EMBEDDING_MODEL_FILENAME);
    writeFileSync(sourcePath, 'test-model-bytes');

    const result = await withEnvAsync({
      GBRAIN_HOME: home,
      GBRAIN_EMBEDDING_MODEL: undefined,
    }, () => installEmbeddingModel(sourcePath));

    expect(result.path).toBe(join(home, '.gbrain', 'models', EMBEDDING_MODEL_FILENAME));
    expect(readFileSync(result.path, 'utf8')).toBe('test-model-bytes');
    expect(result.replaced).toBe(false);
    expect(result.bytesWritten).toBe('test-model-bytes'.length);
  });

  test('installEmbeddingModel refuses overwrite unless --force is used', async () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-model-force-'));
    const sourceDir = mkdtempSync(join(tmpdir(), 'gbrain-model-force-source-'));
    const firstSource = join(sourceDir, 'first.gguf');
    const secondSource = join(sourceDir, 'second.gguf');
    writeFileSync(firstSource, 'first');
    writeFileSync(secondSource, 'second');

    await withEnvAsync({
      GBRAIN_HOME: home,
      GBRAIN_EMBEDDING_MODEL: undefined,
    }, () => installEmbeddingModel(firstSource));

    await expect(withEnvAsync({
      GBRAIN_HOME: home,
      GBRAIN_EMBEDDING_MODEL: undefined,
    }, () => installEmbeddingModel(secondSource))).rejects.toThrow('--force');

    const replaced = await withEnvAsync({
      GBRAIN_HOME: home,
      GBRAIN_EMBEDDING_MODEL: undefined,
    }, () => installEmbeddingModel(secondSource, { force: true }));

    expect(replaced.replaced).toBe(true);
    expect(readFileSync(replaced.path, 'utf8')).toBe('second');
  });
});

describe('gbrain models CLI', () => {
  test('models status/path/install work end-to-end with a local source file', async () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-models-cli-home-'));
    const sourceDir = mkdtempSync(join(tmpdir(), 'gbrain-models-cli-source-'));
    const sourcePath = join(sourceDir, EMBEDDING_MODEL_FILENAME);
    writeFileSync(sourcePath, 'cli-model-bytes');

    const install = Bun.spawnSync({
      cmd: ['bun', 'run', 'src/cli.ts', 'models', 'install', '--source', sourcePath, '--json'],
      cwd: CLI_CWD,
      env: {
        ...process.env,
        GBRAIN_HOME: home,
        GBRAIN_EMBEDDING_MODEL: '',
      },
    });
    expect(install.exitCode).toBe(0);
    const installJson = JSON.parse(new TextDecoder().decode(install.stdout));
    expect(installJson.status).toBe('success');
    expect(readFileSync(installJson.path, 'utf8')).toBe('cli-model-bytes');

    const status = Bun.spawnSync({
      cmd: ['bun', 'run', 'src/cli.ts', 'models', 'status', '--json'],
      cwd: CLI_CWD,
      env: {
        ...process.env,
        GBRAIN_HOME: home,
        GBRAIN_EMBEDDING_MODEL: '',
      },
    });
    expect(status.exitCode).toBe(0);
    const statusJson = JSON.parse(new TextDecoder().decode(status.stdout));
    expect(statusJson.installed).toBe(true);
    expect(statusJson.path).toBe(installJson.path);

    const pathResult = Bun.spawnSync({
      cmd: ['bun', 'run', 'src/cli.ts', 'models', 'path', '--json'],
      cwd: CLI_CWD,
      env: {
        ...process.env,
        GBRAIN_HOME: home,
        GBRAIN_EMBEDDING_MODEL: '',
      },
    });
    expect(pathResult.exitCode).toBe(0);
    const pathJson = JSON.parse(new TextDecoder().decode(pathResult.stdout));
    expect(pathJson.path).toBe(installJson.path);
  });

  test('doctor --fast reports a missing embedding model with the new install hint', async () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-models-doctor-home-'));
    const result = Bun.spawnSync({
      cmd: ['bun', 'run', 'src/cli.ts', 'doctor', '--fast', '--json'],
      cwd: CLI_CWD,
      env: {
        ...process.env,
        GBRAIN_HOME: home,
        GBRAIN_EMBEDDING_MODEL: '',
      },
    });
    expect(result.exitCode).toBe(0);
    const payload = JSON.parse(new TextDecoder().decode(result.stdout));
    const check = payload.checks.find((entry: any) => entry.name === 'embedding_model');
    expect(check).toBeTruthy();
    expect(check.status).toBe('warn');
    expect(check.message).toContain('gbrain models install');
  });
});
