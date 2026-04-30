import { stderr, stdout } from 'process';
import {
  EMBEDDING_MODEL_FILENAME,
  formatEmbeddingModelInstallHint,
  getEmbeddingModelInstallSource,
  getEmbeddingModelStatus,
  installEmbeddingModel,
} from '../core/model-artifacts.ts';

function parseFlag(args: string[], flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

function hasFlag(args: string[], flag: string): boolean {
  return args.includes(flag);
}

function formatBytes(bytes: number | null): string {
  if (bytes == null) return 'unknown';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export async function runModels(args: string[]): Promise<void> {
  const sub = args[0];
  if (!sub || sub === '--help' || sub === '-h') {
    printHelp();
    return;
  }

  switch (sub) {
    case 'path':
      runPath(args.slice(1));
      return;
    case 'status':
      runStatus(args.slice(1));
      return;
    case 'install':
      await runInstall(args.slice(1));
      return;
    default:
      console.error(`gbrain models: unknown subcommand "${sub}"`);
      printHelp();
      process.exit(2);
  }
}

function printHelp(): void {
  console.log(`gbrain models — local model provisioning

USAGE
  gbrain models path [--json]
  gbrain models status [--json]
  gbrain models install [--source <path-or-url>] [--path <dest>] [--force] [--json]

NOTES
  - The embedding model filename is ${EMBEDDING_MODEL_FILENAME}
  - If --source is omitted, gbrain uses GBRAIN_EMBEDDING_MODEL_SOURCE
  - If --path is omitted, gbrain uses GBRAIN_EMBEDDING_MODEL or the default local path
`);
}

function runPath(args: string[]): void {
  const jsonOutput = hasFlag(args, '--json');
  const status = getEmbeddingModelStatus();
  if (jsonOutput) {
    console.log(JSON.stringify({ path: status.path }));
    return;
  }
  console.log(status.path);
}

function runStatus(args: string[]): void {
  const jsonOutput = hasFlag(args, '--json');
  const status = getEmbeddingModelStatus();
  if (jsonOutput) {
    console.log(JSON.stringify(status));
    return;
  }

  console.log(`Embedding model: ${status.installed ? 'installed' : 'missing'}`);
  console.log(`Path: ${status.path}`);
  console.log(`Configured by: ${status.configuredBy}`);
  if (status.sizeBytes != null) {
    console.log(`Size: ${formatBytes(status.sizeBytes)}`);
  }
  if (status.installSource) {
    console.log(`Install source: ${status.installSource}`);
  }
  if (!status.installed) {
    console.log(formatEmbeddingModelInstallHint(status));
  }
}

async function runInstall(args: string[]): Promise<void> {
  if (hasFlag(args, '--help') || hasFlag(args, '-h')) {
    printHelp();
    return;
  }

  const jsonOutput = hasFlag(args, '--json');
  const force = hasFlag(args, '--force');
  const source = parseFlag(args, '--source') ?? getEmbeddingModelInstallSource();
  const destination = parseFlag(args, '--path') ?? null;

  if (!source) {
    const status = getEmbeddingModelStatus(destination);
    const msg = formatEmbeddingModelInstallHint(status);
    if (jsonOutput) {
      console.log(JSON.stringify({ status: 'error', reason: 'missing_source', message: msg }));
    } else {
      console.error(msg);
    }
    process.exit(1);
  }

  let lastProgress = 0;
  const result = await installEmbeddingModel(source, {
    destinationPath: destination,
    force,
    onProgress: (downloaded, total) => {
      if (jsonOutput) return;
      if (total && total > 0) {
        const pct = Math.floor((downloaded / total) * 100);
        if (pct !== lastProgress && (pct === 100 || pct - lastProgress >= 10)) {
          stderr.write(`[models] download ${pct}% (${formatBytes(downloaded)} / ${formatBytes(total)})\n`);
          lastProgress = pct;
        }
      } else if (downloaded > 0 && downloaded - lastProgress >= 25 * 1024 * 1024) {
        stderr.write(`[models] downloaded ${formatBytes(downloaded)}\n`);
        lastProgress = downloaded;
      }
    },
  });

  if (jsonOutput) {
    console.log(JSON.stringify({
      status: 'success',
      path: result.path,
      source: result.source,
      replaced: result.replaced,
      bytes_written: result.bytesWritten,
    }));
    return;
  }

  stdout.write(`Installed ${EMBEDDING_MODEL_FILENAME} to ${result.path}\n`);
  stdout.write(`Source: ${result.source}\n`);
  stdout.write(`Bytes: ${formatBytes(result.bytesWritten)}\n`);
  if (result.replaced) {
    stdout.write('Replaced existing model.\n');
  }
}
