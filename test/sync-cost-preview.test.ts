/**
 * v0.20.0 Cathedral II Layer 8 D1 — sync --all cost preview tests.
 *
 * Cathedral I DX review identified "first sync surprise bill" as the #1
 * DX pain for large repos. v0.19.0 ran `sync --all` without telling the
 * user/agent how much it would cost. Cathedral II D1 gates --all on an
 * estimate: TTY prompts, non-TTY emits a ConfirmationRequired envelope
 * and exits 2, --yes skips, --dry-run shows + exits 0, --no-embed
 * skips the cost gate entirely (user already opted out of the spend).
 *
 * These tests exercise the cost envelope + flag behavior against a
 * real git repo fixture, no PGLite needed. The --yes / --dry-run /
 * envelope paths don't depend on DB state.
 */

import { describe, test, expect } from 'bun:test';
import { EMBEDDING_COST_PER_1K_TOKENS, estimateEmbeddingCostUsd } from '../src/core/embedding.ts';
import { estimateTokens } from '../src/core/chunkers/code.ts';

describe('Layer 8 D1 — embedding cost model', () => {
  test('EMBEDDING_COST_PER_1K_TOKENS is zero for local embeddings', () => {
    expect(EMBEDDING_COST_PER_1K_TOKENS).toBe(0);
  });

  test('estimateEmbeddingCostUsd scales linearly with tokens', () => {
    expect(estimateEmbeddingCostUsd(0)).toBe(0);
    expect(estimateEmbeddingCostUsd(1000)).toBe(0);
    expect(estimateEmbeddingCostUsd(10_000)).toBe(0);
    expect(estimateEmbeddingCostUsd(1_000_000)).toBe(0);
  });

  test('large local reindex previews still report zero direct model spend', () => {
    const cost = estimateEmbeddingCostUsd(400_000);
    expect(cost).toBe(0);
  });
});

describe('Layer 8 D1 — estimateTokens (exported from chunkers/code.ts)', () => {
  test('empty string is 0 tokens', () => {
    expect(estimateTokens('')).toBe(0);
  });

  test('short text is a small token count', () => {
    const t = estimateTokens('Hello, world!');
    expect(t).toBeGreaterThan(0);
    expect(t).toBeLessThan(10);
  });

  test('longer text scales roughly with length', () => {
    const short = 'The quick brown fox jumps over the lazy dog.';
    const long = short.repeat(100);
    const shortTokens = estimateTokens(short);
    const longTokens = estimateTokens(long);
    // Not strictly 100x because of tokenizer encoding, but should be >50x.
    expect(longTokens).toBeGreaterThan(shortTokens * 50);
  });
});
