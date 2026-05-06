// lib/inference-sparse.ts — forward pass with content-dependent top-K sparse
// attention (the SubQ / Native-Sparse-Attention family).
//
// Built as a sibling to lib/inference.ts so the dense playground keeps
// working unmodified. Most ops (matmul, layerNorm, softmax, relu) are
// reused from lib/inference.ts. The only thing that changes is the per-head
// attention forward, which masks all but the top-K scores per query before
// softmax.

import type { Matrix, Weights, Block } from "./weights";
import {
  matmul,
  layerNorm,
  softmax,
  relu,
  addInPlace,
} from "./inference";

/** Single causal head with optional top-K sparse masking.
 *
 *  topK <= 0 or topK >= T disables sparsification (acts identically to the
 *  dense head in lib/inference.ts).
 *
 *  When 0 < topK < T, for each query position we keep only the topK largest
 *  attention scores and mask the rest to -Infinity before softmax. Selection
 *  is content-dependent: which positions get kept depends on q·k similarity. */
export function headForwardSparse(
  x: Float32Array,
  T: number,
  C: number,
  qW: Matrix,
  kW: Matrix,
  vW: Matrix,
  topK: number
): Float32Array {
  const headSize = qW.rows;
  const q = matmul(x, T, C, qW);
  const k = matmul(x, T, C, kW);
  const v = matmul(x, T, C, vW);

  // Compute scaled dot-product scores with causal mask.
  const scores = new Float32Array(T * T);
  const scale = 1 / Math.sqrt(headSize);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      if (j > i) {
        scores[i * T + j] = -Infinity;
      } else {
        let s = 0;
        const qi = i * headSize;
        const kj = j * headSize;
        for (let h = 0; h < headSize; h++) s += q[qi + h] * k[kj + h];
        scores[i * T + j] = s * scale;
      }
    }
  }

  // Top-K sparse mask: per row, keep only the topK largest scores.
  if (topK > 0 && topK < T) {
    for (let i = 0; i < T; i++) {
      const off = i * T;
      // Position i can only attend to positions 0..i (after the causal mask
      // above, j>i is already -Infinity). The valid candidate count is i+1.
      const valid = i + 1;
      if (valid <= topK) continue; // keep everything when there's <= K options

      // Find the topK-th largest score among the valid positions to use as
      // a threshold. Simple O(valid * topK) selection — fine at T <= 128.
      const thresh = nthLargest(scores, off, valid, topK);
      for (let j = 0; j < valid; j++) {
        if (scores[off + j] < thresh) scores[off + j] = -Infinity;
      }
    }
  }

  const w = softmax(scores, T, T);

  // Weighted sum of values.
  const out = new Float32Array(T * headSize);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      const wij = w[i * T + j];
      if (wij === 0) continue;
      const vj = j * headSize;
      const oi = i * headSize;
      for (let h = 0; h < headSize; h++) out[oi + h] += wij * v[vj + h];
    }
  }
  return out;
}

/** Return the value v such that exactly `n` entries in scores[off..off+len)
 *  are >= v (i.e. v is the n-th largest). Uses a small heap-free approach:
 *  copy the slice, partial-sort by repeated max extraction. n is small (≤ T)
 *  so this is fine. */
function nthLargest(
  scores: Float32Array,
  off: number,
  len: number,
  n: number
): number {
  // Copy candidates into a fresh array we can mutate.
  const buf = new Float32Array(len);
  for (let i = 0; i < len; i++) buf[i] = scores[off + i];
  let kth = -Infinity;
  for (let pick = 0; pick < n; pick++) {
    let maxI = 0;
    let maxV = -Infinity;
    for (let i = 0; i < len; i++) {
      if (buf[i] > maxV) {
        maxV = buf[i];
        maxI = i;
      }
    }
    kth = maxV;
    buf[maxI] = -Infinity;
  }
  return kth;
}

/** Multi-head attention with sparse top-K. Mirrors multiHeadAttention in
 *  lib/inference.ts but routes through headForwardSparse. */
export function multiHeadAttentionSparse(
  x: Float32Array,
  T: number,
  C: number,
  heads: { q: Matrix; k: Matrix; v: Matrix }[],
  proj: { weight: Matrix; bias: Float32Array },
  topK: number
): Float32Array {
  const headSize = heads[0].q.rows;
  const nHead = heads.length;
  const concat = new Float32Array(T * C);
  for (let h = 0; h < nHead; h++) {
    const out = headForwardSparse(x, T, C, heads[h].q, heads[h].k, heads[h].v, topK);
    for (let t = 0; t < T; t++) {
      for (let s = 0; s < headSize; s++) {
        concat[t * C + h * headSize + s] = out[t * headSize + s];
      }
    }
  }
  return matmul(concat, T, C, proj.weight, proj.bias);
}

export function blockForwardSparse(
  x: Float32Array,
  T: number,
  C: number,
  blk: Block,
  topK: number
): Float32Array {
  const ln1 = layerNorm(x, T, C, blk.ln1.gain, blk.ln1.bias);
  const attnOut = multiHeadAttentionSparse(ln1, T, C, blk.heads, blk.proj, topK);
  addInPlace(x, attnOut);

  const ln2 = layerNorm(x, T, C, blk.ln2.gain, blk.ln2.bias);
  const hidden = matmul(ln2, T, C, blk.mlp.fc1, blk.mlp.b1);
  relu(hidden);
  const mlpOut = matmul(hidden, T, blk.mlp.fc1.rows, blk.mlp.fc2, blk.mlp.b2);
  addInPlace(x, mlpOut);
  return x;
}

/** Full forward pass with sparse attention everywhere. Returns logits for the
 *  last position only. */
export function forwardSparse(
  weights: Weights,
  ids: number[],
  topK: number
): Float32Array {
  const T = ids.length;
  const C = weights.meta.n_embd;

  const x = new Float32Array(T * C);
  const tok = weights.tok_emb.data;
  const pos = weights.pos_emb.data;
  for (let t = 0; t < T; t++) {
    const tokRow = ids[t] * C;
    const posRow = t * C;
    const xRow = t * C;
    for (let c = 0; c < C; c++) {
      x[xRow + c] = tok[tokRow + c] + pos[posRow + c];
    }
  }

  let h: Float32Array = x;
  for (const blk of weights.blocks) {
    h = blockForwardSparse(h, T, C, blk, topK);
  }

  h = layerNorm(h, T, C, weights.final_ln.gain, weights.final_ln.bias);

  const lastRow = (T - 1) * C;
  const lastH = h.slice(lastRow, lastRow + C);
  return matmul(lastH, 1, C, weights.lm_head.weight, weights.lm_head.bias);
}
