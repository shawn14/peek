# Browser LLM Playground — Design

**Date:** 2026-05-03
**Status:** Approved
**Repos touched:** `tiny-llm`, `peek`

## Goal

Add a "Now you try" page to peek.school where any visitor can type a prompt and watch the trained 825K-param model generate text **locally in the browser**. No server, no API. Existing JSON weight files are extended; a TypeScript port of the forward pass runs on the main thread.

This is the realization of the "Live inference in the browser" item from the "What's next" section of `/process`. Pays off the entire 9-step walkthrough by letting the reader operate the model whose math they just learned.

## Non-goals

- Not a general-purpose chat UI.
- Not WebGPU / WebAssembly accelerated. Plain JS matmul is fast enough at this scale (~5–10 ms/token).
- No streaming over network, no server inference fallback.
- No top-k or nucleus sampling — temperature only. Top-k stays in "What's next."
- No analytics, no auth, no rate limiting.

## Architecture

**Where it runs:** entirely in the browser. Static Next.js page; weights ship as JSON in `/public/data/`.

**Forward pass:** pure-TypeScript port of `train.py`'s forward pass. Embed → 4 transformer blocks (each: LN → multi-head attention → residual → LN → MLP → residual) → final LN → lm_head → softmax → sample.

**Threading:** main thread, with a `requestAnimationFrame` yield between tokens. Web Worker rejected — ~150 lines of plumbing for no perceivable benefit at 5–10 ms/token.

**Weight loading:** one-time fetch on page mount. Browser caches subsequent visits.

## UX / page layout

`/playground` becomes Step 10 in the home-page walkthrough.

Top to bottom:

1. **Chapter header** — `ChapterHeader` component, num "10", title "Now you try". One-paragraph framing: "Everything you've read up to now, doing its job in real time. The 825K weights you trained are loaded into your browser and run on every keystroke. There is no server."
2. **Prompt input** — single-line text field, max length 64 chars (model's `BLOCK_SIZE` is 128, so capping the prompt at 64 reserves at least 64 chars of headroom for generation before the sliding window kicks in). Default `"ROMEO:"`.
3. **Controls row:**
   - **Temperature slider** — range 0 to 2, step 0.1, default 1.0. Live label below the slider changes by zone:
     - 0.0–0.3 — "always picks the safest letter"
     - 0.3–0.8 — "playing it cool"
     - 0.8–1.2 — "honest about what it knows"
     - 1.2–2.0 — "getting reckless"
   - **Length input** — number, 1 to 400, default 200.
4. **Generate / Stop button** — toggles label and behavior while running.
5. **Output area** — fixed-height monospace block. Prompt rendered in muted color; generated chars in normal weight, appended as they stream in. Cursor block at the tail while generating.
6. **"What just happened?"** — collapsed by default. When opened, shows the most recent token's top-5 probabilities and the temperature-adjusted distribution side by side.
7. **Loading state** — first visit shows "Loading the kid (~600 KB)" with progress; cached on subsequent visits.

## Data flow

### On mount
```
useEffect fetches the 6 JSON files in parallel
  → parses into Float32Arrays (~30 ms)
  → stores in useRef so re-renders don't re-allocate
  → status: "ready"
```

### On Generate click
```
loop runs N times (N = length input):
  context = (prompt + accumulated_output).slice(-BLOCK_SIZE)
  logits = forward(context)        // last position only
  probs  = softmax(logits / temperature)
  next   = sample(probs)
  accumulated_output += vocab[next]
  await requestAnimationFrame
  setState(accumulated_output)     // triggers re-render
```

Cancellation: a `cancelled` ref is checked at the top of each iteration.

## Files

### In `tiny-llm` repo

**Modified: `export_for_web.py`**

Add one new export section that dumps every transformer block's weights:

```
blocks.json — {
  final_ln: { gain: [128], bias: [128] },   // ln_f, applied after the last block
  blocks: [
    {                            // for each of 4 blocks
      ln1: { gain: [128], bias: [128] },
      heads: [                   // 4 heads per block, head_size = 32
        { q: [32×128], k: [32×128], v: [32×128] }   // PyTorch Linear weight shape
      ],
      proj: { weight: [128×128], bias: [128] },     // head concat → output projection
      ln2: { gain: [128], bias: [128] },
      mlp: { fc1: [512×128], b1: [512], fc2: [128×512], b2: [128] }
    }
  ]
}
```

Approx raw size: 2.4 MB; gzipped ~600 KB.

**Files fetched on page load (6):**
- `vocab.json` — char list (existing)
- `meta.json` — architecture summary; used to validate JSON shapes match the runtime model (existing)
- `tok_emb.json` — token embedding matrix (existing)
- `pos_emb.json` — position embedding matrix (existing)
- `lm_head.json` — final projection weight + bias (existing)
- `blocks.json` — all transformer-block weights + final LayerNorm (new)

### In `peek` repo

**New: `app/playground/page.tsx`** — React page with state management, controls, output rendering.

**New: `lib/inference.ts`** — pure TypeScript forward pass. No React imports. Exports:
- `forward(weights: Weights, ids: number[]): Float32Array` — returns logits for the last position
- `sample(probs: Float32Array): number` — returns a token id
- The `sample` function body is left for the user to implement (5–10 line teaching moment).

**New: `lib/weights.ts`** — fetches all JSON files, parses into `Float32Array`s, returns a typed `Weights` object.

**Modified: `app/page.tsx`** — append Step 10 entry to STEPS array.

**Modified: `components/Nav.tsx`** — add Playground link.

**Modified: `app/process/page.tsx`** — remove the "Live inference in the browser" bullet from "What's next" (it's now built).

## Edge cases

- **Empty prompt** — Generate button disabled until ≥1 char.
- **Prompt > BLOCK_SIZE (128)** — sliding window: only feed the last 128 chars of `prompt + accumulated`.
- **Out-of-vocab characters** in prompt — strip them, show inline note ("removed N unsupported chars").
- **Stop mid-generation** — cancellation flag checked each iteration; loop bails cleanly.
- **Weight fetch failure** — show retry button and one-line error.
- **Slow connection** — fetch progress indicator with byte count.

## Verification

No automated tests; the site has no test infrastructure.

**Numerical correctness check** (manual, one-time):
1. Open the playground with prompt `"ROMEO"`, temperature 0 (forces argmax).
2. Compare the first-token logits against `predictions.json` (which contains the ground-truth Python output).
3. Match to 4 decimal places confirms the TypeScript port is correct.

If the check fails, the bug is almost certainly in `lib/inference.ts` — most likely a transposed matrix, a missing residual connection, or a LayerNorm applied in the wrong order.

## Performance budget

- First page load: <1 s on broadband (600 KB gzipped weights).
- Per-token latency: <20 ms target. 200-char generation: <4 s.
- Re-renders during streaming: 1 per token, simple state update.
- No layout thrash: output container has fixed min-height.

## What this unblocks

After this lands, "Watch a single weight learn" (the second "What's next" item) becomes the obvious next addition — same data-loading pattern, different visualization.
