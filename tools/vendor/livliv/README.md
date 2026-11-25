# LivLiv audio lip sync attribution

Upstream: https://github.com/livliv-dev/player

Revision: `83ecd9c48f33d9439411e596841193c24dc7b005`

`tools/sam_ellen_livliv.py` adapts the envelope from `src/runtime/speech.ts`,
`VowelCue` / `vowelTrack` from `src/runtime/vowel.ts`, and the mouth-family
mapping from `src/vrm.ts`. Copyright 2026 LivLiv; Apache License 2.0.
The original LICENSE and NOTICE are retained in this directory.

Changes (2026-11-05): Python/NumPy implementation, offline FFmpeg decoding,
configurable audio lead, unit-budget MMD mouth mapping, neutral end sample,
Blender baking, expression integration and restoration. This is an approximate
spectral cue, not phoneme recognition. No featured LivLiv character assets are
included or used.
