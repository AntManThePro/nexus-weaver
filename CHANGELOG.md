# Changelog

All notable changes to NEXUS Weaver are documented here.

## [1.0.0] — 2026-09-26

### Added
- Infinite-canvas node graph bridging hardware and software: INPUT (IMU, MIC, RF-SDR),
  COMPUTE (CPU / GPU / TPU with distinct scheduling personalities), MEMORY (state buffer
  with visible fill + overflow drops), OUTPUT (SERVO, LED-MATRIX, HID-PAYLOAD).
- Buses with weight / latency / bandwidth, drawn as multi-pass neon beziers (thickness = weight).
- Real-time particle dataflow on a preallocated 4096-slot object pool; fixed 120 Hz sim step.
- Live kernel injection: `customKernel(p, node, t)` with HOT RELOAD (button / Ctrl|Cmd+Enter),
  compile smoke-test, per-call try/catch, auto-revert to last-known-good after 12 faults,
  4 presets (passthrough, ReLU clamp, jitter/fuzz QA stress, HID priority boost).
- TRAIN: finite-difference gradient descent on bus weights against an analytic flow model;
  backward pink error pulses, weight-delta labels, 2-weight loss-landscape inset with descent
  ball + trail, loss sparkline, auto-stop on plateau.
- Simulated telemetry HUD: Loss, Throughput, Error rate, Stability, Power draw, pool usage, FPS.
- COMPILE TO PAYLOAD: pretty graph JSON, minified standalone `createNexusGraph(canvas, GRAPH)`
  runtime, and pure `step(state, dt)` function; copy + JSON download.
- SHARE: graph encoded into the URL hash (deflate-raw + base64url, base64 JSON fallback).
- Mouse, touch (pinch / two-finger pan) and keyboard controls; phone bottom-sheet + tab bar.
- Boot sequence overlay (<1.5 s, skippable), buffer-overflow screen flash + shake.
- GitHub Pages workflow.
