# Development

Contributor-focused documentation: continuous integration, publishing to registries, and how crates are classified in this workspace.

## Topics

- [Architecture / runtime execution](/architecture/) — crate boundaries; **Runtime execution hints** covers game-oriented defaults, merge order, cancellation, and WASM option fields (`chatExecution`, `runtimeChatExecutionDefaults`, …).
- [CI and testing](./ci-and-testing) — workflows, test lanes, and e2e environments
- [Publishing](./publishing) — crates.io and npm release order and automation
- [Crate taxonomy](./crates-taxonomy) — which crates ship to crates.io vs stay internal
- [QTS crate split (future)](./qts-crate-split) — optional decomposition of `xlai-qts-core`
- [Workspace refactor migration](../migration/workspace-refactor) — import and path changes
- [Native vendor layout](./native-vendor) — `vendor/native` submodules and source overrides
