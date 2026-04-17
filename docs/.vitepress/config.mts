import { defineConfig } from 'vitepress';

const repo = 'https://github.com/yetanother.ai/xlai';

export default defineConfig({
  title: 'XLAI',
  description:
    'Rust-first AI integration workspace — unified API with pluggable backends for native apps and the browser.',
  lang: 'en-US',
  base: '/',
  lastUpdated: true,
  srcExclude: [
    'publishing.md',
    'crates-taxonomy.md',
    'qts-export-and-hf-publish.md',
    'qts-vocoder-streaming.md',
    'qts-wasm-browser-runtime.md',
    'qts-wasm-model-manifest.md',
    'huggingface-qts-model-card.md',
  ],
  themeConfig: {
    siteTitle: 'XLAI',
    search: {
      provider: 'local',
    },
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'Architecture', link: '/architecture/' },
      { text: 'Rust', link: '/rust/' },
      { text: 'JavaScript', link: '/js/' },
      { text: 'Providers', link: '/providers/' },
      { text: 'QTS', link: '/qts/' },
      { text: 'Development', link: '/development/' },
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Introduction', link: '/guide/' },
            { text: 'Getting started', link: '/guide/getting-started' },
            { text: 'Configuration', link: '/guide/configuration' },
          ],
        },
      ],
      '/architecture/': [
        {
          text: 'Architecture',
          items: [{ text: 'Workspace architecture', link: '/architecture/' }],
        },
      ],
      '/rust/': [
        {
          text: 'Rust SDK',
          items: [
            { text: 'Overview', link: '/rust/' },
            { text: 'Chat, agents, and tools', link: '/rust/chat-and-agents' },
            { text: 'Streaming', link: '/rust/streaming' },
          ],
        },
      ],
      '/js/': [
        {
          text: 'JavaScript / WASM',
          items: [{ text: '@yai-xlai/xlai', link: '/js/' }],
        },
      ],
      '/providers/': [
        {
          text: 'Providers',
          items: [{ text: 'Supported providers', link: '/providers/' }],
        },
      ],
      '/qts/': [
        {
          text: 'QTS (TTS)',
          items: [
            { text: 'Overview', link: '/qts/' },
            {
              text: 'Export and Hugging Face',
              link: '/qts/export-and-hf-publish',
            },
            { text: 'Vocoder streaming', link: '/qts/vocoder-streaming' },
            {
              text: 'Browser / WASM runtime',
              link: '/qts/wasm-browser-runtime',
            },
            {
              text: 'Browser model manifest',
              link: '/qts/wasm-model-manifest',
            },
            {
              text: 'Hugging Face model card',
              link: '/qts/huggingface-qts-model-card',
            },
          ],
        },
      ],
      '/development/': [
        {
          text: 'Development',
          items: [
            { text: 'Overview', link: '/development/' },
            { text: 'CI and testing', link: '/development/ci-and-testing' },
            { text: 'Publishing', link: '/development/publishing' },
            { text: 'Crate taxonomy', link: '/development/crates-taxonomy' },
            { text: 'QTS crate split (future)', link: '/development/qts-crate-split' },
            { text: 'Native vendor layout', link: '/development/native-vendor' },
          ],
        },
      ],
      '/migration/': [
        {
          text: 'Migration',
          items: [
            {
              text: 'Workspace refactor',
              link: '/migration/workspace-refactor',
            },
          ],
        },
      ],
    },
    socialLinks: [{ icon: 'github', link: repo }],
    editLink: {
      pattern: `${repo}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'Released under the Apache License 2.0.',
      copyright: 'Copyright © yetanother.ai',
    },
    outline: {
      level: [2, 3],
    },
  },
});
