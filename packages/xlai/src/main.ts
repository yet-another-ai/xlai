import { packageVersion } from './index';

const app = document.querySelector<HTMLDivElement>('#app');

if (app === null) {
  throw new Error('Expected #app to exist');
}

const appRoot = app;

async function render(): Promise<void> {
  try {
    const version = await packageVersion();
    appRoot.innerHTML = `
      <main>
        <h1>XLAI Wasm</h1>
        <p>Rust package version: <code>${version}</code></p>
      </main>
    `;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    appRoot.innerHTML = `
      <main>
        <h1>XLAI Wasm</h1>
        <p>Failed to initialize wasm: <code>${message}</code></p>
      </main>
    `;
  }
}

void render();
