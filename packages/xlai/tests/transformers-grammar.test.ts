import { describe, expect, it } from 'vitest';
import { rustPayloadToGrammar } from '../src/transformers/adapter';

describe('rustPayloadToGrammar', () => {
  it('maps json_schema from Rust', () => {
    const g = rustPayloadToGrammar(
      {
        type: 'json_schema',
        schema: { type: 'object', properties: { a: { type: 'string' } } },
      },
      undefined,
    );
    expect(g?.type).toBe('json_schema');
    if (g?.type === 'json_schema') {
      expect(g.schema.properties).toBeDefined();
    }
  });

  it('maps lark payload including startSymbol', () => {
    const g = rustPayloadToGrammar(
      {
        type: 'lark',
        grammar: 'start: "a"',
        startSymbol: 'start',
      },
      undefined,
    );
    expect(g?.type).toBe('lark');
    if (g?.type === 'lark') {
      expect(g.grammar).toContain('start');
      expect(g.startSymbol).toBe('start');
    }
  });

  it('wraps tool schema as json_schema grammar', () => {
    const schema = { type: 'object', additionalProperties: false };
    const g = rustPayloadToGrammar(undefined, schema);
    expect(g?.type).toBe('json_schema');
    if (g?.type === 'json_schema') {
      expect(g.schema).toEqual(schema);
    }
  });

  it('prefers explicit grammar when both are absent tool wins first check — tool only', () => {
    const g = rustPayloadToGrammar(
      { type: 'json_schema', schema: { type: 'string' } },
      { type: 'object' },
    );
    // toolSchema branch runs first in implementation
    expect(g?.type).toBe('json_schema');
    if (g?.type === 'json_schema') {
      expect(g.schema).toEqual({ type: 'object' });
    }
  });
});
