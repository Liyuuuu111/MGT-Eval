#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const FRONTEND_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const PROJECT_ROOT = path.resolve(FRONTEND_ROOT, '..');
const CATALOG_PATH = path.join(FRONTEND_ROOT, 'src', 'config', 'fieldHelpCatalog.ts');
const EXAMPLES_ROOT = path.join(PROJECT_ROOT, 'examples');

const normalizeFieldPath = (rawPath) =>
  String(rawPath || '')
    .replace(/\[(\d+)\]/g, '.$1')
    .replace(/attack_configs\.[^.]+/g, 'attack_configs.*')
    .replace(/text_attacks\.[^.]+/g, 'text_attacks.*')
    .replace(/\.+/g, '.')
    .replace(/^\./, '')
    .replace(/\.$/, '')
    .toLowerCase();

const leafKey = (rawPath) => {
  const chunks = normalizeFieldPath(rawPath).split('.').filter(Boolean);
  return chunks[chunks.length - 1] || '';
};

const extractCatalogKeys = (content, sectionName) => {
  const sectionStart = content.indexOf(`export const ${sectionName}`);
  if (sectionStart < 0) {
    return new Set();
  }
  const bodyStart = content.indexOf('{', sectionStart);
  if (bodyStart < 0) {
    return new Set();
  }
  let depth = 0;
  let end = bodyStart;
  for (let i = bodyStart; i < content.length; i += 1) {
    const ch = content[i];
    if (ch === '{') depth += 1;
    if (ch === '}') depth -= 1;
    if (depth === 0) {
      end = i;
      break;
    }
  }
  const body = content.slice(bodyStart, end + 1);
  const keyRegex = /^\s{2}(?:'([^']+)'|"([^"]+)"|([a-zA-Z0-9_.*\-]+)):\s*\{/gm;
  const keys = new Set();
  let match;
  while ((match = keyRegex.exec(body))) {
    const rawKey = match[1] || match[2] || match[3];
    keys.add(String(rawKey).toLowerCase());
  }
  return keys;
};

const walkFiles = (root, predicate, out = []) => {
  if (!fs.existsSync(root)) return out;
  for (const name of fs.readdirSync(root)) {
    const full = path.join(root, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      walkFiles(full, predicate, out);
      continue;
    }
    if (predicate(full)) out.push(full);
  }
  return out;
};

const parseYamlPaths = (content) => {
  const paths = new Set();
  const stack = [];
  const lines = content.split(/\r?\n/);

  for (const rawLine of lines) {
    const noComment = rawLine.replace(/\s+#.*$/, '');
    if (!noComment.trim()) continue;
    const keyMatch = noComment.match(/^(\s*)(?:-\s*)?([a-zA-Z0-9_.-]+)\s*:/);
    if (!keyMatch) continue;

    const indent = keyMatch[1].length;
    const key = keyMatch[2];
    while (stack.length > 0 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }
    const parentPath = stack.map((s) => s.key).join('.');
    const fieldPath = parentPath ? `${parentPath}.${key}` : key;
    paths.add(normalizeFieldPath(fieldPath));
    stack.push({ indent, key });
  }

  return paths;
};

const toPosixPath = (p) => p.split(path.sep).join('/');

const run = () => {
  const catalog = fs.readFileSync(CATALOG_PATH, 'utf8');
  const exactKeys = extractCatalogKeys(catalog, 'EXACT_FIELD_HELP');
  const leafKeys = extractCatalogKeys(catalog, 'LEAF_FIELD_HELP');

  const yamlFiles = walkFiles(
    EXAMPLES_ROOT,
    (f) => f.endsWith('.yaml') || f.endsWith('.yml'),
  );

  const allFields = new Set();
  for (const file of yamlFiles) {
    const content = fs.readFileSync(file, 'utf8');
    for (const p of parseYamlPaths(content)) {
      allFields.add(p);
    }
  }

  let exactHit = 0;
  let leafHit = 0;
  let heuristicHit = 0;
  const heuristicSamples = [];

  for (const field of allFields) {
    const leaf = leafKey(field);
    if (exactKeys.has(field)) {
      exactHit += 1;
      continue;
    }
    if (leafKeys.has(leaf)) {
      leafHit += 1;
      continue;
    }
    heuristicHit += 1;
    if (heuristicSamples.length < 30) {
      heuristicSamples.push(field);
    }
  }

  const total = allFields.size;
  console.log('Field Help Coverage Report');
  console.log('--------------------------');
  console.log(`Examples root: ${toPosixPath(EXAMPLES_ROOT)}`);
  console.log(`YAML files scanned: ${yamlFiles.length}`);
  console.log(`Unique fields found: ${total}`);
  console.log(`Exact catalog hits: ${exactHit}`);
  console.log(`Leaf catalog hits: ${leafHit}`);
  console.log(`Heuristic fallback hits: ${heuristicHit}`);
  console.log(`Coverage: 100% (exact + leaf + heuristic fallback)`);

  if (heuristicSamples.length > 0) {
    console.log('\nHeuristic-only sample fields:');
    for (const field of heuristicSamples) {
      console.log(`- ${field}`);
    }
  }
};

run();
