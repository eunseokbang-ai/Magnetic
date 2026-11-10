import { describe, expect, it } from "vitest";
import { parse } from "@babel/parser";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

// App() is composed of hooks (src/hooks/) and its screen of components that
// take no props and read what they need with useApp() (src/AppContext.js).
// Two things can go wrong there that neither the build nor the linter sees,
// because a destructured name that is not provided is simply `undefined`
// until something is clicked:
//
//   - a component reads a name from useApp() that App does not publish;
//   - a hook reads a name from the hooks before it that none of them returns,
//     or that only a hook called after it returns.
//
// Both are checked here from the source text, so adding a panel or moving a
// handler between hooks fails this test instead of failing in the field.

const SRC = dirname(fileURLToPath(import.meta.url));
const ast = (file) => parse(readFileSync(file, "utf8"), { sourceType: "module", plugins: ["jsx"] });

function walk(node, visit) {
  if (!node || typeof node.type !== "string") return;
  visit(node);
  for (const key of Object.keys(node)) {
    const value = node[key];
    if (Array.isArray(value)) value.forEach((child) => walk(child, visit));
    else if (value && typeof value.type === "string") walk(value, visit);
  }
}

const sourceFiles = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? sourceFiles(join(dir, entry.name)) : /\.jsx?$/.test(entry.name) && !/\.test\./.test(entry.name) ? [join(dir, entry.name)] : []
  );

// the default-exported function of a module, and what it returns / takes apart
function defaultFunction(file) {
  const node = ast(file).program.body.find((n) => n.type === "ExportDefaultDeclaration");
  return node && node.declaration.type === "FunctionDeclaration" ? node.declaration : null;
}

function returnedNames(fn) {
  const ret = fn.body.body.find((s) => s.type === "ReturnStatement");
  if (!ret || ret.argument.type !== "ObjectExpression") return null;
  return ret.argument.properties.map((p) => p.key.name);
}

function destructuredFrom(fn, source) {
  // names in `const { a, b } = <source>;` at the top of the function
  for (const stmt of fn.body.body) {
    if (stmt.type !== "VariableDeclaration") continue;
    const d = stmt.declarations[0];
    if (d.id.type === "ObjectPattern" && d.init && source(d.init)) return d.id.properties.map((p) => p.key.name);
  }
  return [];
}

// ---- what the hooks are and what App does with them
const hooks = new Map(); // name -> { returns, inputs }
for (const file of sourceFiles(join(SRC, "hooks"))) {
  const fn = defaultFunction(file);
  if (!fn) continue;
  const param = fn.params[0];
  hooks.set(fn.id.name, {
    returns: returnedNames(fn),
    inputs: param && param.type === "Identifier" ? destructuredFrom(fn, (init) => init.type === "Identifier" && init.name === param.name) : [],
  });
}

const appFn = defaultFunction(join(SRC, "App.jsx"));
const calls = []; // in call order: { variable, hook, given: [variables spread in] }
const ownNames = [];
let published = null;
for (const stmt of appFn.body.body) {
  if (stmt.type !== "VariableDeclaration") continue;
  const d = stmt.declarations[0];
  const call = d.init && d.init.type === "CallExpression" && hooks.has(d.init.callee.name) ? d.init : null;
  if (call && d.id.type === "Identifier") {
    const arg = call.arguments[0];
    const given = !arg ? [] : arg.type === "Identifier" ? [arg.name] : arg.properties.map((p) => p.argument.name);
    calls.push({ variable: d.id.name, hook: call.callee.name, given });
  } else if (d.id.type === "Identifier" && d.id.name === "app") {
    published = d.init;
  }
}

describe("the hooks App is composed of", () => {
  it("are all found and each returns an object of names", () => {
    expect(calls.length).toBeGreaterThanOrEqual(8);
    for (const { hook } of calls) expect(hooks.get(hook).returns, hook).not.toBeNull();
  });

  it("each read only what the hooks called before them return", () => {
    const returnsOf = new Map();
    const problems = [];
    for (const { variable, hook, given } of calls) {
      const available = new Set(given.flatMap((v) => returnsOf.get(v) ?? [`<${v} is not a hook result called earlier>`]));
      for (const name of hooks.get(hook).inputs) {
        if (!available.has(name)) problems.push(`${hook} reads "${name}", which nothing passed to it provides`);
      }
      for (const v of given) if (!returnsOf.has(v)) problems.push(`${hook} is given "${v}" before it exists`);
      returnsOf.set(variable, hooks.get(hook).returns);
    }
    expect(problems).toEqual([]);
  });
});

describe("the screen's components", () => {
  // what useApp() hands out: every hook result spread into `app`, plus the names App adds itself
  const names = new Set();
  const byVariable = new Map(calls.map((c) => [c.variable, hooks.get(c.hook).returns]));
  for (const p of published.properties) {
    if (p.type === "SpreadElement") byVariable.get(p.argument.name).forEach((n) => names.add(n));
    else names.add(p.key.name);
  }
  void ownNames;

  const readers = [];
  for (const file of sourceFiles(SRC)) {
    walk(ast(file), (node) => {
      if (node.type !== "VariableDeclarator" || !node.init || node.init.type !== "CallExpression" || node.init.callee.name !== "useApp") return;
      if (node.id.type !== "ObjectPattern") return;
      readers.push({ file: relative(SRC, file).replace(/\\/g, "/"), names: node.id.properties.map((p) => p.key.name) });
    });
  }

  it("exist: the layout regions and the workflow sections", () => {
    const files = readers.map((r) => r.file);
    expect(files).toEqual(expect.arrayContaining(["components/layout/LeftSidebar.jsx", "components/layout/MapArea.jsx", "components/workflow/ProcessingSection.jsx"]));
    expect(readers.length).toBeGreaterThanOrEqual(13);
  });

  it("read from useApp() only names App publishes", () => {
    const missing = readers.flatMap((r) => r.names.filter((n) => !names.has(n)).map((n) => `${r.file}: ${n}`));
    expect(missing).toEqual([]);
  });

  it("App publishes every hook it calls", () => {
    const spread = published.properties.filter((p) => p.type === "SpreadElement").map((p) => p.argument.name);
    expect(spread.sort()).toEqual(calls.map((c) => c.variable).sort());
  });
});
