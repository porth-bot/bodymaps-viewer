/**
 * Small DOM control factories.
 *
 * These exist so the panel code reads as a description of the UI rather than a
 * wall of createElement calls. Each returns both the element and an `update`
 * so a control can be driven from state without the panel tracking references
 * by hand.
 */

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export interface Control<T> {
  root: HTMLElement;
  update(value: T): void;
}

export function section(title: string, options: { collapsed?: boolean } = {}): {
  root: HTMLElement;
  body: HTMLElement;
} {
  const root = el('section', 'panel');
  const header = el('button', 'panel-header');
  header.type = 'button';
  header.setAttribute('aria-expanded', options.collapsed ? 'false' : 'true');
  const caret = el('span', 'caret', '▾');
  header.append(caret, el('span', 'panel-title', title));

  const body = el('div', 'panel-body');
  if (options.collapsed) {
    root.classList.add('collapsed');
  }
  header.addEventListener('click', () => {
    const collapsed = root.classList.toggle('collapsed');
    header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  });
  root.append(header, body);
  return { root, body };
}

export function slider(opts: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format?(v: number): string;
  onInput(v: number): void;
}): Control<number> {
  const root = el('div', 'control');
  const head = el('div', 'control-head');
  const name = el('label', 'control-label', opts.label);
  const readout = el('span', 'control-value');
  head.append(name, readout);

  const input = el('input');
  input.type = 'range';
  input.min = String(opts.min);
  input.max = String(opts.max);
  input.step = String(opts.step);
  input.value = String(opts.value);

  const fmt = opts.format ?? ((v: number) => String(v));
  readout.textContent = fmt(opts.value);

  input.addEventListener('input', () => {
    const v = Number(input.value);
    readout.textContent = fmt(v);
    opts.onInput(v);
  });

  root.append(head, input);
  return {
    root,
    update(v) {
      if (document.activeElement !== input) input.value = String(v);
      readout.textContent = fmt(v);
    },
  };
}

export function toggle(opts: {
  label: string;
  value: boolean;
  hint?: string;
  onChange(v: boolean): void;
}): Control<boolean> {
  const root = el('label', 'toggle');
  const input = el('input');
  input.type = 'checkbox';
  input.checked = opts.value;
  input.addEventListener('change', () => opts.onChange(input.checked));
  const track = el('span', 'toggle-track');
  const text = el('span', 'toggle-label', opts.label);
  if (opts.hint) root.title = opts.hint;
  root.append(input, track, text);
  return {
    root,
    update(v) {
      input.checked = v;
    },
  };
}

export function segmented<T extends string>(opts: {
  label?: string;
  options: Array<{ value: T; label: string; title?: string }>;
  value: T;
  onChange(v: T): void;
}): Control<T> {
  const root = el('div', 'control');
  if (opts.label) root.append(el('div', 'control-label', opts.label));
  const group = el('div', 'segmented');
  group.setAttribute('role', 'radiogroup');
  const buttons = new Map<T, HTMLButtonElement>();

  for (const o of opts.options) {
    const b = el('button', 'segment', o.label);
    b.type = 'button';
    if (o.title) b.title = o.title;
    b.setAttribute('role', 'radio');
    b.addEventListener('click', () => opts.onChange(o.value));
    buttons.set(o.value, b);
    group.append(b);
  }

  const apply = (v: T) => {
    for (const [key, b] of buttons) {
      const on = key === v;
      b.classList.toggle('active', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
    }
  };
  apply(opts.value);
  root.append(group);
  return { root, update: apply };
}

export function select<T extends string>(opts: {
  label: string;
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange(v: T): void;
}): Control<T> {
  const root = el('div', 'control');
  const head = el('div', 'control-head');
  head.append(el('label', 'control-label', opts.label));
  const input = el('select', 'select');
  for (const o of opts.options) {
    const opt = el('option', undefined, o.label);
    opt.value = o.value;
    input.append(opt);
  }
  input.value = opts.value;
  input.addEventListener('change', () => opts.onChange(input.value as T));
  root.append(head, input);
  return {
    root,
    update(v) {
      input.value = v;
    },
  };
}

export function button(label: string, onClick: () => void, className = ''): HTMLButtonElement {
  const b = el('button', `btn ${className}`.trim(), label);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

export function rgbCss(color: readonly [number, number, number], alpha = 1): string {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

/** Compact volume formatting: mL below a litre, L above, always 3 significant digits. */
export function formatVolume(ml: number): string {
  if (ml >= 1000) return `${(ml / 1000).toFixed(2)} L`;
  if (ml >= 100) return `${ml.toFixed(0)} mL`;
  if (ml >= 10) return `${ml.toFixed(1)} mL`;
  return `${ml.toFixed(2)} mL`;
}

export function formatCount(n: number): string {
  return n.toLocaleString('en-US');
}
