import { describe, expect, it } from 'vitest';

import {
  applyMat4, applyMat4Dir, glIdentity, glInvert, glMultiply, glNormalMatrix,
  glOrtho, glPerspective, glLookAt, mat4From, mat4Identity, mat4Invert,
  mat4Multiply, toGLMat, vcross, vnorm, type Vec3,
} from '../src/core/mat4';

/** Column-major mat3 times a vector. */
function mat3Apply(m: Float32Array, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

/** Column-major mat4 times a direction, ignoring translation. */
function glApplyDir(m: Float32Array, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
  ];
}

function angleBetween(a: Vec3, b: Vec3): number {
  const na = vnorm(a);
  const nb = vnorm(b);
  const d = Math.max(-1, Math.min(1, na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]));
  return (Math.acos(d) * 180) / Math.PI;
}

describe('mat4Invert', () => {
  it('inverts an affine so the round trip returns the original point', () => {
    const m = mat4From([
      0.8164, 0, 0, -417.18,
      0, 0.8164, 0, -417.18,
      0, 0, 2.5, 0,
      0, 0, 0, 1,
    ]);
    const inv = mat4Invert(m);
    const p: Vec3 = [251, 174, 35];
    const back = applyMat4(inv, applyMat4(m, p));
    for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(p[i], 6);
  });

  it('inverts a matrix whose first pivot is zero', () => {
    // Needs the partial pivoting: a naive Gauss-Jordan divides by zero here.
    const m = mat4From([
      0, 2, 0, 1,
      3, 0, 0, 2,
      0, 0, 4, 3,
      0, 0, 0, 1,
    ]);
    const p: Vec3 = [1, 2, 3];
    const back = applyMat4(mat4Invert(m), applyMat4(m, p));
    for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(p[i], 9);
  });

  it('throws on a singular matrix instead of returning NaNs', () => {
    const m = mat4From([1, 2, 3, 0, 2, 4, 6, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(() => mat4Invert(m)).toThrow(/singular/i);
  });

  it('composes with multiply the way the affines are actually used', () => {
    const a = mat4From([1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2, 0, 0, 0, 1]);
    const b = mat4From([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1]);
    const p: Vec3 = [1, 1, 1];
    // (a*b) applied to p must equal a applied to (b applied to p).
    const viaProduct = applyMat4(mat4Multiply(a, b), p);
    const viaSteps = applyMat4(a, applyMat4(b, p));
    for (let i = 0; i < 3; i++) expect(viaProduct[i]).toBeCloseTo(viaSteps[i], 9);
  });

  it('applyMat4Dir ignores the translation column', () => {
    const m = mat4From([2, 0, 0, 100, 0, 2, 0, 100, 0, 0, 2, 100, 0, 0, 0, 1]);
    expect([...applyMat4Dir(m, [1, 0, 0])]).toEqual([2, 0, 0]);
  });
});

describe('glNormalMatrix', () => {
  it('is the identity for a pure translation', () => {
    const m = glIdentity();
    m[12] = 0.4;
    m[13] = 0.4;
    m[14] = 1.25;
    const n = glNormalMatrix(m);
    for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]] as Vec3[]) {
      expect(angleBetween(mat3Apply(n, v), v)).toBeLessThan(1e-4);
    }
  });

  /**
   * The property that defines a normal matrix: it must keep normals
   * perpendicular to the surface after a non-uniform transform. Testing that
   * directly, rather than comparing against a hand-written inverse-transpose,
   * means the test still passes judgement if the storage convention changes.
   */
  it('keeps normals perpendicular to tangents under anisotropic scaling', () => {
    const m = glIdentity();
    m[0] = 0.81641;
    m[5] = 0.81641;
    m[10] = 2.5;
    const n = glNormalMatrix(m);

    const normal: Vec3 = vnorm([1, 1, 1]);
    for (const t of [[1, -1, 0], [0, 1, -1]] as Vec3[]) {
      // Build a tangent genuinely perpendicular to the normal, transform both,
      // and require they stay perpendicular.
      const tangent = vnorm(vcross(normal, vcross(t, normal)));
      const tn = glApplyDir(m, tangent);
      const nn = mat3Apply(n, normal);
      const dot = vnorm(tn)[0] * vnorm(nn)[0] + vnorm(tn)[1] * vnorm(nn)[1] + vnorm(tn)[2] * vnorm(nn)[2];
      expect(Math.abs(dot)).toBeLessThan(1e-5);
    }
  });

  it('stays perpendicular under a rotation composed with anisotropic scale', () => {
    // The case that catches a transposed normal matrix. With a symmetric upper
    // 3x3 (translation, uniform scale, axis-aligned scale) the transpose is
    // indistinguishable from the correct answer, so a rotation has to be in
    // the mix for this test to have any teeth.
    const th = 0.7;
    const c = Math.cos(th);
    const s = Math.sin(th);
    const scale: Vec3 = [0.81641, 0.81641, 2.5];
    const m = glIdentity();
    // Column-major R * S: column j is R's column j times scale[j].
    m[0] = c * scale[0]; m[1] = s * scale[0]; m[2] = 0;
    m[4] = -s * scale[1]; m[5] = c * scale[1]; m[6] = 0;
    m[8] = 0; m[9] = 0; m[10] = scale[2];

    const n = glNormalMatrix(m);
    const normal: Vec3 = vnorm([1, 1, 1]);
    for (const t of [[1, -1, 0], [0, 1, -1], [1, 0, -1]] as Vec3[]) {
      const tangent = vnorm(vcross(normal, vcross(t, normal)));
      const tn = vnorm(glApplyDir(m, tangent));
      const nn = vnorm(mat3Apply(n, normal));
      expect(Math.abs(tn[0] * nn[0] + tn[1] * nn[1] + tn[2] * nn[2])).toBeLessThan(1e-5);
    }
  });
});

describe('GL matrices', () => {
  it('toGLMat transposes row-major into column-major', () => {
    const m = mat4From([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const g = toGLMat(m);
    // Row 0 of the row-major input becomes column 0 of the output.
    expect([...g.slice(0, 4)]).toEqual([1, 5, 9, 13]);
  });

  it('glInvert round-trips a view-projection', () => {
    const vp = glMultiply(glPerspective(0.6, 1.4, 1, 1000), glLookAt([10, 20, 30], [0, 0, 0], [0, 0, 1]));
    const inv = glInvert(vp);
    const id = glMultiply(vp, inv);
    for (let i = 0; i < 16; i++) {
      expect(id[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 4);
    }
  });

  it('glLookAt puts the target on the negative z axis of view space', () => {
    const eye: Vec3 = [0, 100, 0];
    const view = glLookAt(eye, [0, 0, 0], [0, 0, 1]);
    // The target maps to (0, 0, -distance).
    const t = [
      view[0] * 0 + view[4] * 0 + view[8] * 0 + view[12],
      view[1] * 0 + view[5] * 0 + view[9] * 0 + view[13],
      view[2] * 0 + view[6] * 0 + view[10] * 0 + view[14],
    ];
    expect(t[0]).toBeCloseTo(0, 5);
    expect(t[1]).toBeCloseTo(0, 5);
    expect(t[2]).toBeCloseTo(-100, 4);
  });

  it('glLookAt does not blow up when looking straight along up', () => {
    // Orbiting through the pole must not produce NaNs from a zero cross product.
    const view = glLookAt([0, 0, 100], [0, 0, 0], [0, 0, 1]);
    for (let i = 0; i < 16; i++) expect(Number.isFinite(view[i])).toBe(true);
  });

  it('glPerspective maps the near and far planes to -1 and 1 in NDC', () => {
    const near = 2;
    const far = 500;
    const p = glPerspective(1.0, 1.0, near, far);
    const project = (z: number) => {
      const clipZ = p[10] * z + p[14];
      const clipW = p[11] * z;
      return clipZ / clipW;
    };
    expect(project(-near)).toBeCloseTo(-1, 5);
    expect(project(-far)).toBeCloseTo(1, 5);
  });

  it('glOrtho maps its box corners to the NDC cube', () => {
    const m = glOrtho(-2, 6, -1, 3, 1, 9);
    const apply = (v: Vec3): Vec3 => [
      m[0] * v[0] + m[12],
      m[5] * v[1] + m[13],
      m[10] * v[2] + m[14],
    ];
    const lo = apply([-2, -1, -1]);
    const hi = apply([6, 3, -9]);
    expect(lo[0]).toBeCloseTo(-1, 6);
    expect(lo[1]).toBeCloseTo(-1, 6);
    expect(lo[2]).toBeCloseTo(-1, 6);
    expect(hi[0]).toBeCloseTo(1, 6);
    expect(hi[1]).toBeCloseTo(1, 6);
    expect(hi[2]).toBeCloseTo(1, 6);
  });

  it('mat4Identity and glIdentity leave inputs untouched', () => {
    const p: Vec3 = [3, -4, 5];
    expect([...applyMat4(mat4Identity(), p)]).toEqual([3, -4, 5]);
  });
});
