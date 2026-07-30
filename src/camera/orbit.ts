/**
 * Orbit camera for the 3D view, in volume local millimetres.
 *
 * Azimuth 0 puts the camera anterior to the patient looking posteriorly, the
 * view a clinician expects to open on. Up is fixed to +S (superior) rather
 * than free: a body view that can roll loses the read of which way is up.
 */

import {
  glLookAt, glMultiply, glPerspective, vadd, vcross, vnorm, vscale, vsub,
  type GLMat, type Vec3,
} from '../core/mat4';

const UP: Vec3 = [0, 0, 1];
const MAX_ELEVATION = Math.PI / 2 - 0.02;

export class OrbitCamera {
  /** Point the camera orbits, in local mm. */
  target: Vec3 = [0, 0, 0];
  distance = 500;
  /** Radians. 0 = anterior view. */
  azimuth = 0;
  /** Radians, clamped just short of the poles so `up` never degenerates. */
  elevation = 0.25;
  fovY = (35 * Math.PI) / 180;

  private radius = 300;

  /** Frame the whole volume, given its extent in mm. */
  frame(extent: Vec3): void {
    this.target = [extent[0] / 2, extent[1] / 2, extent[2] / 2];
    this.radius = 0.5 * Math.hypot(extent[0], extent[1], extent[2]);
    this.distance = this.radius / Math.sin(this.fovY / 2) * 0.85;
  }

  reset(): void {
    this.azimuth = 0;
    this.elevation = 0.25;
    this.distance = this.radius / Math.sin(this.fovY / 2) * 0.85;
  }

  setStandardView(view: 'anterior' | 'posterior' | 'left' | 'right' | 'superior' | 'inferior'): void {
    const table: Record<string, [number, number]> = {
      anterior: [0, 0],
      posterior: [Math.PI, 0],
      left: [-Math.PI / 2, 0],
      right: [Math.PI / 2, 0],
      superior: [0, MAX_ELEVATION],
      inferior: [0, -MAX_ELEVATION],
    };
    const [az, el] = table[view];
    this.azimuth = az;
    this.elevation = el;
  }

  orbit(dAzimuth: number, dElevation: number): void {
    this.azimuth += dAzimuth;
    this.elevation = Math.max(-MAX_ELEVATION, Math.min(MAX_ELEVATION, this.elevation + dElevation));
  }

  dolly(factor: number): void {
    // Clamped so the wheel cannot push through the target or lose the volume
    // off at a vanishing point.
    this.distance = Math.max(this.radius * 0.15, Math.min(this.radius * 20, this.distance * factor));
  }

  /** Pan in screen space. The distance scaling keeps anatomy under the pointer at any zoom. */
  pan(dxPixels: number, dyPixels: number, viewportHeight: number): void {
    const worldPerPixel = (2 * this.distance * Math.tan(this.fovY / 2)) / Math.max(viewportHeight, 1);
    const eye = this.position();
    const forward = vnorm(vsub(this.target, eye));
    const right = vnorm(vcross(forward, UP));
    const up = vcross(right, forward);
    this.target = vadd(
      this.target,
      vadd(vscale(right, -dxPixels * worldPerPixel), vscale(up, dyPixels * worldPerPixel)),
    );
  }

  position(): Vec3 {
    const ce = Math.cos(this.elevation);
    return [
      this.target[0] + this.distance * Math.sin(this.azimuth) * ce,
      this.target[1] + this.distance * Math.cos(this.azimuth) * ce,
      this.target[2] + this.distance * Math.sin(this.elevation),
    ];
  }

  view(): GLMat {
    return glLookAt(this.position(), this.target, UP);
  }

  projection(aspect: number): GLMat {
    // The near plane tracks the orbit distance instead of being a fixed small
    // number, so depth precision stays usable when zoomed out to a whole body.
    const near = Math.max(this.distance - this.radius * 3, this.radius * 0.01);
    const far = this.distance + this.radius * 6;
    return glPerspective(this.fovY, aspect, near, far);
  }

  viewProjection(aspect: number): GLMat {
    return glMultiply(this.projection(aspect), this.view());
  }
}
