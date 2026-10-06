/**
 * Where the model's view sits on the camera picture.
 *
 * The model never sees the whole frame: it takes the bottom rows of each
 * camera frame (the road; the sky and the bonnet line above carry nothing
 * about steering) at full width and scales them to its 200 x 66 input. The
 * windshield draws the saliency map back onto exactly that band.
 */

/** Rows the model sees, counted up from the bottom of a camera frame. The same
 * number as the engine's crop; a test reads it from the engine's source. */
export const CROP_ROWS = 150;

/** The model's band on a camera frame `cameraHeight` rows tall, as fractions
 * of the frame's height: where it starts, and how tall it is. */
export function cropBand(cameraHeight: number): { top: number; height: number } {
  const rows = Math.min(CROP_ROWS, cameraHeight);
  return { top: (cameraHeight - rows) / cameraHeight, height: rows / cameraHeight };
}

/** How much wider the video is than the camera picture inside it: 4:2:0 video
 * needs an even width, so an odd-width camera frame gains one repeated column
 * on the right, which the windshield crops away. */
export function videoOverhang(camera: readonly [number, number], video: readonly [number, number] | null): number {
  return video ? video[0] / camera[0] : 1;
}
