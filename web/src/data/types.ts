/**
 * The shapes of the site data the page loads, as the site-data builder writes
 * them (`sightline site-data`). Paths inside results.json are relative to the
 * site root; paths inside a clip.json are relative to the clip's own folder.
 */

/** The page's two models, by the names the data uses for them: the model the
 * page ships, and its twin -- the same network trained on a shuffled split. */
export type ModelKey = "sightline" | "shuffled";

export const MODEL_KEYS: readonly ModelKey[] = ["sightline", "shuffled"];

/** One demo clip: `n` scored frames, with `lead_in` history frames before them
 * in the sprite sheets. */
export interface Clip {
  readonly id: string;
  readonly drive: string;
  /** How the page names the clip, e.g. "2018: minutes it never trained on". */
  readonly title: string;
  readonly fps: number;
  readonly n: number;
  readonly lead_in: number;
  readonly first_frame: number;
  /** The model's history offsets, in frames; every one is at most `lead_in`. */
  readonly frame_lags: readonly number[];
  /** How the clip was chosen, in words. */
  readonly rule: string;
  /** The driver's steering angle on each scored frame, degrees. */
  readonly labels_deg: readonly number[];
  /** Each model's output on each scored frame, from the reference runtime on
   * the reference decoder's frames. */
  readonly expected: Readonly<Record<ModelKey, readonly number[]>>;
  readonly sprite: {
    readonly files: readonly string[];
    readonly frames_per_sheet: number;
    readonly frame_w: number;
    readonly frame_h: number;
    /** sha256 of each sprite frame's RGB bytes as the reference decoder reads them. */
    readonly decoded_sha256: readonly string[];
  };
  /** The camera video, or null when the build wrote none. */
  readonly video: string | null;
  readonly video_size: readonly [number, number] | null;
  /** The camera frame's size; the video may carry one padding column more. */
  readonly camera_size: readonly [number, number];
  /** Mean absolute error on this clip alone: each model's, and steering straight ahead's. */
  readonly clip_mae_deg: Readonly<Record<ModelKey | "zero", number>>;
  /** The share of the scored frames the twin trained on, when the clip records it. */
  readonly twin_trained_frac?: number;
  /** The share of the scored frames within one frame of one the twin trained on. */
  readonly twin_neighbour_frac?: number;
  /** How far a browser's own prediction may stray from `expected`. */
  readonly tolerance?: { readonly angle_deg: number };
}

/** What the cockpit reads from results.json. The rest of the file stays in
 * `raw` for the parts of the page that show the measurements. */
export interface Results {
  readonly assets: {
    readonly models: Readonly<Record<ModelKey, string>>;
    readonly clips: Readonly<Record<string, string>>;
  };
  readonly shipped: { readonly run: string; readonly model: string; readonly protocol: string };
  readonly twin: { readonly run: string };
  readonly raw: Readonly<Record<string, unknown>>;
}
