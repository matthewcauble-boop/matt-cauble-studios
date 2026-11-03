"""Living-scene demo: add small life and a storybook grade to ANY generated shot, no assets needed.
python tools/living/demo.py in.mp4 out.mp4 [--mask mask.mp4] [--no-grade] [--seed 1]
--mask: a black/white video of the character(s) (e.g. from SAM 2 video); they keep full colour under the grade."""
import argparse, math
import cv2, numpy as np
import life as L

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--mask")
ap.add_argument("--no-grade", action="store_true"); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
frames, fps = L.read_frames(a.src); H, W = frames[0].shape[:2]; k = W / 854.0
masks = L.read_frames(a.mask)[0] if a.mask else None
rng = np.random.default_rng(a.seed)
bees = [L.loop_path(rng.uniform(0.1, 0.9) * W, rng.uniform(0.6, 0.85) * H, 40 * k, 15 * k, rng.uniform(1.9, 2.9), rng.uniform(3.0, 4.0), rng.uniform(0, 6.3)) for _ in range(4)]
drift = L.Drift(W, H, seed=a.seed, y_range=(0.12 * H, 0.83 * H))
up, down = L.bird_sprite(True), L.bird_sprite(False)
FLOCK = [(0, 0), (-34, 14), (-34, -12), (-66, 26), (-64, -22), (-98, 36), (-95, -30)]
out = L.Writer(a.dst, W, H, fps)
for i, f in enumerate(frames):
    t = i / fps; c = f.copy(); tt = L.on_twos(t, fps)
    for p in bees: L.bee(c, p, t, size=0.22 * k)
    drift.draw(c, t)
    for j, (dx, dy) in enumerate(FLOCK):                      # a loose V crossing the sky, each bird on its own flap
        x = -60 * k + 220 * k * tt + dx * k; y = 0.18 * H + dy * k + 5 * k * math.sin(tt * 6 + j * 1.7)
        L.place(c, up if int(tt * 7 + j * 0.6) % 2 == 0 else down, x, y, (24 - j * 0.6) * k, by_w=True, haze=0.15)
    if not a.no_grade:
        m = L.mask_from_video(masks[min(i, len(masks) - 1)]) if masks else None
        if m is not None and m.shape != (H, W): m = cv2.resize(m, (W, H))
        c = L.storybook_grade(c, m)
    out.write(c)
out.close(); print(a.dst, len(frames) / fps, "s")
