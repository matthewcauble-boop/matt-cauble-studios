"""Living-scene layer kit: make a generated shot feel alive with small things drawn in code.

The video model makes the hero shot (the big action, the camera, the set). Everything small and alive is code:
insects with trails, drifting seeds and motes, flapping birds, animals cut from a painted sheet, animals painted
into a frame and tracked to the plate, a storybook grade that mutes the set and keeps the characters bright, and
code-animated characters composited by black/white difference matting. All local, all free, all re-timeable.

Frames are float32 RGB in 0..1, shape (H, W, 3). Only numpy + OpenCV. See LIVING-SCENES.md for the method.
"""
import math, subprocess
import cv2, numpy as np

# ---------------------------------------------------------------- video io
def read_frames(path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 24; out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f[:, :, ::-1].astype(np.float32) / 255)
    return out, fps

class Writer:
    """Pipe float frames to ffmpeg (H.264, yuv420p)."""
    def __init__(self, path, w, h, fps, crf=17):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                                   "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    def write(self, f): self.p.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    def close(self): self.p.stdin.close(); self.p.wait()

def on_twos(t, fps=24):
    """Quantise time to every second frame: background critters drawn on twos read hand-made, not slick."""
    return math.floor(t * fps / 2) * 2 / fps

# ---------------------------------------------------------------- sprites from a painted sheet
def cut_sheet(path, names, min_px=3000, thr=28, order="rows", grow=9):
    """Cut the drawings out of a sheet painted on a flat background (corner colour). Returns {name: RGBA float}.
    order="rows": top-to-bottom bands, then left-to-right; "cols": left-to-right only. len(names) must match.
    grow joins a drawing's loose bits (legs, tails); lower it when drawings sit close together (a run cycle)."""
    im = cv2.imread(str(path))[:, :, ::-1].astype(np.float32)
    bg = np.median(im[:30, :30].reshape(-1, 3), 0); d = np.abs(im - bg).max(2)
    fg = (d > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=grow))
    boxes = [(stats[i, 1], stats[i, 0], stats[i, 3], stats[i, 2]) for i in range(1, n) if (fg[lab == i]).sum() > min_px]
    if order == "rows":
        boxes.sort(key=lambda b: b[0]); bands = []
        for b in boxes:
            if bands and abs(b[0] - bands[-1][0][0]) < 120: bands[-1].append(b)
            else: bands.append([b])
        boxes = [b for band in bands for b in sorted(band, key=lambda b: b[1])]
    else:
        boxes.sort(key=lambda b: b[1])
    assert len(boxes) == len(names), f"{path}: found {len(boxes)} drawings, expected {len(names)}"
    out = {}
    for name, (y, x, h, w) in zip(names, boxes):
        sub, dd = im[y:y + h, x:x + w], d[y:y + h, x:x + w]
        core = _fill_holes((dd > thr).astype(np.uint8))
        a = np.clip((dd - 10) / 40, 0, 1); a = np.where(core > 0, 1.0, a) * cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=2)
        ys, xs = np.nonzero(core)
        out[name] = np.dstack([sub, a * 255])[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    return out

def _fill_holes(m):
    h, w = m.shape; ff = np.zeros((h + 2, w + 2), np.uint8); inv = (m == 0).astype(np.uint8) * 255
    cv2.floodFill(inv, ff, (0, 0), 0)
    return np.maximum(m, (inv > 0).astype(np.uint8))

# ---------------------------------------------------------------- placing things in the plate
def place(c, spr, x, y, h, flip=False, by_w=False, sq=1.0, alpha=1.0, haze=0.0, sky=(0.93, 0.85, 0.7), match=0.0, grass=0, seed=0):
    """Bottom-centre of an RGBA sprite at (x, y), h px tall (or h px WIDE with by_w: use it for flappers, whose drawings
    differ in height between wing-up and wing-down). sq squashes vertically with area kept. haze pulls far things toward
    the sky colour. match (0..1) moves the sprite's colour statistics toward the plate around it. grass (px) draws a
    ragged band of the plate's own pixels back over the feet so it stands IN the grass, not on it."""
    H, W = c.shape[:2]
    if flip: spr = spr[:, ::-1]
    if by_w: h = h * spr.shape[0] / spr.shape[1]
    hh = max(2, int(round(h * sq))); ww = max(2, int(round(spr.shape[1] * h / spr.shape[0] / sq)))
    s = cv2.resize(spr, (ww, hh), interpolation=cv2.INTER_AREA)
    rgb, a = s[:, :, :3], s[:, :, 3:4] * alpha
    if haze: rgb = rgb * (1 - haze) + np.array(sky, np.float32) * haze
    X0, Y0 = int(round(x - ww / 2)), int(round(y - hh))
    xa, ya, xb, yb = max(0, X0), max(0, Y0), min(W, X0 + ww), min(H, Y0 + hh)
    if xa >= xb or ya >= yb: return
    r = c[ya:yb, xa:xb]; sa = a[ya - Y0:yb - Y0, xa - X0:xb - X0]; sr = rgb[ya - Y0:yb - Y0, xa - X0:xb - X0]
    if match:
        pad = int(hh * 0.8); bg = c[max(0, ya - pad):min(H, yb + pad // 2), max(0, xa - pad):min(W, xb + pad)].reshape(-1, 3)
        m = sa[..., 0] > 0.5
        if m.any():
            mu_s, sd_s = sr[m].mean(0), sr[m].std(0) + 1e-3; mu_b, sd_b = bg.mean(0), bg.std(0)
            tgt = (sr - mu_s) / sd_s * (0.55 * sd_s + 0.45 * sd_b) + (0.62 * mu_s + 0.38 * mu_b)
            sr = sr * (1 - match) + np.clip(tgt, 0, 1) * match
        sr = cv2.GaussianBlur(sr, (0, 0), 0.6)
    orig = r.copy(); r[:] = r * (1 - sa) + sr * sa
    if grass:
        gh = min(grass, r.shape[0]); rng = np.random.default_rng(seed or int(x * 7 + y)); band = np.zeros(r.shape[:2], np.float32)
        for gx in range(0, r.shape[1], 2):
            band[r.shape[0] - int(gh * rng.uniform(0.4, 1.0)):, gx:gx + 1] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), 0.7)[..., None]; r[:] = r * (1 - band) + orig * band

def shadow(c, x, y, w, k=0.25):
    """Soft contact shadow (flat ellipse) at (x, y), w px wide. Fade k to 0 as the thing leaves the ground."""
    H, W = c.shape[:2]; yy, xx = np.ogrid[:H, :W]
    m = np.clip(1 - (((xx - x) / (w / 2)) ** 2 + ((yy - y) / (w / 8)) ** 2), 0, 1) * k
    c *= (1 - m[..., None] * 0.6)

# ---------------------------------------------------------------- insects with trails
def loop_path(cx, cy, rx, ry, w1, w2, ph):
    """A wandering, never-repeating-looking figure loop for a hovering insect: path(t) -> (x, y)."""
    return lambda t: (cx + rx * math.sin(w1 * t + ph) + 0.3 * rx * math.sin(2.7 * w1 * t),
                      cy + ry * math.sin(w2 * t + ph * 1.3) + 0.25 * ry * math.cos(3.1 * w2 * t))

def bee(c, path, t, size=0.22, n_trail=18, dt=0.025, trail_rgb=(1.0, 0.97, 0.86), trail_alpha=0.35):
    """A tiny soft bee (amber body, two stripes, flickering wings) plus a fading wisp of where it has been.
    size 0.22 is about 4 px at 854x480: barely seeable is the point. Bigger reads as a sticker."""
    H, W = c.shape[:2]; SS = 4; pts = [path(t - k * dt) for k in range(n_trail, -1, -1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bx0, by0 = int(max(0, min(xs) - 4)), int(max(0, min(ys) - 4)); bx1, by1 = int(min(W, max(xs) + 5)), int(min(H, max(ys) + 5))
    if bx1 > bx0 and by1 > by0:
        lay = np.zeros(((by1 - by0) * SS, (bx1 - bx0) * SS), np.float32)
        for k in range(len(pts) - 1):
            al = (k / (len(pts) - 1)) ** 1.8 * trail_alpha
            p0 = (int((pts[k][0] - bx0) * SS), int((pts[k][1] - by0) * SS)); p1 = (int((pts[k + 1][0] - bx0) * SS), int((pts[k + 1][1] - by0) * SS))
            cv2.line(lay, p0, p1, float(al), max(1, int(SS * 0.5)), cv2.LINE_AA)
        lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), SS * 0.35), (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA)[..., None]
        r = c[by0:by1, bx0:bx1]; r[:] = r * (1 - lay) + np.array(trail_rgb, np.float32) * lay
    x, y = path(t); x2, _ = path(t + 0.02)
    S = 8; R = int(5 * size * S); n = 4 * R; spr = np.zeros((n, n, 4), np.float32); cx, cy = n // 2, n // 2 + R // 3
    flap = 0.5 + 0.5 * math.sin(t * 95)
    for ang in (-30, 15):
        cv2.ellipse(spr, (cx - R // 4, cy - int(R * (0.75 + 0.35 * flap))), (int(R * 0.7), int(R * 0.42)), ang, 0, 360, (0.95, 0.97, 1.0, 0.5), -1, cv2.LINE_AA)
    cv2.ellipse(spr, (cx, cy), (int(R * 1.15), R), 0, 0, 360, (0.98, 0.72, 0.18, 1.0), -1, cv2.LINE_AA)
    for sx in (-0.3, 0.25):
        cv2.ellipse(spr, (cx + int(sx * R), cy), (int(R * 0.15), int(R * 0.92)), 0, 0, 360, (0.25, 0.16, 0.1, 1.0), -1, cv2.LINE_AA)
    cv2.circle(spr, (cx + int(R * 0.9), cy - int(R * 0.15)), max(1, R // 5), (0.12, 0.08, 0.06, 1.0), -1, cv2.LINE_AA)
    spr = cv2.GaussianBlur(spr, (0, 0), S * 0.35); ow = max(3, int(round(20 * size))); spr = cv2.resize(spr, (ow, ow), interpolation=cv2.INTER_AREA)
    if x2 < x: spr = spr[:, ::-1]
    X0, Y0 = int(x - ow / 2), int(y - ow / 2)
    if X0 < 0 or Y0 < 0 or X0 + ow > W or Y0 + ow > H: return
    a = spr[:, :, 3:4]; r = c[Y0:Y0 + ow, X0:X0 + ow]; r[:] = r * (1 - a) + spr[:, :, :3] * a

# ---------------------------------------------------------------- drifting seeds and motes
def _seed_sprite(r, glow):
    S = 6; R = int(r * S); n = 2 * R + 4 * S; img = np.zeros((n, n), np.float32); cx = cy = n // 2
    for k in range(14):
        a = 2 * math.pi * k / 14 + 0.2 * math.sin(k * 3.7)
        cv2.line(img, (cx, cy), (int(cx + R * math.cos(a)), int(cy + R * 0.85 * math.sin(a))), 0.55, max(1, S // 3), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, S // 2), 0.9, -1, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx + int(R * 0.15), cy + int(R * 1.1)), 0.7, max(1, S // 3), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), S * (0.5 + glow))
    return cv2.resize(img, (max(3, n // S), max(3, n // S)), interpolation=cv2.INTER_AREA)

class Drift:
    """Backlit seed heads and pollen motes on a breeze, with depth (near = bigger, faster, softer).
    drift_x adds a screen-space push, e.g. negative when the camera trucks right so they parallax past."""
    def __init__(self, w, h, seed=1, n_seeds=16, n_motes=26, y_range=(60, 400), rgb=(1.0, 1.0, 0.96), mote_rgb=(1.0, 0.93, 0.72)):
        rng = np.random.default_rng(seed); self.w, self.h, self.rgb, self.mote_rgb = w, h, rgb, mote_rgb; self.q = []
        for _ in range(n_seeds):
            d = rng.uniform(0, 1); r = 1.4 + 4.5 * d ** 1.5
            self.q.append(dict(x=rng.uniform(-40, w), y=rng.uniform(*y_range), r=r, vx=10 + 26 * d, vy=-3 - 6 * rng.uniform(),
                               ph=rng.uniform(0, 6.3), al=0.75 + 0.25 * d, spr=_seed_sprite(r, 0.2 + 0.6 * d ** 2)))
        for _ in range(n_motes):
            self.q.append(dict(x=rng.uniform(0, w), y=rng.uniform(*y_range), r=0.8, vx=rng.uniform(2, 8), vy=rng.uniform(-4, 2),
                               ph=rng.uniform(0, 6.3), al=rng.uniform(0.15, 0.35), spr=None))
    def draw(self, c, t, drift_x=0.0):
        W, H = self.w, self.h; glow = np.array([[.3, .6, .3], [.6, 1, .6], [.3, .6, .3]], np.float32)[..., None]
        for q in self.q:
            x = (q["x"] + (q["vx"] + drift_x * (0.4 + q["r"] / 6)) * t + 6 * math.sin(t * 0.9 + q["ph"])) % (W + 80) - 40
            y = q["y"] + q["vy"] * t + 5 * math.sin(t * 1.3 + q["ph"] * 1.7)
            if q["spr"] is None:
                xi, yi = int(x), int(y)
                if 1 <= xi < W - 1 and 1 <= yi < H - 1:
                    m = glow * q["al"] * (0.6 + 0.4 * math.sin(t * 3 + q["ph"]))
                    r = c[yi - 1:yi + 2, xi - 1:xi + 2]; r[:] = r * (1 - m) + np.array(self.mote_rgb, np.float32) * m
                continue
            sp = q["spr"]; hh, ww = sp.shape; X0, Y0 = int(x - ww / 2), int(y - hh / 2)
            if X0 < 0 or Y0 < 0 or X0 + ww > W or Y0 + hh > H: continue
            a = (sp * q["al"])[..., None]; r = c[Y0:Y0 + hh, X0:X0 + ww]; r[:] = r * (1 - a) + np.array(self.rgb, np.float32) * a

# ---------------------------------------------------------------- a procedural bird (no sheet needed)
def bird_sprite(up, w=40, rgb=(0.32, 0.45, 0.62)):
    """A soft painted-looking little bird, wings up or down. Use sheet drawings in the plate's style when you have them."""
    S = 6; W_, H_ = w * S, int(w * 0.7) * S; img = np.zeros((H_, W_, 4), np.float32); cx, cy = W_ // 2, H_ // 2 + (S * 2 if up else -S * 2)
    col = tuple(rgb) + (1.0,)
    cv2.ellipse(img, (cx, cy), (int(W_ * 0.16), int(H_ * 0.14)), 0, 0, 360, col, -1, cv2.LINE_AA)
    wy = cy - int(H_ * 0.32) if up else cy + int(H_ * 0.22)
    for sgn in (-1, 1):
        pts = np.array([[cx, cy], [cx + sgn * int(W_ * 0.46), wy], [cx + sgn * int(W_ * 0.2), cy - S * 2]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    cv2.circle(img, (cx + int(W_ * 0.15), cy - S * 2), int(H_ * 0.09), col, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), S * 0.6)
    return cv2.resize(img, (w, int(w * 0.7)), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------- tracking a painted patch to a moving plate
def track_plate(ref, frames, roi, n_feat=3000, min_inliers=25, smooth=7):
    """Similarity transform (scale, rotate, shift) from ref to every frame, from ORB features inside roi=(x0,y0,x1,y1)
    on static scenery (far bank, buildings, trees: not water, not the action). RANSAC, gaps filled from the nearest
    good frame, then a median over `smooth` frames so the patch never jitters. ECC drifted and skewed in our tests."""
    g = lambda im: cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    H, W = ref.shape[:2]; mask = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = roi; mask[y0:y1, x0:x1] = 255
    orb = cv2.ORB_create(n_feat, fastThreshold=7); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    k0, d0 = orb.detectAndCompute(g(ref), mask); raw = []
    for f in frames:
        k1, d1 = orb.detectAndCompute(g(f), mask); M = None
        if d1 is not None and len(k1) > 20:
            mt = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:400]
            if len(mt) > 20:
                p0 = np.float32([k0[m.queryIdx].pt for m in mt]); p1 = np.float32([k1[m.trainIdx].pt for m in mt])
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000)
                if M is not None and inl.sum() < min_inliers: M = None
        raw.append(M)
    good = [i for i, M in enumerate(raw) if M is not None]
    if not good: raise RuntimeError("no frame tracked: widen the roi or pick static scenery")
    raw = [M if M is not None else raw[min(good, key=lambda j: abs(j - i))] for i, M in enumerate(raw)]
    h = smooth // 2
    return [np.median(np.stack(raw[max(0, i - h):i + h + 1]), 0).astype(np.float32) for i in range(len(raw))], len(good)

def diff_matte(edited, ref, box, thr=0.07, keep=2, colour=None):
    """Cut what an image edit ADDED to a frame: blurred difference (painterly re-render noise cancels), inside box,
    optionally only pixels where colour(rgb)->bool is true, the `keep` biggest blobs, holes filled."""
    H, W = ref.shape[:2]; x0, y0, x1, y1 = box
    d = np.abs(cv2.GaussianBlur(edited, (0, 0), 1.0) - cv2.GaussianBlur(ref, (0, 0), 1.0)).max(2)
    m = np.zeros((H, W), np.uint8); m[y0:y1, x0:x1] = (d[y0:y1, x0:x1] > thr)
    k = np.ones((3, 3), np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if colour is not None: m &= cv2.dilate(colour(edited).astype(np.uint8), k, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + np.argsort(st[1:, cv2.CC_STAT_AREA])[-keep:]; m = np.isin(lab, big).astype(np.uint8)
    m = _fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)

class TrackedPatch:
    """Something painted INTO one frame of the shot (deer on a far bank, a cottage light, a fisherman) and pinned to the
    plate for the whole shot. Swap `states` by time (e.g. head down -> head up at an event). Anything that comes in
    front of the bank (splash, spray, the hero) hides it: where the frame differs a lot from the tracked reference."""
    def __init__(self, ref, states, mattes, warps, scale=1.0, anchor=None, occ_thr=0.16, grass_band=(3, 8)):
        self.ref, self.warps, self.occ_thr, self.grass = ref, warps, occ_thr, grass_band
        self.states = list(zip(states, mattes))
        if anchor is None:
            ys, xs = np.nonzero(np.maximum.reduce(mattes) > 0.05); anchor = (0.5 * (xs.min() + xs.max()), float(ys.max()))
        self.S = np.array([[scale, 0, anchor[0] * (1 - scale)], [0, scale, anchor[1] * (1 - scale)], [0, 0, 1]], np.float32)
    def draw(self, c, i, state=0):
        H, W = c.shape[:2]; M = self.warps[i]; img, ma = self.states[state]
        A = (np.vstack([M, [0, 0, 1]]) @ self.S)[:2]
        p = cv2.warpAffine(img, A, (W, H)); a = cv2.warpAffine(ma, A, (W, H))[..., None]
        rw = cv2.warpAffine(self.ref, M, (W, H))
        occ = (np.abs(c - rw).max(2) - self.occ_thr) / 0.1 > 0.3
        occ = cv2.GaussianBlur(cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(np.float32), (0, 0), 1.0)[..., None]
        a = a * (1 - occ)
        if self.grass:
            ys, xs = np.nonzero(a[..., 0] > 0.5)
            if len(ys):
                rng = np.random.default_rng(5); bot = ys.max(); band = np.zeros((H, W), np.float32)
                for gx in range(xs.min(), xs.max() + 1, 2):
                    band[bot - int(rng.uniform(*self.grass)):bot + 2, gx:gx + 1] = 1
                a = a * (1 - cv2.GaussianBlur(band, (0, 0), 0.6)[..., None])
        c[:] = c * (1 - a) + p * a

# ---------------------------------------------------------------- the storybook grade
def storybook_grade(f, mask=None, keep=0.6, contrast=0.86, lift=0.025, paper=(1.02, 1.0, 0.94), paper_add=(0.02, 0.015, 0.0),
                    cream=(0.96, 0.92, 0.82), cream_mix=0.05, char_sat=1.06):
    """Mute the set like a printed picture book (keep 60% of its colour, softer contrast, lifted shadows, warm paper,
    a breath of cream) and keep the character(s) under `mask` (H,W float 0..1) at full colour, a touch richer.
    The set stays alive (its motion is untouched); only its colour steps back so the characters pop."""
    lum = (f * np.array([0.299, 0.587, 0.114], np.float32)).sum(2, keepdims=True)
    bg = f * keep + lum * (1 - keep); bg = 0.5 + (bg - 0.5) * contrast + lift
    bg = bg * np.array(paper, np.float32) + np.array(paper_add, np.float32)
    bg = bg * (1 - cream_mix) + np.array(cream, np.float32) * cream_mix
    if mask is None: return np.clip(bg, 0, 1)
    ch = lum + (f - lum) * char_sat; a = mask[..., None]
    return np.clip(bg * (1 - a) + ch * a, 0, 1)

def meadow_grade(f, mask=None, green_shift=4.0, sat=1.12, value=0.97, warm=(1.01, 1.0, 0.98)):
    """The alternative to storybook_grade when the set should stay lush: nudge yellow-greens toward green (hue in OpenCV
    degrees/2, weighted to the 15-45 band so skies, skin tones and reds don't move), lift saturation a little, keep the light.
    The character(s) under `mask` stay exactly as generated."""
    u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8[..., ::-1], cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s = hsv[..., 0], hsv[..., 1]
    w = np.clip(1 - np.abs(h - 30) / 15, 0, 1)                       # 1 at yellow-green, 0 outside 15..45
    hsv[..., 0] = h + green_shift * w
    hsv[..., 1] = np.clip(s * (1 + (sat - 1) * (0.4 + 0.6 * w)), 0, 255)
    hsv[..., 2] = hsv[..., 2] * value
    bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[..., ::-1].astype(np.float32) / 255 * np.array(warm, np.float32)
    if mask is None: return np.clip(bg, 0, 1)
    a = mask[..., None]
    return np.clip(bg * (1 - a) + f * a, 0, 1)

def mask_from_video(frame_u8_or_float, grow=2, blur=1.2):
    """A mask frame (white = character, e.g. from SAM 2 video) -> soft float mask, grown a little to cover the edge."""
    m = frame_u8_or_float[..., 0] if frame_u8_or_float.ndim == 3 else frame_u8_or_float
    m = (m > (127 if m.dtype == np.uint8 else 0.5)).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=grow)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), blur)

# ---------------------------------------------------------------- code characters onto the generated shot
def bw_matte(on_black, on_white):
    """Render a code layer twice, over pure black and pure white. alpha = 1 - (white - black), colour = the black render
    (already premultiplied). Exact alpha for soft edges, shadows and dust. Composite: out = pm + plate * (1 - alpha)."""
    alpha = np.clip(1 - (on_white - on_black).mean(2, keepdims=True), 0, 1)
    return on_black, alpha

def over(plate, pm, alpha): return pm + plate * (1 - alpha)
