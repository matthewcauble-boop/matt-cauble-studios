# tools/living: generate the shot, code the life

The method is in [LIVING-SCENES.md](../../LIVING-SCENES.md). This folder is the layer kit (numpy + OpenCV + ffmpeg).

```
python tools/living/demo.py hero_shot.mp4 alive.mp4                 # bees, seeds, motes, a flock + storybook grade
python tools/living/demo.py hero_shot.mp4 alive.mp4 --mask hero_mask.mp4   # hero keeps full colour
```

`life.py`, in pipeline order:

| Function | Use |
|---|---|
| `read_frames`, `Writer`, `on_twos` | float RGB frames in, H.264 out; quantise critter time to twos |
| `cut_sheet(path, names)` | drawings painted on a flat background -> RGBA sprites (paint the sheet in the plate's style) |
| `place(c, sprite, x, y, h, ...)` | bottom-centre placement; `by_w` for flappers, `sq` squash, `haze` for distance, `match` colour to the plate, `grass` blades over the feet |
| `shadow(c, x, y, w, k)` | soft contact shadow; fade `k` as the thing leaves the ground |
| `bee(c, loop_path(...), t)` | a ~4 px bee with a faint fading trail |
| `Drift(w, h).draw(c, t, drift_x)` | backlit seed heads + pollen motes with depth; `drift_x` for camera parallax |
| `bird_sprite(up)` | a procedural little bird when you have no sheet |
| `track_plate(ref, frames, roi)` | ORB + RANSAC similarity per frame on static scenery, gap-filled, 7-frame median |
| `diff_matte(edited, ref, box)` | cut what an image edit painted into a frame |
| `TrackedPatch(...).draw(c, i, state)` | the painted-in patch pinned to the plate, states swapped by time, hidden behind nearer motion, grass over the feet |
| `storybook_grade(f, mask)` | mute the set like a picture book; characters under the mask keep full colour |
| `mask_from_video(frame)` | a SAM 2 mask frame -> soft float mask |
| `bw_matte(on_black, on_white)`, `over` | exact alpha for code-animated characters rendered over black and white |

A painted-in far animal, end to end:

```python
import life as L
frames, fps = L.read_frames("shot.mp4"); ref = frames[k]            # the frame you sent to the image edit
edit = load("shot_with_deer.png"); edit_up = load("shot_with_deer_heads_up.png")
box = (600, 165, 790, 268)
brown = lambda e: (e[..., 0] > e[..., 1] - 0.01) & (e[..., 1] > e[..., 2]) & (e[..., 0] - e[..., 2] > 0.10)
m1, m2 = L.diff_matte(edit, ref, box, colour=brown), L.diff_matte(edit_up, ref, box, colour=brown)
warps, n_good = L.track_plate(ref, frames, roi=(480, 120, 854, 330))  # far bank, buildings, trees
deer = L.TrackedPatch(ref, [edit, edit_up], [m1, m2], warps, scale=1.25)
for i, f in enumerate(frames): deer.draw(f, i, state=1 if i / fps > t_splash else 0)
```
