"""Lightweight staff-line + notehead detector for camera-distorted PrIMuS.

Produces:
  - an upscaled annotated overlay PNG (staff grid + notehead pitch labels)
  - stdout list of detected noteheads: index, x, y, estimated pitch

This is only a pitch-anchoring aid for manual transcription. Rhythm, rests,
flags, beams, clef and key are still read visually from the image. Treble clef
(G2) is assumed for pitch mapping; correct manually if the clef differs.
"""
import sys
import os
import numpy as np
import cv2


PITCH_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]


def diatonic_to_name(step_from_e4):
    # step 0 == E4 (bottom line, treble). Going up the diatonic scale.
    # Absolute diatonic index of E4:
    base = 4 * 7 + 2  # octave4 * 7 + E(index2)  -> E is index 2 in C,D,E,F,G,A,B
    idx = base + step_from_e4
    octave = idx // 7
    letter = PITCH_LETTERS[idx % 7]
    return f"{letter}{octave}"


def detect(img_path, out_path, clef="G2", scale=3):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"cannot read {img_path}")
    h, w = gray.shape
    # ink = dark -> make ink white in `bw`
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)

    # ---- estimate staff line spacing via vertical run-lengths of ink ----
    black_runs = []
    white_runs = []
    for x in range(0, w, max(1, w // 200)):
        col = bw[:, x] > 0
        run = 0
        prev = False
        for v in col:
            if v == prev:
                run += 1
            else:
                if prev:
                    black_runs.append(run)
                else:
                    white_runs.append(run)
                run = 1
                prev = v
    # staff line thickness ~ mode of small black runs; spacing ~ mode of white runs
    def mode_small(vals, lo, hi):
        vals = [v for v in vals if lo <= v <= hi]
        if not vals:
            return None
        counts = np.bincount(vals)
        return int(np.argmax(counts))

    line_thick = mode_small(black_runs, 1, max(2, h // 20)) or 2
    space = mode_small(white_runs, 2, h) or (h // 6)
    staff_d = line_thick + space  # distance between adjacent line centers

    # ---- staff line mask via horizontal morphology ----
    hk = max(15, w // 25)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))

    # ---- notehead mask: keep filled elliptical blobs, drop lines/stems ----
    # remove horizontal lines first
    no_lines = cv2.subtract(bw, horiz)
    # remove thin vertical stems
    vk = max(3, int(staff_d * 1.2))
    vert = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    heads = cv2.subtract(no_lines, vert)
    ek = max(3, int(space * 0.7)) | 1
    heads = cv2.morphologyEx(heads, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)))
    heads = cv2.morphologyEx(heads, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(heads, 8)
    min_area = (space * space) * 0.28
    max_area = (staff_d * staff_d) * 4.0
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw_w = stats[i, cv2.CC_STAT_WIDTH]
        bw_h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if bw_w > staff_d * 3 or bw_h > staff_d * 3:
            continue
        cx, cy = centroids[i]
        blobs.append((cx, cy, area, bw_w, bw_h))
    blobs.sort(key=lambda b: b[0])

    def local_lines(xc):
        x0 = int(max(0, xc - staff_d))
        x1 = int(min(w, xc + staff_d))
        strip = horiz[:, x0:x1]
        rowsum = strip.sum(axis=1)
        thr = rowsum.max() * 0.35 if rowsum.max() > 0 else 1
        ys = np.where(rowsum > thr)[0]
        if len(ys) == 0:
            return None
        # cluster consecutive ys
        clusters = []
        cur = [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= max(2, line_thick + 1):
                cur.append(y)
            else:
                clusters.append(int(np.mean(cur)))
                cur = [y]
        clusters.append(int(np.mean(cur)))
        return clusters

    results = []
    for (cx, cy, area, bwd, bhd) in blobs:
        lines = local_lines(cx)
        if not lines or len(lines) < 2:
            results.append((cx, cy, None))
            continue
        lines = sorted(lines)
        # derive a regular grid: topmost line + median spacing
        diffs = np.diff(lines)
        d = float(np.median(diffs)) if len(diffs) else staff_d
        if d < 2:
            d = staff_d
        top = lines[0]
        bottom_line = top + 4 * d  # E4 line for treble
        # position in half-spaces above bottom line
        steps = (bottom_line - cy) / (d / 2.0)
        step_int = int(round(steps))
        results.append((cx, cy, step_int))

    # ---- render overlay ----
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    out_lines = []
    for idx, (cx, cy, step) in enumerate(results):
        X, Y = int(cx * scale), int(cy * scale)
        cv2.circle(canvas, (X, Y), int(space * 0.5 * scale), (0, 0, 255), 2)
        if step is None:
            name = "?"
        elif clef == "G2":
            name = diatonic_to_name(step)
        elif clef == "C1":
            # C1 clef: bottom line = ? For C-clef on line1, that line = C4.
            # bottom line (index0 from bottom) is line1 -> C4 => step0 => C4
            base = 4 * 7 + 0
            i2 = base + step
            name = f"{PITCH_LETTERS[i2 % 7]}{i2 // 7}"
        else:
            name = diatonic_to_name(step)
        cv2.putText(canvas, name, (X - 10, Y - int(space * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        out_lines.append(f"{idx:02d}  x={cx:6.1f}  y={cy:6.1f}  step={step}  pitch={name}")

    cv2.imwrite(out_path, canvas)
    print(f"size={w}x{h} line_thick={line_thick} space={space} staff_d={staff_d} nheads={len(results)}")
    for l in out_lines:
        print(l)


if __name__ == "__main__":
    img = sys.argv[1]
    out = sys.argv[2]
    clef = sys.argv[3] if len(sys.argv) > 3 else "G2"
    detect(img, out, clef=clef)
