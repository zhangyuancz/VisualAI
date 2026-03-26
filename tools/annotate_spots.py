#!/usr/bin/env python3
"""
annotate_spots.py — Interactive parking spot corner annotation tool.

Usage:
    python3 annotate_spots.py --rtsp rtsp://... --output spots.conf
    python3 annotate_spots.py --input frame.jpg --config spots.conf --output spots.conf
    python3 annotate_spots.py --input frame.jpg --config spots.conf --preview

Controls:
    LEFT CLICK   Add a corner to the current spot (up to 4)
    n            Next spot: confirm current spot (auto-closes if <4 corners)
    u            Undo: remove last completed spot
    d            Delete mode: click a spot to remove it, press d again to exit
    r            Re-mark mode: click a spot, then re-click its 4 corners
    s            Save and exit
    q / Esc     Quit without saving
"""

import argparse
import sys
import cv2
import numpy as np

WINDOW = "annotate_spots"
MIN_CLICK_DIST = 10   # px — reject clicks closer than this to last corner
CIRCLE_RADIUS = 6
THICKNESS = 2
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.55
HUD_THICK = 1
HUD_PAD = 8


# ──────────────────────────────────────────────────────────────────────────────
# Spot data structures
# ──────────────────────────────────────────────────────────────────────────────

class Spot:
    def __init__(self, idx: int, label: str, corners):
        self.idx = idx
        self.label = label
        self.corners = corners  # list of [x, y], length 4

    def clone(self):
        return Spot(self.idx, self.label, list(self.corners))


def save_conf(path, spots, img_w, img_h):
    """Write parking spot config in the same format rtsp_yolo reads."""
    with open(path, "w") as f:
        f.write(f"# parking_spots.conf  ({img_w}x{img_h})\n")
        f.write("# id  label  x1  y1  x2  y2  x3  y3  x4  y4\n")
        for s in spots:
            c = s.corners
            f.write(f"{s.idx:3d}   {s.label:<6s} "
                    f"{c[0][0]:4d} {c[0][1]:4d} {c[1][0]:4d} {c[1][1]:4d} "
                    f"{c[2][0]:4d} {c[2][1]:4d} {c[3][0]:4d} {c[3][1]:4d}\n")
    print(f"[annotate] Saved {len(spots)} spots → {path}")


def load_conf(path):
    """Parse a parking spot config file. Returns (list of Spot, img_w, img_h) or None."""
    spots = []
    img_w, img_h = None, None
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                if not img_w and "x" in line and "(" in line:
                    try:
                        tag = line.split("(")[1].split(")")[0]
                        img_w, img_h = map(int, tag.split("x"))
                    except Exception:
                        pass
                continue
            parts = line.split()
            if len(parts) < 9:
                print(f"[warn] line {lineno}: expected ≥9 fields, got {len(parts)}: {line}")
                continue
            try:
                idx   = int(parts[0])
                label = parts[1]
                corners = [[int(parts[i]), int(parts[i + 1])] for i in range(2, 10, 2)]
                spots.append(Spot(idx, label, corners))
            except ValueError as e:
                print(f"[warn] line {lineno}: parse error: {e}")
    return spots, img_w, img_h


def next_label(spots):
    """Suggest next available label (A1, A2, … B1 … or numeric)."""
    labels = {s.label for s in spots}
    for suffix in ["", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        for prefix in ["", "A", "B", "C", "D", "E", "F"]:
            cand = f"{prefix}{suffix}" if prefix or suffix else "1"
            if cand not in labels:
                return cand
    return str(len(spots) + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def point_in_spot(px, py, spot, margin=8):
    """Return True if (px,py) is inside or near the spot's quad."""
    c = spot.corners
    # Bounding-box quick reject
    xs = [p[0] for p in c]
    ys = [p[1] for p in c]
    if not (min(xs) - margin <= px <= max(xs) + margin and
            min(ys) - margin <= py <= max(ys) + margin):
        return False
    # Ray-casting point-in-polygon
    n = len(c)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = c[i]
        xj, yj = c[j]
        if ((yi > py) != (yj > py)) and \
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def shoelace(corners):
    """Polygon area via shoelace formula."""
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) * 0.5


def aabb_clip(corners, x1, y1, x2, y2):
    """
    Sutherland-Hodgman polygon clipping against a single axis-aligned rect.
    corners: list of [x, y]
    Returns list of clipped vertices.
    """
    out = list(corners)
    edges = [
        ([x1, y1], [x1, y2]),   # left
        ([x1, y2], [x2, y2]),   # top
        ([x2, y2], [x2, y1]),   # right
        ([x2, y1], [x1, y1]),   # bottom
    ]
    for clip_a, clip_b in edges:
        if not out:
            return []
        new_out = []
        n = len(out)
        for i in range(n):
            cur = out[i]
            nxt = out[(i + 1) % n]
            cur_in  = (clip_b[0] - clip_a[0]) * (cur[1] - clip_a[1]) >= \
                      (clip_b[1] - clip_a[1]) * (cur[0] - clip_a[0])
            nxt_in  = (clip_b[0] - clip_a[0]) * (nxt[1] - clip_a[1]) >= \
                      (clip_b[1] - clip_a[1]) * (nxt[0] - clip_a[0])
            if cur_in:
                new_out.append(cur)
                if not nxt_in:
                    # compute intersection
                    dx = nxt[0] - cur[0]
                    dy = nxt[1] - cur[1]
                    denom = (clip_b[0] - clip_a[0]) * dy - (clip_b[1] - clip_a[1]) * dx
                    if abs(denom) > 1e-9:
                        t = ((clip_a[0] - cur[0]) * dy - (clip_a[1] - cur[1]) * dx) / denom
                        t = max(0.0, min(1.0, t))
                        new_out.append([cur[0] + t * dx, cur[1] + t * dy])
            elif nxt_in:
                # compute intersection
                dx = nxt[0] - cur[0]
                dy = nxt[1] - cur[1]
                denom = (clip_b[0] - clip_a[0]) * dy - (clip_b[1] - clip_a[1]) * dx
                if abs(denom) > 1e-9:
                    t = ((clip_a[0] - cur[0]) * dy - (clip_a[1] - cur[1]) * dx) / denom
                    t = max(0.0, min(1.0, t))
                    new_out.append([cur[0] + t * dx, cur[1] + t * dy])
        out = new_out
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_spot(img, spot, color, thickness=THICKNESS, label_bg=True):
    """Draw a closed quad outline on the image (in-place)."""
    c = [np.array(p, dtype=np.int32) for p in spot.corners]
    c.append(c[0])
    pts = np.array(c, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    # Label
    if spot.label:
        x, y = c[0]
        text = spot.label
        (tw, th), _ = cv2.getTextSize(text, HUD_FONT, HUD_SCALE * 0.9, 1)
        bg_x1, bg_y1 = x - 2, y - th - 4
        bg_x2, bg_y2 = x + tw + 4, y + 2
        if label_bg:
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          color, -1)
            cv2.putText(img, text, (x, y - 2), HUD_FONT, HUD_SCALE * 0.9,
                        (255, 255, 255) if label_bg else color, 1)
        else:
            cv2.putText(img, text, (x, y - 2), HUD_FONT, HUD_SCALE * 0.9,
                        color, 1)


def draw_in_progress(img, corners, nxt_idx):
    """Draw the current in-progress spot (dashed style)."""
    # Filled outline as dashed: draw every other pixel segment
    all_pts = list(corners)
    color = (0, 255, 255)
    for i in range(len(all_pts)):
        p1 = all_pts[i]
        p2 = all_pts[(i + 1) % len(all_pts)]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1:
            continue
        steps = int(length / 4)
        for s in range(steps):
            t = s / steps
            x1 = int(p1[0] + dx * t)
            y1 = int(p1[1] + dy * t)
            t2 = (s + 0.5) / steps
            x2 = int(p1[0] + dx * t2)
            y2 = int(p1[1] + dy * t2)
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Corner circles and numbers
    for i, (cx, cy) in enumerate(all_pts):
        cv2.circle(img, (cx, cy), CIRCLE_RADIUS, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), CIRCLE_RADIUS, (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (cx - 5, cy + 4), HUD_FONT, 0.45,
                    (0, 0, 0), 1)

    # Next corner ghost at cursor
    if len(corners) < 4:
        hint = f"click {len(corners)+1}/4"
        cv2.putText(img, hint, (10, img.shape[0] - 10), HUD_FONT, HUD_SCALE, (0, 255, 255), 1)


def draw_hud(img, spots, current_corners, mode, delete_target, remark_target, img_w, img_h):
    """Draw the on-screen HUD."""
    h = img.shape[0]

    # Top-left: spot count + status
    mode_str = {
        "normal": "NORMAL",
        "delete": "DELETE ← click spot to remove",
        "remark": "REMARK ← click spot to re-mark",
    }.get(mode, mode)
    lines = [
        f"[{len(spots)} spots]  mode: {mode_str}",
        f"Press '?' for help" if mode == "normal" else "",
    ]
    for i, line in enumerate(lines):
        if line:
            cv2.putText(img, line, (HUD_PAD, 18 + i * 20), HUD_FONT, HUD_SCALE, (255, 255, 0), HUD_THICK)

    # Bottom instructions
    hints = {
        "normal": "CLICK: add corner (4=max) | n: next spot | u: undo | d: delete | r: remark | s: save | q: quit",
        "delete": "CLICK: remove spot | d: exit delete mode | q: quit",
        "remark": "CLICK: first corner of replacement | Esc: cancel | q: quit",
    }
    hint = hints.get(mode, "")
    cv2.putText(img, hint, (HUD_PAD, h - HUD_PAD), HUD_FONT, HUD_SCALE * 0.85, (180, 180, 180), HUD_THICK)


# ──────────────────────────────────────────────────────────────────────────────
# Label input (blocking, using a small named window)
# ──────────────────────────────────────────────────────────────────────────────

def ask_label(default_label):
    """Popup window for entering a spot label."""
    cv2.namedWindow("enter_label")
    cv2.moveWindow("enter_label", 100, 100)

    edit = [default_label]

    def on_change(pos):
        pass  # trackbar could be used for future numeric-only mode

    cv2.createTrackbar("label", "enter_label", 0, 1, on_change)
    cv2.setTrackbarPos("label", "enter_label", 0)

    def on_key(key):
        code = key & 0xFF
        if code == 13:   # Enter
            return True
        elif code == 27: # Escape
            return False
        elif code == 8:  # Backspace
            edit[0] = edit[0][:-1]
        elif 32 <= code < 127:
            edit[0] += chr(code)
        cv2.displayOverlay("enter_label", f"Label: {edit[0]}   [Enter=ok Esc=cancel]", 1)
        return None

    while True:
        cv2.displayOverlay("enter_label", f"Label: {edit[0]}   [Enter=ok Esc=cancel]", 1)
        key = cv2.waitKey(100)
        if key != -1:
            done = on_key(key)
            if done is not None:
                cv2.destroyWindow("enter_label")
                return edit[0] if done else None


# ──────────────────────────────────────────────────────────────────────────────
# RTSP frame capture helper
# ──────────────────────────────────────────────────────────────────────────────

def capture_rtsp_frame(url, timeout_ms=5000):
    """Capture one frame from an RTSP stream using OpenCV."""
    import threading, queue, time

    result = [None]  # [frame]
    q = queue.Queue(maxsize=1)
    stop = [False]

    def grab():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MS, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, 5000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            q.put(frame)
        else:
            q.put(None)

    t = threading.Thread(target=grab, daemon=True)
    t.start()
    t.join(timeout_ms / 1000)
    if not t.is_alive():
        try:
            return q.get_nowait()
        except queue.Empty:
            pass
    print("[error] Timeout capturing frame from RTSP stream.", file=sys.stderr)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main annotation loop
# ──────────────────────────────────────────────────────────────────────────────

def run(img, img_path, conf_path, output_path, preview_only):
    # img: numpy array (H, W, 3) in BGR
    img_h, img_w = img.shape[:2]
    display = img.copy()

    # State
    spots = []           # list of Spot
    current = []         # current in-progress corners: list of [x, y]
    mode = "normal"      # normal | delete | remark
    remark_idx = None    # index of spot being re-marked (in remark mode)
    delete_target = None
    cursor = [0, 0]      # current mouse position
    running = True

    def mouse_cb(event, x, y, flags, param):
        nonlocal mode, delete_target, remark_idx, current, spots, running

        if event == cv2.EVENT_MOUSEMOVE:
            cursor[0], cursor[1] = x, y
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if mode == "delete":
            # Find clicked spot
            for i, s in enumerate(spots):
                if point_in_spot(x, y, s):
                    spots.pop(i)
                    print(f"[annotate] Deleted spot {i}")
                    mode = "normal"
                    break
            return

        if mode == "remark":
            # First click in remark mode: identify which spot
            if remark_idx is None:
                for i, s in enumerate(spots):
                    if point_in_spot(x, y, s):
                        remark_idx = i
                        current = []
                        print(f"[annotate] Re-marking spot {i} ({s.label}) — click 4 corners")
                        return
                # clicked outside any spot: cancel
                mode = "normal"
                remark_idx = None
                current = []
                return
            # Subsequent clicks: add corners
            if len(current) == 0 or \
               ((x - current[-1][0]) ** 2 + (y - current[-1][1]) ** 2) ** 0.5 > MIN_CLICK_DIST:
                current.append([x, y])
                if len(current) == 4:
                    label = spots[remark_idx].label
                    spots[remark_idx] = Spot(remark_idx, label, list(current))
                    print(f"[annotate] Re-marked spot {remark_idx} ({label})")
                    mode = "normal"
                    remark_idx = None
                    current = []
            return

        # Normal mode: add corners
        if len(current) == 4:
            return  # already full, wait for 'n'
        if len(current) > 0 and \
           ((x - current[-1][0]) ** 2 + (y - current[-1][1]) ** 2) ** 0.5 < MIN_CLICK_DIST:
            return  # too close to last click

        current.append([x, y])
        if len(current) == 4:
            print("[annotate] 4 corners placed — press 'n' to confirm")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    while running:
        display = img.copy()

        # Draw completed spots
        for i, s in enumerate(spots):
            color = (0, 0, 255) if (mode == "delete" and point_in_spot(*cursor, s)) \
                    else (0, 200, 0)
            draw_spot(display, s, color)

        # Draw in-progress spot
        if current:
            draw_in_progress(display, current, len(spots))

        # Draw cursor crosshair
        cv2.line(display, (cursor[0] - 8, cursor[1]), (cursor[0] + 8, cursor[1]), (200, 200, 200), 1)
        cv2.line(display, (cursor[0], cursor[1] - 8), (cursor[0], cursor[1] + 8), (200, 200, 200), 1)

        draw_hud(display, spots, current, mode, delete_target, remark_idx, img_w, img_h)

        # Status bar
        if current:
            status = f"  [{len(current)}/4 corners]  next label: '{next_label(spots)}'  |  'n'=confirm  'u'=undo"
        else:
            status = f"  {len(spots)} spot(s)  |  next label: '{next_label(spots)}'"
        cv2.displayOverlay(WINDOW, status)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            if output_path:
                save_conf(output_path, spots, img_w, img_h)
                if preview_only:
                    print("[annotate] Saved (still in preview mode — 'q' to exit)")
                else:
                    running = False
            else:
                print("[warn] No output path set (--output required to save)")
            continue

        if key in (ord("q"), 27):
            if output_path and spots:
                save_conf(output_path, spots, img_w, img_h)
                print("[annotate] Saved on quit.")
            running = False
            continue

        if key == ord("n"):
            if 2 <= len(current) < 4:
                # Auto-close: duplicate first corner to make 4 corners
                current.append(current[0])
            if len(current) == 4:
                label = ask_label(next_label(spots))
                if label:
                    spots.append(Spot(len(spots), label, list(current)))
                    print(f"[annotate] Added spot {len(spots)-1}: {label}")
                current = []
            elif len(current) == 0:
                print("[annotate] No corners to confirm — click first")
            else:
                print(f"[annotate] Need at least 2 corners (got {len(current)})")
            continue

        if key == ord("u"):
            if spots:
                removed = spots.pop()
                print(f"[annotate] Undo: removed spot {removed.idx} ({removed.label})")
            else:
                print("[annotate] Nothing to undo")
            continue

        if key == ord("d"):
            if mode == "delete":
                mode = "normal"
                print("[annotate] Delete mode OFF")
            else:
                mode = "delete"
                print("[annotate] Delete mode ON — click a spot to remove it")
            continue

        if key == ord("r"):
            if mode == "remark":
                mode = "normal"
                remark_idx = None
                current = []
                print("[annotate] Remark mode OFF")
            else:
                mode = "remark"
                remark_idx = None
                current = []
                print("[annotate] Remark mode ON — click a spot to re-mark")
            continue

        if key == ord("?"):
            print(__doc__)

    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  metavar="IMG",
                    help="Image file (JPEG/PNG) to load")
    ap.add_argument("--rtsp",   metavar="URL",
                    help="RTSP URL — capture one frame as input")
    ap.add_argument("--config", metavar="CONF",
                    help="Load an existing config to resume editing / preview")
    ap.add_argument("--output", metavar="CONF",
                    help="Output config file path (required to save)")
    ap.add_argument("--preview", action="store_true",
                    help="Preview mode: load image+config and display, no editing")
    args = ap.parse_args()

    # ── Acquire input image ──
    if args.input:
        img = cv2.imread(args.input)
        if img is None:
            print(f"[error] Cannot load image: {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"[annotate] Loaded image: {args.input}  ({img.shape[1]}x{img.shape[0]})")
        img_path = args.input
    elif args.rtsp:
        img = capture_rtsp_frame(args.rtsp)
        if img is None:
            sys.exit(1)
        print(f"[annotate] Captured frame from RTSP: {args.rtsp}  ({img.shape[1]}x{img.shape[0]})")
        img_path = args.rtsp
    else:
        print("[error] --input or --rtsp required", file=sys.stderr)
        sys.exit(1)

    # ── Load existing config (optional) ──
    existing_spots = []
    if args.config:
        loaded = load_conf(args.config)
        if loaded is None:
            print(f"[error] Cannot load config: {args.config}", file=sys.stderr)
            sys.exit(1)
        existing_spots, cw, ch = loaded
        print(f"[annotate] Loaded {len(existing_spots)} spots from {args.config}")
        if cw and ch and (cw != img.shape[1] or ch != img.shape[0]):
            print(f"[warn] Config resolution {cw}x{ch} differs from image {img.shape[1]}x{img.shape[0]}")
        img_path = args.config

    # ── Preview mode ──
    if args.preview:
        display = img.copy()
        for s in existing_spots:
            draw_spot(display, s, (0, 200, 0))
        cv2.imshow(WINDOW, display)
        cv2.displayOverlay(WINDOW, f"Preview: {len(existing_spots)} spots  (q to quit)")
        print(f"[annotate] Preview mode — {len(existing_spots)} spots loaded")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        sys.exit(0)

    # ── Interactive annotation ──
    output = args.output
    if not output and not args.preview:
        print("[warn] No --output specified; use 's' in the UI to save once set")

    run(img, img_path, args.config or "", output or "", False)


if __name__ == "__main__":
    main()
