import json
import os
import numpy as np
import cv2

CAMERAS = [1, 2, 3, 4, 5]
CANON = (1600, 1200)
PLAN_SIZE = (180, 440)
CALIB_FILE = "calibration.json"


def video_path(root, cam):
    return os.path.join(root, f"camera_new_{cam}.mkv")


def fmt_time(frame, fps):
    ms = int(round(frame / fps * 1000.0))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


def _geom():
    W, H = CANON
    return np.array([W / 2.0, H / 2.0]), float(np.hypot(W, H) / 2.0)


def undistort_px(pts, k):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, R = _geom()
    d = pts - c
    r2 = (np.linalg.norm(d, axis=1) / R) ** 2
    return c + d / (1.0 + k * r2)[:, None]


def distort_px(pts, k):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, R = _geom()
    d = pts - c
    ru = np.linalg.norm(d, axis=1) / R
    if abs(k) < 1e-9:
        return pts.copy()
    disc = np.clip(1.0 - 4.0 * k * ru ** 2, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(ru > 1e-9, (1.0 - np.sqrt(disc)) / (2.0 * k * ru), 0.0)
    scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
    return c + d * scale[:, None]


def undistort_image(img, k):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([gx.ravel() * CANON[0] / w, gy.ravel() * CANON[1] / h], 1)
    src = distort_px(pts, k)
    mapx = (src[:, 0] * w / CANON[0]).reshape(h, w).astype(np.float32)
    mapy = (src[:, 1] * h / CANON[1]).reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


class Calibration:
    def __init__(self, data=None):
        self.data = data or {"k1": 0.0, "plan_size": list(PLAN_SIZE),
                             "cameras": {str(c): {"pairs": [], "ghost_polygons": [], "weight": 1.0}
                                         for c in CAMERAS},
                             "walkable": [], "obstacles": []}
        self._H = {}
        self._mask_cache = {}
        self._ghost = {}

    @staticmethod
    def load(path=CALIB_FILE):
        if not os.path.exists(path):
            return Calibration()
        with open(path, "r", encoding="utf-8") as f:
            cal = Calibration(json.load(f))
        for c in CAMERAS:
            cal.data["cameras"].setdefault(str(c), {"pairs": [], "ghost_polygons": [], "weight": 1.0})
        return cal

    def save(self, path=CALIB_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)

    @property
    def k1(self):
        return float(self.data.get("k1", 0.0))

    def cam(self, c):
        return self.data["cameras"][str(c)]

    def weight(self, c):
        return float(self.cam(c).get("weight", 1.0))

    def homography(self, c):
        if c in self._H:
            return self._H[c]
        pairs = self.cam(c)["pairs"]
        H = None
        if len(pairs) >= 4:
            src = np.array([[p["img"][0] * CANON[0], p["img"][1] * CANON[1]] for p in pairs])
            src = undistort_px(src, self.k1)
            dst = np.array([p["plan"] for p in pairs], np.float64)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(pairs) > 4 else 0, 4.0)
        self._H[c] = H
        return H

    def reproj_error(self, c):
        H = self.homography(c)
        pairs = self.cam(c)["pairs"]
        if H is None or not pairs:
            return None
        src = undistort_px([[p["img"][0] * CANON[0], p["img"][1] * CANON[1]] for p in pairs], self.k1)
        pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(pr - np.array([p["plan"] for p in pairs]), axis=1)))

    def is_calibrated(self, c):
        return self.homography(c) is not None

    def to_plan(self, c, pts_px):
        H = self.homography(c)
        u = undistort_px(pts_px, self.k1)
        return cv2.perspectiveTransform(u.reshape(-1, 1, 2), H).reshape(-1, 2)

    def plan_to_cam(self, c, plan_pts):
        H = self.homography(c)
        p = cv2.perspectiveTransform(np.asarray(plan_pts, np.float64).reshape(-1, 1, 2),
                                     np.linalg.inv(H)).reshape(-1, 2)
        return distort_px(p, self.k1)

    def walkable_mask(self, margin=0):
        key = int(margin)
        if key in self._mask_cache:
            return self._mask_cache[key]
        w, h = self.data.get("plan_size", PLAN_SIZE)
        m = np.zeros((h, w), np.uint8)
        for poly in self.data.get("walkable", []):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
        for poly in self.data.get("obstacles", []):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 0)
        if margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * key + 1, 2 * key + 1))
            m = cv2.dilate(m, k)
        self._mask_cache[key] = m
        return m

    def is_walkable(self, xy, margin=0):
        if not self.data.get("walkable"):
            return True
        m = self.walkable_mask(margin)
        x, y = int(round(xy[0])), int(round(xy[1]))
        return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0

    def ghost_mask(self, c):
        if c not in self._ghost:
            m = np.zeros((CANON[1], CANON[0]), np.uint8)
            for poly in self.cam(c).get("ghost_polygons", []):
                pts = (np.array(poly) * np.array(CANON)).round().astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
            self._ghost[c] = m
        return self._ghost[c]


def scan_video(path, cache_dir=".cache", verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(path)
    cpath = os.path.join(cache_dir, os.path.basename(path) + ".pts.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        if int(z["size"]) == st.st_size and int(z["mtime"]) == int(st.st_mtime):
            return z["pts"], float(z["nom_fps"]), int(z["nom_count"])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("не открывается " + path)
    nom_fps, nom_count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pts = []
    while cap.grab():
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if verbose and len(pts) % 5000 == 0:
            print(f"  [scan] {os.path.basename(path)}: {len(pts)} frames", flush=True)
    cap.release()
    pts = np.array(pts, np.float64)
    if len(pts) == 0:
        raise SystemExit("пустое видео " + path)
    if len(pts) > 2 and (np.mean(np.diff(pts) > 0) < 0.9):
        fps = nom_fps if nom_fps > 0 else 25.0
        print(f"  [scan] {os.path.basename(path)}: using nominal {fps} fps")
        pts = np.arange(len(pts)) * 1000.0 / fps
    pts = np.maximum.accumulate(pts)
    np.savez(cpath, pts=pts, size=st.st_size, mtime=int(st.st_mtime), nom_fps=nom_fps, nom_count=nom_count)
    return pts, nom_fps, nom_count


def scan_all(paths):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(paths)) as ex:
        futs = {c: ex.submit(scan_video, p) for c, p in paths.items()}
        return {c: f.result() for c, f in futs.items()}


def build_timeline(pts_by_cam, target_fps, normalize_spans=False):
    rel, span = {}, {}
    for c, p in pts_by_cam.items():
        p = np.asarray(p, np.float64)
        dt = float(np.median(np.diff(p))) if len(p) > 1 else 40.0
        rel[c] = p - p[0]
        span[c] = rel[c][-1] + dt
    D = min(span.values())
    N = int(np.floor(D / 1000.0 * target_fps + 1e-6))
    t = np.arange(N) * 1000.0 / target_fps
    idx, err = {}, {}
    for c, r in rel.items():
        tt = t * (span[c] / D) if normalize_spans else t
        j = np.clip(np.searchsorted(r, tt), 1, len(r) - 1) if len(r) > 1 else np.zeros(N, int)
        if len(r) > 1:
            j = np.where(np.abs(r[j - 1] - tt) <= np.abs(r[j] - tt), j - 1, j)
        idx[c] = j.astype(np.int32)
        err[c] = float(np.abs(r[j] - tt).max()) if N else 0.0
    return N, idx, err


def seek_read(cap, pts_ms, i, back_ms=1500.0):
    n = len(pts_ms)
    for attempt in range(4):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(pts_ms[i]) - back_ms * (2 ** attempt)))
        for _ in range(900):
            ok, fr = cap.read()
            if not ok:
                break
            j = int(np.searchsorted(pts_ms, cap.get(cv2.CAP_PROP_POS_MSEC) - 0.5))
            if j >= i:
                return fr, min(j, n - 1)
    return None, -1
