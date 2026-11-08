import argparse
import pickle
import queue
import threading
import time
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from common import *

ANKLES = (15, 16)


class Reader(threading.Thread):

    def __init__(self, path, src_idx, pts_ms, keep_full=False, qsize=4):
        super().__init__(daemon=True)
        self.path, self.idx, self.pts, self.keep_full = path, src_idx, pts_ms, keep_full
        self.q = queue.Queue(qsize)
        self.start()

    def _pack(self, fr):
        return cv2.resize(fr, CANON, interpolation=cv2.INTER_AREA), (fr if self.keep_full else None)

    def run(self):
        cap = cv2.VideoCapture(self.path)
        cur, last = -1, None
        for i, s in enumerate(self.idx):
            s = int(s)
            new = (s != cur)
            if i == 0 and s > 0:      
                fr, j = seek_read(cap, self.pts, s)
                if fr is None:
                    break
                cur, s, last, new = j, j, self._pack(fr), True
            elif new:
                for _ in range(s - cur - 1):
                    cap.grab()
                ok, fr = cap.read()
                cur = s
                if ok:
                    last = self._pack(fr)
                elif last is None:
                    break
                else:
                    new = False
            elif last is None:
                break
            self.q.put((last[0], last[1], s, new))
        self.q.put(None)


class Dets:

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1) if len(x1) else np.zeros((0, 4), np.float32)

    @property
    def xywhr(self):
        return self.xywh

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


def parse_result(r):
    n = len(r.boxes)
    if n == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
    b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    cf = r.boxes.conf.cpu().numpy().astype(np.float32)
    kps = [None] * n
    if getattr(r, "keypoints", None) is not None and r.keypoints.xy.numel():
        kxy = r.keypoints.xy.cpu().numpy()
        kcf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32)
        kps = [(kxy[i], kcf[i]) for i in range(n)]
    return b, cf, kps


def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def detect_roi(model, full, roi, args, device, kw):
    Hf, Wf = full.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in (roi[0] * Wf, roi[1] * Hf, roi[2] * Wf, roi[3] * Hf)]
    crop = full[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
    r = model.predict([crop], imgsz=args.zoom_imgsz, conf=args.conf, iou=args.nms_iou, classes=[0],
                      device=device, verbose=False, **kw)[0]
    b, cf, kps = parse_result(r)
    cw, ch = x1 - x0, y1 - y0
    sx, sy = CANON[0] / Wf, CANON[1] / Hf
    off = np.array([x0, y0], np.float32); sc = np.array([sx, sy], np.float32)
    ob, oc, ok_ = [], [], []
    for bb, c, k in zip(b, cf, kps):
        if (bb[0] < 3 and x0 > 0) or (bb[1] < 3 and y0 > 0) or (bb[2] > cw - 3 and x1 < Wf) or (bb[3] > ch - 3 and y1 < Hf):
            continue
        ob.append(np.concatenate([(bb[:2] + off) * sc, (bb[2:] + off) * sc])); oc.append(c)
        ok_.append(None if k is None else ((k[0] + off) * sc, k[1]))
    return np.array(ob, np.float32).reshape(-1, 4), np.array(oc, np.float32), ok_


def merge_dets(parts, iou_thr=0.6):
    b = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 4), np.float32)
    c = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.float32)
    k = [x for p in parts for x in p[2]]
    if len(b) <= 1:
        return b, c, k
    order = np.argsort(-c); keep = []
    iou = iou_matrix(b, b)
    for i in order:
        if all(iou[i, j] <= iou_thr for j in keep):
            keep.append(i)
    keep = sorted(keep)
    return b[keep], c[keep], [k[i] for i in keep]


def split_hugs(boxes, confs, kps, ar_thr):
    ob, oc, ok_ = [], [], []
    for b, c, k in zip(boxes, confs, kps):
        w, h = b[2] - b[0], b[3] - b[1]
        if ar_thr > 0 and h > 0 and w / h > ar_thr:
            mid = (b[0] + b[2]) / 2; ov = 0.08 * w
            ob += [[b[0], b[1], mid + ov, b[3]], [mid - ov, b[1], b[2], b[3]]]
            oc += [c * 0.9, c * 0.9]; ok_ += [None, None]
        else:
            ob.append(list(b)); oc.append(c); ok_.append(k)
    return np.array(ob, np.float32).reshape(-1, 4), np.array(oc, np.float32), ok_


def foot_of(box, kp):
    if kp is not None:
        xy, cf = kp
        pts = [xy[i] for i in ANKLES if cf[i] > 0.5]
        if pts:
            return np.mean(pts, 0), True
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y2], np.float32), False


def filter_detections(cam, cal, boxes, kps, args):
    W, H = CANON
    keep, plan_xy, foot_ok = [], [], []
    gm = cal.ghost_mask(cam)
    for i, (b, k) in enumerate(zip(boxes, kps)):
        foot, ok = foot_of(b, k)
        # TV
        if gm.any():
            fx, fy = int(np.clip(foot[0], 0, W - 1)), int(np.clip(foot[1], 0, H - 1))
            x1, y1, x2, y2 = [int(v) for v in b]
            sub = gm[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
            if gm[fy, fx] > 0 or (sub.size and sub.mean() >= args.mask_ratio):
                continue
        # Mirror
        if cal.is_calibrated(cam):
            p = cal.to_plan(cam, foot[None])[0]
            if not cal.is_walkable(p, args.walk_margin if ok else args.walk_margin_soft):
                continue
        else:
            p = np.array([np.nan, np.nan])
        keep.append(i); plan_xy.append(p); foot_ok.append(ok)
    return keep, plan_xy, foot_ok


def parse_zoom(items, cal, cams):
    rois = {c: [list(r) for r in cal.cam(c).get("zoom_rois", [])] for c in cams}
    for it in items or []:
        c, v = it.split(":")
        rois.setdefault(int(c), []).append([float(x) for x in v.split(",")])
    return rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--calib", default=CALIB_FILE)
    ap.add_argument("--out", default="local_tracks.pkl")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--nms-iou", type=float, default=0.85)
    ap.add_argument("--tracker", default="botsort_static.yaml")
    ap.add_argument("--zoom", action="append")
    ap.add_argument("--zoom-imgsz", type=int, default=1280)
    ap.add_argument("--hug-ar", type=float, default=0.0)
    ap.add_argument("--mask-ratio", type=float, default=0.8)
    ap.add_argument("--walk-margin", type=int, default=6)
    ap.add_argument("--walk-margin-soft", type=int, default=25)
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--normalize-spans", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-uncalibrated", action="store_true")
    ap.add_argument("--cameras", type=int, nargs="+", default=CAMERAS)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import IterableSimpleNamespace
    try:
        from ultralytics.utils import YAML
        load_yaml = YAML.load
    except ImportError:
        from ultralytics.utils import yaml_load as load_yaml
    from ultralytics.trackers.bot_sort import BOTSORT

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cal = Calibration.load(args.calib)
    for c in args.cameras:
        if not cal.is_calibrated(c) and not args.allow_uncalibrated:
            raise SystemExit(f"camera {c} not calibrated")
    rois = parse_zoom(args.zoom, cal, args.cameras)
    print(f"[init] device={device}, detailed search zones: { {c: r for c, r in rois.items() if r} }")
    model = YOLO(args.model)
    kw = {"quantize": 16} if device != "cpu" else {}

    scan = scan_all({c: video_path(args.root, c) for c in args.cameras})
    pts = {c: scan[c][0] for c in args.cameras}
    size = {}
    for c in args.cameras:
        cap = cv2.VideoCapture(video_path(args.root, c)); size[c] = (int(cap.get(3)), int(cap.get(4))); cap.release()
    eff_fps = {c: (len(pts[c]) - 1) * 1000.0 / max(pts[c][-1] - pts[c][0], 1e-6) for c in args.cameras}
    fps = float(args.target_fps or round(float(np.median(list(eff_fps.values()))), 2))
    N_total, src_idx, terr = build_timeline(pts, fps, args.normalize_spans)
    spans = {c: (pts[c][-1] - pts[c][0]) / 1000.0 for c in args.cameras}
    for c in args.cameras:
        d_ = np.diff(src_idx[c]) if N_total > 1 else np.zeros(0)
        gaps = np.diff(pts[c]); gmax = gaps.max() if len(gaps) else 0.0
    print(f"[sync] fps: {fps}, frames: {N_total}")
    if max(spans.values()) - min(spans.values()) > 1.0:
        print("[sync] Attention: try using --normalize-spans, length difference > 1.0s")
    info = {c: (eff_fps[c], len(pts[c]), size[c]) for c in args.cameras}
    n_total = N_total - args.start
    if args.max_frames:
        n_total = min(n_total, args.max_frames)
    readers = {c: Reader(video_path(args.root, c), src_idx[c][args.start:args.start + n_total], pts[c], keep_full=bool(rois.get(c)))
               for c in args.cameras}
    tcfg = IterableSimpleNamespace(**load_yaml(args.tracker))
    import inspect
    _params = [p for p in inspect.signature(BOTSORT.__init__).parameters if p not in ("self", "args")]
    _kw = {_params[0]: int(round(fps))} if _params else {}
    trackers = {c: BOTSORT(tcfg, **_kw) for c in args.cameras}

    rows = []           # (f, cam, tid, x1,y1,x2,y2, conf, px, py, foot_ok)
    cache = {}
    cams = list(args.cameras)
    processed = 0
    t0 = time.time()
    for f in range(n_total):
        items = {}
        for c in cams:
            it = readers[c].q.get()
            if it is not None:
                items[c] = it
        if len(items) < len(cams):
            break
        new = [c for c in cams if items[c][3] or c not in cache]
        if new:
            results = model.predict([items[c][0] for c in new], imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                                    classes=[0], device=device, verbose=False, **kw)
            for c, r in zip(new, results):
                parts = [parse_result(r)]
                if items[c][1] is not None:
                    for roi in rois.get(c, []):
                        parts.append(detect_roi(model, items[c][1], roi, args, device, kw))
                b, cf, kps = merge_dets(parts)
                b, cf, kps = split_hugs(b, cf, kps, args.hug_ar)
                keep, plan_xy, foot_ok = filter_detections(c, cal, b, kps, args)
                cache[c] = (b[keep], cf[keep], plan_xy, foot_ok)
        for c in cams:
            b, cf, plan_xy, foot_ok = cache[c]
            det = Dets(b, cf, np.zeros(len(b)))
            tr = trackers[c].update(det, items[c][0])
            if tr is None or len(tr) == 0 or len(b) == 0:
                continue
            tr = np.asarray(tr)
            iou = iou_matrix(tr[:, :4].astype(np.float32), b)
            ti, di = linear_sum_assignment(1.0 - iou)
            for t_i, idx in zip(ti, di):
                if iou[t_i, idx] < 0.3:
                    continue
                tid, idx = int(tr[t_i][4]), int(idx)
                x1, y1, x2, y2 = b[idx]
                rows.append([f + args.start, c, tid, x1 / CANON[0], y1 / CANON[1], x2 / CANON[0], y2 / CANON[1],
                             float(cf[idx]), float(plan_xy[idx][0]), float(plan_xy[idx][1]), bool(foot_ok[idx])])
        processed = f + 1
        if f % 250 == 0:
            el = time.time() - t0
            print(f"[{f}/{n_total}] {el / (f + 1):.3f} sec per frame, remaining ~{(n_total - f) * el / (f + 1) / 60:.1f} мин", flush=True)
    n_total = processed

    arr = np.array(rows, dtype=object) if rows else np.zeros((0, 11), object)
    out = {
        "fps": fps, "n_frames": n_total, "start": args.start, "canon": CANON, "cameras": args.cameras,
        "video_size": {c: info[c][2] for c in args.cameras},
        "src_fps": eff_fps, "src_index": src_idx, "src_pts_ms": pts,
        "f": arr[:, 0].astype(np.int32), "cam": arr[:, 1].astype(np.int8), "tid": arr[:, 2].astype(np.int32),
        "box": arr[:, 3:7].astype(np.float32), "conf": arr[:, 7].astype(np.float32),
        "plan": arr[:, 8:10].astype(np.float32), "foot_ok": arr[:, 10].astype(bool),
    }
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("saved ", args.out)


if __name__ == "__main__":
    main()
