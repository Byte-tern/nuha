import pickle
import bisect
import numpy as np
import cv2
from common import fmt_time, PLAN_SIZE

PRESENCE_WINDOW = 3


def id_color(i):
    h = (i * 47) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


class Result:
    def __init__(self, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self.d = d
        self.fps = float(d["fps"])
        self.start = int(d["start"])
        self.n_frames = int(d["n_frames"])
        self.last_frame = self.start + self.n_frames - 1
        self.cameras = list(d["cameras"])
        self.src_index = d.get("src_index")
        self.src_pts = d.get("src_pts_ms")
        self.tracks = d["tracks"]
        self.ids = sorted(self.tracks)
        f = d["f"]; order = np.argsort(f, kind="stable")
        self.f = f[order]; self.cam = d["cam"][order]; self.gid = d["gid"][order]
        self.box = d["box"][order]; self.plan = d["plan"][order]; self.fok = d["foot_ok"][order]
        self.traj = {}
        for i in self.ids:
            m = self.gid == i
            ff, pp, w = self.f[m], self.plan[m], np.where(self.fok[m], 1.0, 0.4)
            ok = ~np.isnan(pp[:, 0])
            ff, pp, w = ff[ok], pp[ok], w[ok]
            if len(ff) == 0:
                self.traj[i] = (np.zeros(0, int), np.zeros((0, 2))); continue
            uf, inv = np.unique(ff, return_inverse=True)
            xy = np.zeros((len(uf), 2)); ws = np.zeros(len(uf))
            np.add.at(xy, inv, pp * w[:, None]); np.add.at(ws, inv, w)
            xy /= ws[:, None]
            k = 7
            if len(xy) >= k:
                pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
                xy = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(2)], 1)
            self.traj[i] = (uf, xy)
        self.first_seen = {}
        for g, c, f_ in zip(self.gid, self.cam, self.f):
            k = (int(g), int(c))
            if k not in self.first_seen:
                self.first_seen[k] = int(f_)
        self._a = np.searchsorted(self.f, np.arange(self.start, self.last_frame + 2), "left")

    def pts_of(self, cam):
        return None if not self.src_pts or cam not in self.src_pts else self.src_pts[cam]

    def src_frame(self, cam, frame):
        if not self.src_index or cam not in self.src_index:
            return frame
        arr = self.src_index[cam]
        return int(arr[min(max(frame, 0), len(arr) - 1)])

    def _rows(self, frame, back=0):
        lo = frame - back - self.start; hi = frame - self.start
        lo = max(lo, 0)
        return self._a[lo], self._a[min(hi + 1, len(self._a) - 1)]

    def boxes(self, frame, cam):
        a, b = self._rows(frame)
        m = self.cam[a:b] == cam
        return self.gid[a:b][m], self.box[a:b][m]

    def shown_finished(self, i, frame):
        t = self.tracks[i]
        return t["finished"] and frame > t["last"]

    def started_ids(self, frame):
        return [i for i in self.ids if self.tracks[i]["start"] <= frame]

    def active_ids(self, frame):
        return [i for i in self.started_ids(frame) if not self.shown_finished(i, frame)]

    def table(self, frame):
        a, b = self._rows(frame, PRESENCE_WINDOW - 1)
        seen = {}
        for g, c in zip(self.gid[a:b], self.cam[a:b]):
            seen.setdefault(int(g), set()).add(int(c))
        rows = []
        for i in self.started_ids(frame):
            t = self.tracks[i]
            fin = fmt_time(t["last"], self.fps) if self.shown_finished(i, frame) else "ongoing"
            rows.append((i, f"{i:05d}", fmt_time(t["start"], self.fps), fin,
                         [self._cam_status(i, c, frame, seen) for c in (1, 2, 3, 4, 5)]))
        return rows

    def _cam_status(self, i, c, frame, seen):
        if c in seen.get(i, ()):
            return "V"
        return "X" if self.first_seen.get((i, c), 10 ** 12) <= frame else " "

    def route(self, i, frame):
        uf, xy = self.traj[i]
        k = bisect.bisect_right(uf.tolist(), frame)
        return xy[:k]

    def position(self, i, frame):
        r = self.route(i, frame)
        return None if len(r) == 0 else r[-1]


def draw_video(img, ids, boxes, size):
    w, h = size
    v = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    for gid, b in zip(ids, boxes):
        x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
        col = id_color(int(gid))
        cv2.rectangle(v, (x1, y1), (x2, y2), col, 2)
        txt = str(int(gid))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(v, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(v, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return v


def draw_plan(plan_bgr, res, frame, selected, size):
    w, h = size
    sx, sy = w / plan_bgr.shape[1], h / plan_bgr.shape[0]
    v = cv2.resize(plan_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    pt = lambda p: (int(p[0] * sx), int(p[1] * sy))
    if selected is not None and selected in res.tracks:
        r = res.route(selected, frame)
        col = id_color(selected)
        if len(r) > 1:
            cv2.polylines(v, [np.array([pt(p) for p in r], np.int32)], False, col, 2, cv2.LINE_AA)
        if len(r) >= 1:
            cv2.circle(v, pt(r[0]), 5, (0, 160, 0), -1)               # старт
            cv2.circle(v, pt(r[-1]), 6, col, -1); cv2.circle(v, pt(r[-1]), 7, (0, 0, 0), 1)
            cv2.putText(v, str(selected), (pt(r[-1])[0] + 8, pt(r[-1])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else:
        for i in res.active_ids(frame):
            p = res.position(i, frame)
            if p is None:
                continue
            col = id_color(i)
            cv2.circle(v, pt(p), 6, col, -1); cv2.circle(v, pt(p), 7, (0, 0, 0), 1)
            cv2.putText(v, str(i), (pt(p)[0] + 8, pt(p)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return v
