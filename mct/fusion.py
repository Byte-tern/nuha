import argparse
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *

INF = 1e6


class P:
    max_gap = 30
    min_track_len = 8
    r0 = 20.0
    k_gate = 2.5 
    vmax = 1.2
    incons_dist = 45.0
    incons_frames = 20
    merge_dist = 10.0 
    k_merge = 2.0 
    merge_frames = 10
    off_min_overlap = 10
    off_close_ratio = 0.7
    off_veto_frames = 3
    unreliable_w = 0.4
    foot_sigma_min = 3.0
    foot_sigma_frac = 0.04
    sig_cap = 40.0
    sig_w0 = 8.0
    min_duration_s = 1.0


class Local:
    __slots__ = ("cam", "tid", "gid", "n", "pending", "pos", "w", "sig", "bad", "live", "first_f", "last_f")

    def __init__(self, cam, tid, f):
        self.cam, self.tid, self.gid = cam, tid, None
        self.n, self.pending, self.pos, self.w, self.sig = 0, [], None, 1.0, 0.0
        self.bad, self.live, self.first_f, self.last_f = 0, False, f, f


class Glob:
    def __init__(self, gid, f):
        self.id, self.start_f, self.last_f = gid, f, f
        self.pos, self.sig, self.finished, self.locals = None, 0.0, False, set()
        self.live_pos, self.live_sig = None, 0.0


def plan_sigma(d, cal, P):
    n = len(d["f"])
    if "sig" in d:
        return np.asarray(d["sig"], np.float32)
    sig = np.zeros(n, np.float32)
    if cal is None:
        return sig
    for c in set(int(x) for x in d["cam"]):
        if not cal.is_calibrated(c):
            continue
        m = d["cam"] == c
        bx = d["box"][m]
        foot = np.stack([(bx[:, 0] + bx[:, 2]) / 2 * CANON[0], bx[:, 3] * CANON[1]], 1)
        hpx = (bx[:, 3] - bx[:, 1]) * CANON[1]
        s_px = P.foot_sigma_min + P.foot_sigma_frac * hpx
        p0 = cal.to_plan(c, foot)
        jx = cal.to_plan(c, foot + np.array([1.0, 0.0])) - p0
        jy = cal.to_plan(c, foot + np.array([0.0, 1.0])) - p0
        j = np.sqrt((jx ** 2).sum(1) + (jy ** 2).sum(1)) / np.sqrt(2.0)
        sig[m] = np.minimum(j * s_px, P.sig_cap)
    return sig


def gate(sa, sb, P):
    return max(P.r0, P.k_gate * float(np.hypot(sa, sb)))


def offline_merge(fin, fr, cam, plan, sig, flags, P):
    def stats(r):
        m = fin == r
        ff, pp, ss, cc = fr[m], plan[m], sig[m], cam[m]
        ok = ~np.isnan(pp[:, 0])
        ff, pp, ss, cc = ff[ok], pp[ok], ss[ok], cc[ok]
        uf, inv = np.unique(ff, return_inverse=True)
        xy = np.zeros((len(uf), 2)); ws = np.zeros(len(uf)); sg = np.full(len(uf), np.inf); cm = np.zeros(len(uf), np.int32)
        w = 1.0 / (1.0 + (ss / P.sig_w0) ** 2)
        np.add.at(xy, inv, pp * w[:, None]); np.add.at(ws, inv, w)
        np.minimum.at(sg, inv, ss); np.bitwise_or.at(cm, inv, (1 << cc.astype(np.int32)))
        return uf, xy / np.maximum(ws, 1e-9)[:, None], sg, cm

    roots = sorted({int(x) for x in fin if x >= 0})
    st = {r: stats(r) for r in roots}
    merged = True
    while merged:
        merged = False
        best = None
        for i, a in enumerate(roots):
            for b in roots[i + 1:]:
                fa, pa, sa, ca = st[a]; fb, pb, sb, cb = st[b]
                if len(fa) == 0 or len(fb) == 0:
                    continue
                common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
                if len(common) < P.off_min_overlap:
                    continue
                if int(np.sum((ca[ia] & cb[ib]) != 0)) > P.off_veto_frames:
                    continue
                dist = np.linalg.norm(pa[ia] - pb[ib], axis=1)
                thr = np.maximum(P.merge_dist, P.k_merge * np.hypot(sa[ia], sb[ib]))
                ratio = float(np.mean(dist < thr))
                if ratio >= P.off_close_ratio and (best is None or ratio > best[0]):
                    best = (ratio, a, b)
        if best:
            _, a, b = best
            first = lambda r: (int(fr[fin == r].min()), r)
            old, new = (a, b) if first(a) <= first(b) else (b, a)
            last = lambda r: int(fr[fin == r].max())
            flags[old] = flags[new] if last(new) > last(old) else flags[old]
            fin[fin == new] = old
            roots.remove(new); del st[new]; st[old] = stats(old)
            merged = True
    return fin


def fuse(d, cal=None, P=P, verbose=True):
    fr, cam, tid = d["f"], d["cam"], d["tid"]
    plan, fok = d["plan"], d["foot_ok"]
    sig = plan_sigma(d, cal, P)
    order = np.argsort(fr, kind="stable")
    fr_sorted = fr[order]
    n = len(fr)
    row_gid = np.full(n, -1, np.int32)
    cam_w = {c: (cal.weight(c) if cal else 1.0) for c in d["cameras"]}

    locals_, globs, alias = {}, {}, {}
    active = []
    next_gid = [1]
    merge_cnt = {}

    def new_glob(f):
        g = Glob(next_gid[0], f); next_gid[0] += 1; globs[g.id] = g; active.append(g); return g

    def bind(L, g):
        L.gid = g.id; g.locals.add((L.cam, L.tid))
        for r in L.pending:
            row_gid[r] = g.id
        L.pending = []
        g.start_f = min(g.start_f, L.first_f)

    frames_present = np.unique(fr_sorted)
    t_first = int(fr_sorted[0]) if n else 0
    t_last = d["start"] + d["n_frames"] - 1
    pos_idx = {int(f): (np.searchsorted(fr_sorted, f, "left"), np.searchsorted(fr_sorted, f, "right")) for f in frames_present}
    valid = lambda L: L.pos is not None and not np.any(np.isnan(L.pos))

    def gpos(g):
        ls = [locals_[k] for k in g.locals if locals_[k].live and valid(locals_[k])]
        if not ls:
            return None, 0.0
        w = np.array([l.w for l in ls]); p = np.array([l.pos for l in ls])
        return (p * w[:, None]).sum(0) / w.sum(), min(l.sig for l in ls)

    for t in range(t_first, t_last + 1):
        a, b = pos_idx.get(t, (0, 0))
        live = {}
        for r in order[a:b]:
            key = (int(cam[r]), int(tid[r]))
            L = locals_.get(key)
            if L is None:
                L = Local(key[0], key[1], t); locals_[key] = L
            if L.gid is not None and globs[L.gid].finished:
                L.gid, L.pending, L.n, L.first_f = None, [], 0, t
            L.n += 1; L.last_f = t; L.pos = plan[r]; L.sig = float(sig[r])
            L.w = cam_w[key[0]] * (1.0 if fok[r] else P.unreliable_w) / (1.0 + (L.sig / P.sig_w0) ** 2)
            if L.gid is None:
                L.pending.append(r)
            else:
                row_gid[r] = L.gid
            live[key] = L
        for L in locals_.values():
            L.live = False
        for L in live.values():
            L.live = True
        for g in active:
            g.live_pos, g.live_sig = gpos(g)

        for g in active:
            ls = [locals_[k] for k in g.locals if locals_[k].live and valid(locals_[k])]
            if len(ls) >= 2:
                for L in ls:
                    others = [o for o in ls if o is not L]
                    w = np.array([o.w for o in others]); po = np.array([o.pos for o in others])
                    c = (po * w[:, None]).sum(0) / w.sum()
                    thr = max(P.incons_dist, 3.0 * np.hypot(L.sig, min(o.sig for o in others)))
                    L.bad = L.bad + 1 if np.linalg.norm(L.pos - c) > thr else 0
                    if L.bad >= P.incons_frames:
                        g.locals.discard((L.cam, L.tid)); L.gid, L.bad, L.pending = None, 0, []
                        L.n = P.min_track_len
        for g in active:
            g.live_pos, g.live_sig = gpos(g)

        U = [L for L in live.values() if L.gid is None and L.n >= P.min_track_len and valid(L)]
        used_cams = {g.id: {locals_[k].cam for k in g.locals if locals_[k].live} for g in active}
        matched = set()
        if U and active:
            C = np.full((len(U), len(active)), INF)
            for i, L in enumerate(U):
                for j, g in enumerate(active):
                    if L.cam in used_cams[g.id]:
                        continue
                    live_g = g.live_pos is not None
                    gp = g.live_pos if live_g else g.pos
                    if gp is None:
                        continue
                    gs = g.live_sig if live_g else g.sig
                    gap = 0 if live_g else (t - g.last_f)
                    R = gate(L.sig, gs, P) + P.vmax * gap
                    dp = float(np.linalg.norm(L.pos - gp))
                    if dp <= R:
                        C[i, j] = dp / R
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < INF / 2:
                    bind(U[i], active[j]); matched.add(i); used_cams[active[j].id].add(U[i].cam)
        left = [L for i, L in enumerate(U) if i not in matched]

        if left:
            pairs = []
            for i in range(len(left)):
                for j in range(i + 1, len(left)):
                    A_, B_ = left[i], left[j]
                    if A_.cam == B_.cam:
                        continue
                    dp = float(np.linalg.norm(A_.pos - B_.pos)); R = gate(A_.sig, B_.sig, P)
                    if dp <= R:
                        pairs.append((dp / R, i, j))
            parent = list(range(len(left)))
            cams_of = {i: {left[i].cam} for i in range(len(left))}
            find = lambda x: x if parent[x] == x else find(parent[x])
            for c, i, j in sorted(pairs):
                ri, rj = find(i), find(j)
                if ri != rj and not (cams_of[ri] & cams_of[rj]):
                    parent[rj] = ri; cams_of[ri] |= cams_of[rj]
            groups = {}
            for i in range(len(left)):
                groups.setdefault(find(i), []).append(left[i])
            for members in groups.values():
                g = new_glob(min(m.first_f for m in members))
                for m in members:
                    bind(m, g)
        for g in active:
            g.live_pos, g.live_sig = gpos(g)

        sat = set()
        act = [g for g in active if g.live_pos is not None]
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                g1, g2 = act[i], act[j]
                c1 = {locals_[k].cam for k in g1.locals if locals_[k].live}
                c2 = {locals_[k].cam for k in g2.locals if locals_[k].live}
                if c1 & c2:
                    continue
                thr = max(P.merge_dist, P.k_merge * np.hypot(g1.live_sig, g2.live_sig))
                if np.linalg.norm(g1.live_pos - g2.live_pos) <= thr:
                    sat.add((g1.id, g2.id))
        keys = set(merge_cnt) | sat
        merge_cnt = {k: (merge_cnt.get(k, 0) + 1 if k in sat else max(0, merge_cnt.get(k, 0) - 1)) for k in keys}
        merge_cnt = {k: v for k, v in merge_cnt.items() if v > 0}
        for (i1, i2), cnt in list(merge_cnt.items()):
            if cnt >= P.merge_frames and not globs[i1].finished and not globs[i2].finished:
                old, new = (globs[i1], globs[i2]) if (globs[i1].start_f, i1) <= (globs[i2].start_f, i2) else (globs[i2], globs[i1])
                for k in new.locals:
                    locals_[k].gid = old.id; old.locals.add(k)
                old.start_f = min(old.start_f, new.start_f)
                alias[new.id] = old.id
                new.finished = True; new.locals = set()
                active.remove(new)
                merge_cnt = {k: v for k, v in merge_cnt.items() if new.id not in k}
                break

        for g in active:
            g.live_pos, g.live_sig = gpos(g)
            if g.live_pos is not None:
                g.pos, g.sig, g.last_f = g.live_pos, g.live_sig, t
        for g in list(active):
            if g.live_pos is None and t - g.last_f > P.max_gap:
                g.finished = True
                active.remove(g)

    def root(x):
        while x in alias:
            x = alias[x]
        return x
    fin = np.array([root(int(g)) if g >= 0 else -1 for g in row_gid], np.int32)
    flags = {r: globs[r].finished for r in {int(x) for x in fin if x >= 0}}
    n_online = len(flags)
    fin = offline_merge(fin, fr, cam, plan, sig, flags, P)
    ok = fin >= 0
    min_frames = int(round(P.min_duration_s * d["fps"]))
    roots_all = sorted({int(x) for x in fin[ok]}, key=lambda r: (int(fr[ok][fin[ok] == r].min()), r))
    roots, dropped = [], 0
    for r in roots_all:
        m = fin == r
        if int(fr[m].max()) - int(fr[m].min()) + 1 >= min_frames:
            roots.append(r)
        else:
            dropped += 1
    newid = {r: i + 1 for i, r in enumerate(roots)}
    gid_final = np.array([newid.get(int(x), -1) if x >= 0 else -1 for x in fin], np.int32)
    tracks = {}
    for r in roots:
        m = fin == r
        tracks[newid[r]] = {"start": int(fr[m].min()), "last": int(fr[m].max()), "finished": bool(flags[r])}
    if verbose:
        nf = sum(t["finished"] for t in tracks.values())
    sel = gid_final >= 0
    return {"fps": d["fps"], "n_frames": d["n_frames"], "start": d["start"], "cameras": d["cameras"],
            "video_size": d["video_size"], "tracks": tracks,
            "src_fps": d.get("src_fps"), "src_index": d.get("src_index"), "src_pts_ms": d.get("src_pts_ms"),
            "f": fr[sel], "cam": cam[sel], "gid": gid_final[sel], "box": d["box"][sel], "plan": plan[sel],
            "foot_ok": fok[sel], "ltid": tid[sel]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="local_tracks.pkl")
    ap.add_argument("--out", default="fusion_result.pkl")
    ap.add_argument("--calib", default=CALIB_FILE)
    ap.add_argument("--r0", type=float, default=P.r0)
    ap.add_argument("--k-gate", type=float, default=P.k_gate)
    ap.add_argument("--k-merge", type=float, default=P.k_merge)
    ap.add_argument("--max-gap", type=int, default=P.max_gap)
    ap.add_argument("--min-track-len", type=int, default=P.min_track_len)
    ap.add_argument("--min-duration", type=float, default=P.min_duration_s)
    a = ap.parse_args()
    P.r0, P.k_gate, P.k_merge, P.max_gap = a.r0, a.k_gate, a.k_merge, a.max_gap
    P.min_track_len, P.min_duration_s = a.min_track_len, a.min_duration
    with open(a.inp, "rb") as fh:
        data = pickle.load(fh)
    res = fuse(data, Calibration.load(a.calib), P)
    with open(a.out, "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    print("saved ", a.out)
