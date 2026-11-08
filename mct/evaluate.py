"""
Оценка качества. Результат (fusion_result.pkl) оценивается в двух режимах.

1) БЕЗ разметки (всегда): число треков, доля коротких, согласованность проекций одного ID с разных камер на плане
   (одновременно индикатор калибровки и синхронизации), подозрение на дубли, наблюдения вне проходимой области.
2) С эталоном (--gt gt.csv): детекция (P/R/F1), IDF1/IDP/IDR (сквозные между камерами и по каждой камере), MOTA, ID switch,
   фрагментация/смешение ID, ложные треки, ошибки времени старта и finished, ошибка положения на плане (--gt-plan).

  python evaluate.py --result fusion_result.pkl --calib calibration.json
  python evaluate.py --result fusion_result.pkl --gt gt.csv [--gt-plan gt_plan.csv] [--clip-end 4500]
  python evaluate.py --result fusion_result.pkl --dump-frames clip1 --from 1500 --to 3000 --every 25 --root .   # кадры для разметки

Формат gt.csv (заголовок обязателен):  frame,cam,id,x1,y1,x2,y2[,tag]
  frame - номер кадра ШКАЛЫ времени (как в таблице интерфейса/--dump-frames), cam - 1..5,
  id - ЕДИНЫЙ идентификатор человека для всех камер; по правилу ТЗ человек, пропавший со ВСЕХ камер > 30 кадров, при возвращении получает НОВЫЙ id,
  x1..y2 - рамка в пикселях кадра 1600x1200 либо долях 0..1, tag - необязательно: hug (идут в обнимку), mirror, far, occluded и т.п.
  Размечаются только выбранные кадры (например, каждый 25-й); на каждом из них размечаются ВСЕ 5 камер (если людей нет - просто без строк:
  так считаются ложные срабатывания). Оцениваются ровно эти кадры.
Формат gt_plan.csv: frame,id,x,y  (точка стоп человека на плане, пикселях officePlan.jpg; несколько десятков точек).
"""
import argparse
import csv
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from result import Result

IOU_THR = 0.5
MIN_MATCH = 5 

def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gt(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            b = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            if max(b) > 1.5:
                b = [b[0] / CANON[0], b[1] / CANON[1], b[2] / CANON[0], b[3] / CANON[1]]
            rows.append((int(r["frame"]), int(r["cam"]), int(r["id"]), b, (r.get("tag") or "").strip()))
    return rows


def label_free(R, cal=None):
    fps = R.fps
    print("=== Metrics (label-free) ===")
    dur = {i: (t["last"] - t["start"] + 1) / fps for i, t in R.tracks.items()}
    fin = sum(t["finished"] for t in R.tracks.values())
    v = np.array(list(dur.values())) if dur else np.zeros(1)
    print(f"tracks: {len(R.tracks)} (finished {fin}, ongoing {len(R.tracks) - fin}); length, sec: median {np.median(v):.1f}, "
          f"p10 {np.percentile(v, 10):.1f}, p90 {np.percentile(v, 90):.1f}; less than 2 sec: {np.mean(v < 2):.0%}; "
          f"tracks per minute: {len(R.tracks) / max(R.n_frames / fps / 60, 1e-9):.1f}")
    for c in R.cameras:
        m = R.cam == c
        print(f"  camera {c}: seen {int(m.sum())}, ID amount {len(set(R.gid[m].tolist()))}")
    order = np.lexsort((R.cam, R.f, R.gid))
    g, f, c, p = R.gid[order], R.f[order], R.cam[order], R.plan[order]
    dists, pair_d = [], {}
    i = 0
    while i < len(g):
        j = i
        while j < len(g) and g[j] == g[i] and f[j] == f[i]:
            j += 1
        if j - i >= 2:
            for a in range(i, j):
                for b in range(a + 1, j):
                    if c[a] != c[b] and not np.isnan(p[a, 0]) and not np.isnan(p[b, 0]):
                        d = float(np.linalg.norm(p[a] - p[b])); dists.append(d)
                        pair_d.setdefault((int(c[a]), int(c[b])), []).append(d)
        i = j
    if dists:
        d = np.array(dists)
        print(f"ID location difference: median {np.median(d):.1f}, p90 {np.percentile(d, 90):.1f}  ")
        for k, v_ in sorted(pair_d.items()):
            print(f"   cameras {k[0]}-{k[1]}: медиана {np.median(v_):.1f}, p90 {np.percentile(v_, 90):.1f}, n={len(v_)}")
    else:
        print("no multi-camera detection found")
    ids = R.ids; sus = []
    for a in range(len(ids)):
        fa, pa = R.traj[ids[a]]
        for b in range(a + 1, len(ids)):
            fb, pb = R.traj[ids[b]]
            common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
            if len(common) >= fps:
                d = np.linalg.norm(pa[ia] - pb[ib], axis=1)
                if np.mean(d < 12) > 0.8:
                    sus.append((ids[a], ids[b], len(common) / fps, float(np.median(d))))
    print(f"suspicious crops (2+ ID >=1 in a single place): {len(sus)}", *[f"\n   ID {a} и {b}: {t:.1f} sec, median {m:.1f} px" for a, b, t, m in sus[:10]])
    if cal is not None and cal.data.get("walkable"):
        ok = ~np.isnan(R.plan[:, 0])
        out = sum(0 if cal.is_walkable(q, 8) else 1 for q in R.plan[ok]) / max(ok.sum(), 1)
        print(f"detections beyond allowed polygon: {out:.1%}")


def with_gt(R, gt_rows, gt_plan=None, clip_end=None):
    print("\n=== Metrics (etalon) ===")
    frames = sorted({(f, c) for f in {r[0] for r in gt_rows} for c in R.cameras})    
    by = {}
    for f, c, i, b, tag in gt_rows:
        by.setdefault((f, c), []).append((i, np.array(b), tag))
    clip_start, clip_end = min(f for f, *_ in frames), clip_end or max(f for f, *_ in frames)
    gt_frames_n = len({f for f, _ in frames})
    tp = fp = fn = idsw = 0
    pot = {}  
    got = {}
    cam_data = {c: {"tp": 0, "fp": 0, "fn": 0, "pot": {}, "ng": 0, "np": 0} for c in R.cameras}
    seq = {}                       # g -> [(frame, pred_id)]
    tag_hit, tag_all, fp_cam = {}, {}, {c: 0 for c in R.cameras}
    n_gt = n_pr = 0
    for (f, c) in frames:
        g = by.get((f, c), [])
        pid, pbox = R.boxes(f, c)
        gb = np.array([x[1] for x in g]).reshape(-1, 4); iou = iou_mat(gb, pbox)
        n_gt += len(g); n_pr += len(pid)
        cam_data[c]["ng"] += len(g); cam_data[c]["np"] += len(pid)
        for a, (gi, _, _) in enumerate(g):
            for b in range(len(pid)):
                if iou[a, b] >= IOU_THR:
                    pot[(gi, int(pid[b]))] = pot.get((gi, int(pid[b])), 0) + 1
                    cam_data[c]["pot"][(gi, int(pid[b]))] = cam_data[c]["pot"].get((gi, int(pid[b])), 0) + 1
        mt = set()
        if iou.size:
            ri, ci = linear_sum_assignment(1 - iou)
            for a, b in zip(ri, ci):
                if iou[a, b] >= IOU_THR:
                    mt.add(a); tp += 1; cam_data[c]["tp"] += 1
                    got[(g[a][0], int(pid[b]))] = got.get((g[a][0], int(pid[b])), 0) + 1
                    seq.setdefault(g[a][0], []).append((f, int(pid[b])))
                    if g[a][2]:
                        tag_hit[g[a][2]] = tag_hit.get(g[a][2], 0) + 1
            mp = {b for a, b in zip(ri, ci) if iou[a, b] >= IOU_THR}
        else:
            mp = set()
        fn += len(g) - len(mt); cam_data[c]["fn"] += len(g) - len(mt)
        fp += len(pid) - len(mp); cam_data[c]["fp"] += len(pid) - len(mp); fp_cam[c] += len(pid) - len(mp)
        for _, _, tg in g:
            if tg:
                tag_all[tg] = tag_all.get(tg, 0) + 1
    P_, R_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"detection (IoU>={IOU_THR}): precision {P_:.3f}, recall {R_:.3f}, F1 {2 * P_ * R_ / max(P_ + R_, 1e-9):.3f}  (TP {tp}, FP {fp}, FN {fn})")
    print("  FP:", fp_cam)
    for t_, n_ in sorted(tag_all.items()):
        print(f"  recall for tag '{t_}': {tag_hit.get(t_, 0) / n_:.3f} (n={n_})")

    def idf1(potc, ng, npd):
        gi = sorted({k[0] for k in potc}); pi = sorted({k[1] for k in potc})
        if not gi or not pi:
            return 0.0, 0.0, 0.0, 0
        M = np.zeros((len(gi), len(pi)))
        for (a, b), v in potc.items():
            M[gi.index(a), pi.index(b)] = v
        r_, c_ = linear_sum_assignment(-M)
        idtp = int(M[r_, c_].sum())
        return idtp / max(idtp + 0.5 * (ng - idtp) + 0.5 * (npd - idtp), 1e-9), idtp / max(npd, 1), idtp / max(ng, 1), idtp
    f1, idp, idr, idtp = idf1(pot, n_gt, n_pr)
    print(f"IDF1 (all cams): {f1:.3f}  IDP {idp:.3f}  IDR {idr:.3f}")
    for c in R.cameras:
        d = cam_data[c]
        if d["ng"]:
            print(f"  IDF1 camera {c}: {idf1(d['pot'], d['ng'], d['np'])[0]:.3f}  (R {d['tp'] / max(d['tp'] + d['fn'], 1):.2f}, P {d['tp'] / max(d['tp'] + d['fp'], 1):.2f})")
    for g_, s_ in seq.items():
        s_.sort(); last = None
        for _, p_ in s_:
            if last is not None and p_ != last:
                idsw += 1
            last = p_
    print(f"ID switches: {idsw};  MOTA: {1 - (fn + fp + idsw) / max(n_gt, 1):.3f}")
    gts = sorted({k[0] for k in got}); prs = sorted({k[1] for k in got})
    frag = {g_: [p_ for (a, p_), v in got.items() if a == g_ and v >= MIN_MATCH] for g_ in {k[0] for k in pot}}
    mix = {p_: [g_ for (g_, b), v in got.items() if b == p_ and v >= MIN_MATCH] for p_ in prs}
    print(f"fragmented: {sum(max(len(v) - 1, 0) for v in frag.values())} false ID; "
          f"no id ID: {sum(1 for v in frag.values() if not v)} из {len(frag)}")
    print(f"merged: {sum(max(len(v) - 1, 0) for v in mix.values())}")
    gt_ids = {r[2] for r in gt_rows}
    matched_pred = {p_ for (g_, p_), v in got.items() if v >= 3}
    false_tr = [i for i in R.ids if i not in matched_pred and R.tracks[i]["start"] <= clip_end and R.tracks[i]["last"] >= clip_start]
    print(f"false tracks: {len(false_tr)}  {false_tr[:15]}")
    print(f"people amount: etalon {len(gt_ids)}, tracks found {len([i for i in R.ids if R.tracks[i]['start'] <= clip_end and R.tracks[i]['last'] >= clip_start])}")
    st_err, fin_err, ok_flag, n_flag = [], [], 0, 0
    best = {}
    for g_ in gt_ids:
        cand = [(v, p_) for (a, p_), v in got.items() if a == g_]
        if cand:
            best[g_] = max(cand)[1]
    for g_, p_ in best.items():
        gf = [r[0] for r in gt_rows if r[2] == g_]
        g0, g1 = min(gf), max(gf)
        t = R.tracks[p_]
        if g0 > clip_start + 1:
            st_err.append((t["start"] - g0) / R.fps)
        gt_fin = g1 < clip_end - 30
        pr_fin = t["finished"] and t["last"] <= clip_end - 30
        n_flag += 1; ok_flag += int(gt_fin == pr_fin)
        if gt_fin and pr_fin:
            fin_err.append((t["last"] - g1) / R.fps)
    if st_err:
        print(f"start error, sec: median {np.median(np.abs(st_err)):.2f}, p90 {np.percentile(np.abs(st_err), 90):.2f} (n={len(st_err)})")
    if fin_err:
        print(f"finished error, sec: median {np.median(np.abs(fin_err)):.2f}, p90 {np.percentile(np.abs(fin_err), 90):.2f} (n={len(fin_err)})")
    if n_flag:
        print(f"flag finished/ongoing is correct: {ok_flag}/{n_flag}")
    if gt_plan:
        errs = []
        with open(gt_plan, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                g_, f = int(r["id"]), int(r["frame"])
                if g_ in best:
                    fr_, xy = R.traj[best[g_]]
                    if len(fr_):
                        k = int(np.argmin(np.abs(fr_ - f)))
                        if abs(fr_[k] - f) <= 3:
                            errs.append(float(np.hypot(xy[k][0] - float(r["x"]), xy[k][1] - float(r["y"]))))
        if errs:
            e = np.array(errs)
            print(f"location error, px: median {np.median(e):.1f}, ground {e.mean():.1f}, p90 {np.percentile(e, 90):.1f} (n={len(e)})")


def dump_frames(R, root, out, f0, f1, every):
    import cv2
    os.makedirs(out, exist_ok=True)
    for c in R.cameras:
        cap = cv2.VideoCapture(video_path(root, c)); pts = R.pts_of(c)
        for f in range(f0, f1 + 1, every):
            i = R.src_frame(c, f)
            fr, _ = seek_read(cap, pts, i) if pts is not None else (None, -1)
            if fr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
                if not ok: continue
            cv2.imwrite(os.path.join(out, f"cam{c}_f{f:06d}.jpg"), cv2.resize(fr, CANON))
        cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="fusion_result.pkl")
    ap.add_argument("--calib", default=CALIB_FILE)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-plan", default=None)
    ap.add_argument("--clip-end", type=int, default=None)
    ap.add_argument("--dump-frames", default=None)
    ap.add_argument("--from", dest="f0", type=int, default=0)
    ap.add_argument("--to", dest="f1", type=int, default=None)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = Result(a.result)
    if a.dump_frames:
        dump_frames(R, a.root, a.dump_frames, a.f0, a.f1 if a.f1 is not None else R.last_frame, a.every)
    else:
        cal = Calibration.load(a.calib) if os.path.exists(a.calib) else None
        label_free(R, cal)
        if a.gt:
            with_gt(R, load_gt(a.gt), a.gt_plan, a.clip_end)
