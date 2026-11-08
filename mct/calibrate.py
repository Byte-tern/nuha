import argparse
import os
import numpy as np
import cv2
from scipy.optimize import minimize_scalar
from common import *


def grab_frame(root, cam, frame_no, image=None):
    if image:
        img = cv2.imread(image)
    else:
        cap = cv2.VideoCapture(video_path(root, cam))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"unable to read {frame_no} from {video_path(root, cam)}")
    return img


def fit_scale(img, maxw=1500, maxh=880):
    h, w = img.shape[:2]
    return min(maxw / w, maxh / h, 1.0)



def line_residual(points_norm_lines, k, size):
    res = []
    for pts in points_norm_lines:
        if len(pts) < 3:
            continue
        p = np.array(pts) * np.array(CANON)
        u = undistort_px(p, k)
        c = u.mean(0)
        _, _, vt = np.linalg.svd(u - c)
        n = vt[1]
        res.extend(((u - c) @ n).tolist())
    return float(np.sqrt(np.mean(np.square(res)))) if res else 0.0


def mode_k(a):
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit("no image " + a.image)
    cal = Calibration.load(a.calib)
    h, w = img.shape[:2]
    s = fit_scale(img)
    disp = cv2.resize(img, None, fx=s, fy=s)
    lines, cur = [], []
    state = {"k": cal.k1}

    def redraw():
        v = disp.copy()
        for ln in lines + [cur]:
            pts = [(int(x * w * s), int(y * h * s)) for x, y in ln]
            for p in pts:
                cv2.circle(v, p, 4, (0, 255, 255), -1)
            for i in range(len(pts) - 1):
                cv2.line(v, pts[i], pts[i + 1], (0, 200, 255), 1)
        rms = line_residual(lines + [cur], state["k"], None)
        cv2.putText(v, f"k1={state['k']:.4f}  RMS={rms:.2f}px",
                    (700, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("k1: select dots on world's straight lines", v)
        und = undistort_image(disp, state["k"])
        cv2.imshow("k1: no distortion", und)

    def on_mouse(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append((x / (w * s), y / (h * s)))
            redraw()

    def on_track(v):
        state["k"] = (v - 900) / 1000.0     # -0.9 .. +0.6
        redraw()

    cv2.namedWindow("k1: select dots on world's straight lines")
    cv2.setMouseCallback("k1: select dots on world's straight lines", on_mouse)
    cv2.createTrackbar("k1*1000+900", "k1: select dots on world's straight lines",
                       int(round(state["k"] * 1000 + 900)), 1500, on_track)
    redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("n") and cur:
            lines.append(cur[:]); cur.clear(); redraw()
        elif key == ord("u"):
            if cur: cur.pop()
            elif lines: lines.pop()
            redraw()
        elif key == ord("c"):
            allv = lines + ([cur] if cur else [])
            r = minimize_scalar(lambda k: line_residual(allv, k, None), bounds=(-0.9, 0.3), method="bounded")
            state["k"] = float(r.x)
            cv2.setTrackbarPos("k1*1000+900", "k1: select dots on world's straight lines",
                               int(round(r.x * 1000 + 900)))
            print(f"optimal k1={r.x:.4f}, RMS={r.fun:.2f}px (uncorrected {line_residual(allv, 0.0, None):.2f}px)")
            redraw()
        elif key == ord("s"):
            cal.data["k1"] = state["k"]; cal.save(a.calib); print("k1 saved:", state["k"])
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()



def mode_points(a):
    cal = Calibration.load(a.calib)
    cam = a.camera
    img = grab_frame(a.root, cam, a.frame, a.image)
    plan = cv2.imread(a.plan)
    ih, iw = img.shape[:2]
    s = fit_scale(img)
    z = a.plan_zoom
    pairs = cal.cam(cam)["pairs"]
    pend = {"img": None}

    def draw():
        v = cv2.resize(img, None, fx=s, fy=s)
        p = cv2.resize(plan, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        for i, pr in enumerate(pairs):
            cv2.circle(v, (int(pr["img"][0] * iw * s), int(pr["img"][1] * ih * s)), 5, (0, 0, 255), -1)
            cv2.putText(v, str(i + 1), (int(pr["img"][0] * iw * s) + 6, int(pr["img"][1] * ih * s) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(p, (int(pr["plan"][0] * z), int(pr["plan"][1] * z)), 4, (0, 0, 255), -1)
            cv2.putText(p, str(i + 1), (int(pr["plan"][0] * z) + 5, int(pr["plan"][1] * z) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if pend["img"]:
            cv2.circle(v, (int(pend["img"][0] * iw * s), int(pend["img"][1] * ih * s)), 6, (0, 255, 0), 2)
        cal._H.pop(cam, None)
        err = cal.reproj_error(cam)
        t = f"cam {cam}: pairs={len(pairs)}  error={'-' if err is None else f'{err:.1f}px of plan'}"
        cv2.putText(v, t, (700, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(f"camera {cam}", v)
        cv2.imshow("office plan", p)

    def on_img(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pend["img"] = (x / (iw * s), y / (ih * s)); draw()

    def on_plan(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN and pend["img"]:
            pairs.append({"img": list(pend["img"]), "plan": [x / z, y / z]}); pend["img"] = None; draw()

    cv2.namedWindow(f"camera {cam}"); cv2.namedWindow("office plan")
    cv2.setMouseCallback(f"camera {cam}", on_img); cv2.setMouseCallback("office plan", on_plan)
    draw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("u"):
            if pend["img"]: pend["img"] = None
            elif pairs: pairs.pop()
            draw()
        elif key in (ord("s"), ord("q")):
            cal.save(a.calib); print("saved, pairs:", len(pairs))
            if key == ord("q"): break
        elif key == 27:
            break
    cv2.destroyAllWindows()



def poly_editor(base, polys, title, zoom_x, zoom_y, norm_w=None, norm_h=None, kinds=None):
    cur, mode = [], 0
    kinds = kinds or ["poly"]
    colors = {"walkable": (0, 200, 0), "obstacle": (0, 0, 255), "poly": (0, 200, 255)}

    def draw():
        v = cv2.resize(base, None, fx=zoom_x, fy=zoom_y, interpolation=cv2.INTER_NEAREST)
        ov = v.copy()
        for kd, pts in polys:
            pp = np.array([[x * zoom_x, y * zoom_y] for x, y in pts], np.int32)
            cv2.fillPoly(ov, [pp], colors[kd]); cv2.polylines(v, [pp], True, colors[kd], 2)
        v = cv2.addWeighted(ov, 0.3, v, 0.7, 0)
        for x, y in cur:
            cv2.circle(v, (int(x * zoom_x), int(y * zoom_y)), 4, (255, 255, 0), -1)
        if len(cur) > 1:
            cv2.polylines(v, [np.array([[x * zoom_x, y * zoom_y] for x, y in cur], np.int32)], False, (255, 255, 0), 1)
        cv2.putText(v, f"mode: {kinds[mode]}",
                    (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(title, v)

    def on_mouse(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append((x / zoom_x, y / zoom_y)); draw()

    cv2.namedWindow(title); cv2.setMouseCallback(title, on_mouse); draw()
    saved = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("n") and len(cur) >= 3:
            polys.append((kinds[mode], cur[:])); cur.clear(); draw()
        elif key == ord("m"):
            mode = (mode + 1) % len(kinds); draw()
        elif key == ord("u"):
            if cur: cur.pop()
            elif polys: polys.pop()
            draw()
        elif key == ord("s"):
            saved = True
            if len(cur) >= 3: polys.append((kinds[mode], cur[:])); cur.clear()
            break
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return polys, saved


def mode_walkable(a):
    cal = Calibration.load(a.calib)
    plan = cv2.imread(a.plan)
    polys = [("walkable", [tuple(p) for p in pl]) for pl in cal.data.get("walkable", [])] + \
            [("obstacle", [tuple(p) for p in pl]) for pl in cal.data.get("obstacles", [])]
    polys, ok = poly_editor(plan, polys, "walkable polygon on plan", a.plan_zoom, a.plan_zoom,
                            kinds=["walkable", "obstacle"])
    if ok:
        cal.data["walkable"] = [[list(p) for p in pts] for kd, pts in polys if kd == "walkable"]
        cal.data["obstacles"] = [[list(p) for p in pts] for kd, pts in polys if kd == "obstacle"]
        cal.save(a.calib); print("saved")


def mode_mask(a):
    cal = Calibration.load(a.calib)
    img = grab_frame(a.root, a.camera, a.frame, a.image)
    h, w = img.shape[:2]; s = fit_scale(img)
    polys = [("poly", [(x * w, y * h) for x, y in pl]) for pl in cal.cam(a.camera)["ghost_polygons"]]
    polys, ok = poly_editor(img, polys, f"ghost mask, camera {a.camera}", s, s)
    if ok:
        cal.cam(a.camera)["ghost_polygons"] = [[[x / w, y / h] for x, y in pts] for _, pts in polys]
        cal.save(a.calib); print("saved")


def mode_zoom(a):
    cal = Calibration.load(a.calib)
    img = grab_frame(a.root, a.camera, a.frame, a.image)
    h, w = img.shape[:2]; s = fit_scale(img)
    polys = [("poly", [(x1 * w, y1 * h), (x2 * w, y1 * h), (x2 * w, y2 * h), (x1 * w, y2 * h)])
             for x1, y1, x2, y2 in cal.cam(a.camera).get("zoom_rois", [])]
    polys, ok = poly_editor(img, polys, f"detailed search zones, camera {a.camera}", s, s)
    if ok:
        rois = []
        for _, pts in polys:
            xs = [p[0] / w for p in pts]; ys = [p[1] / h for p in pts]
            rois.append([round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)])
        cal.cam(a.camera)["zoom_rois"] = rois
        cal.save(a.calib); print("saved, zones (x1,y1,x2,y2):", rois)


def mode_check(a):
    cal = Calibration.load(a.calib)
    for c in CAMERAS:
        if not os.path.exists(video_path(a.root, c)):
            continue
        img = cv2.resize(grab_frame(a.root, c, a.frame), CANON)
        if cal.is_calibrated(c):
            print(f"cam {c}: pairs {len(cal.cam(c)['pairs'])}, error {cal.reproj_error(c):.1f}px of plan")
            for poly in cal.data.get("walkable", []):
                pts = cal.plan_to_cam(c, poly)
                cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        gm = cal.ghost_mask(c)
        img[gm > 0] = (0.5 * img[gm > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        for x1, y1, x2, y2 in cal.cam(c).get("zoom_rois", []):
            cv2.rectangle(img, (int(x1 * CANON[0]), int(y1 * CANON[1])), (int(x2 * CANON[0]), int(y2 * CANON[1])), (255, 200, 0), 2)
        cv2.imwrite(f"calib_check_cam{c}.jpg", img)
    print("done: calib_check_cam*.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["k", "points", "walkable", "mask", "zoom", "check"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--calib", default=CALIB_FILE)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--image", default=None, help="use pic instead of vid")
    ap.add_argument("--plan", default="officePlan.jpg")
    ap.add_argument("--plan-zoom", type=float, default=2.0)
    a = ap.parse_args()
    if a.mode == "k":
        a.image = a.image or "koef.png"
    {"k": mode_k, "points": mode_points, "walkable": mode_walkable, "mask": mode_mask, "zoom": mode_zoom, "check": mode_check}[a.mode](a)
