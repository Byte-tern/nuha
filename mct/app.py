import argparse
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from common import video_path, CANON, seek_read
from result import Result, draw_video, draw_plan

DEFAULT_CAMS = [2, 4, 5]
COLW = {"id": 90, "start": 150, "fin": 150, "cam": 55}


class VideoZone:

    def __init__(self, root, cam, res):
        self.root, self.res, self.cam, self.cap, self.pos, self.img, self.pts = root, res, None, None, -2, None, None
        self.set_cam(cam)

    def set_cam(self, cam):
        if self.cap is not None:
            self.cap.release()
        self.cam = cam
        self.cap = cv2.VideoCapture(video_path(self.root, cam))
        self.pos, self.img = -2, None
        self.pts = self.res.pts_of(cam)
        if self.pts is None:                
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.pts = np.arange(max(n, 1)) * 1000.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

    def get(self, i):
        if i == self.pos and self.img is not None:
            return self.img
        if self.pos < i <= self.pos + 40:
            for _ in range(i - self.pos - 1):
                self.cap.grab()
            ok, im = self.cap.read()
            if ok:
                self.img, self.pos = im, i
        else:
            im, j = seek_read(self.cap, self.pts, i)
            if im is not None:
                self.img, self.pos = im, j
        return self.img


class App:
    def __init__(self, tk_root, a):
        self.t = tk_root
        self.res = Result(a.result)
        self.plan = cv2.imread(a.plan)
        self.root_dir = a.root
        self.frame = self.res.start
        self.playing = False
        self.selected = None
        self.last_drawn = None
        self.last_table = 0.0
        self.anchor = (0.0, self.frame)
        self._slider_lock = False

        t = self.t
        t.title("ДЕМО - Цифровые знахари")
        sh = t.winfo_screenheight()
        self.vh = max(200, (sh - 90) // 3); self.vw = self.vh * 4 // 3
        self.pw = int(3 * self.vh * self.plan.shape[1] / self.plan.shape[0]); self.ph = 3 * self.vh
        left = tk.Frame(t); left.grid(row=0, column=0, rowspan=2, sticky="n")
        self.zones, self.labels, self.combos, self.vars = [], [], [], []
        for z in range(3):
            fr = tk.Frame(left, width=self.vw, height=self.vh, bd=1, relief="solid")
            fr.grid(row=z, column=0); fr.grid_propagate(False)
            lb = tk.Label(fr, bg="black"); lb.place(x=0, y=0, width=self.vw, height=self.vh)
            var = tk.StringVar(value=f"Камера {DEFAULT_CAMS[z]}")
            cb = ttk.Combobox(fr, textvariable=var, state="readonly", width=9,
                              postcommand=lambda z=z: self.refresh_values(z))
            cb.place(relx=1.0, y=0, anchor="ne")
            cb.bind("<<ComboboxSelected>>", lambda e, z=z: self.on_cam(z))
            self.zones.append(VideoZone(a.root, DEFAULT_CAMS[z], self.res)); self.labels.append(lb)
            self.combos.append(cb); self.vars.append(var)
            cb["values"] = [f"Камера {c}" for c in self.res.cameras]

        mid = tk.Frame(t); mid.grid(row=0, column=1, sticky="nsew")
        hdr = tk.Frame(mid, bd=1, relief="solid"); hdr.pack(fill="x")
        tk.Label(hdr, text="ДЕМО - Цифровые знахари", font=("Arial", 20, "bold")).pack(anchor="w", padx=8, pady=(4, 0))
        self.info = tk.Label(hdr, text="", font=("Arial", 12), justify="left"); self.info.pack(anchor="w", padx=8)
        ctl = tk.Frame(hdr); ctl.pack(fill="x", padx=8, pady=4)
        self.btn = ttk.Button(ctl, text="▶ Пуск", width=9, command=self.toggle); self.btn.pack(side="left")
        ttk.Button(ctl, text="Снять выбор трека", command=self.clear_sel).pack(side="left", padx=6)
        self.tlabel = tk.Label(ctl, text="", font=("Consolas", 11)); self.tlabel.pack(side="right")
        self.scale = tk.Scale(hdr, from_=self.res.start, to=self.res.last_frame, orient="horizontal",
                              showvalue=False, command=self.on_scale)
        self.scale.pack(fill="x", padx=8, pady=(0, 4))

        tab = tk.Frame(mid, bd=1, relief="solid"); tab.pack(fill="both", expand=True)
        offs = COLW["id"] + COLW["start"] + COLW["fin"]
        top = tk.Frame(tab, height=20); top.pack(fill="x")
        tk.Label(top, text="On camera", bd=1, relief="solid").place(x=offs, y=0, width=5 * COLW["cam"], height=20)
        cols = ("id", "start", "fin", "c1", "c2", "c3", "c4", "c5")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings", selectmode="browse", height=30)
        for c, text, w in (("id", "ID", COLW["id"]), ("start", "Started", COLW["start"]), ("fin", "Finished", COLW["fin"]),
                           ("c1", "1", COLW["cam"]), ("c2", "2", COLW["cam"]), ("c3", "3", COLW["cam"]),
                           ("c4", "4", COLW["cam"]), ("c5", "5", COLW["cam"])):
            self.tree.heading(c, text=text); self.tree.column(c, width=w, anchor="center", stretch=False)
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Button-1>", self.on_click, add="+")
        self.cache = {}

        right = tk.Frame(t, bd=1, relief="solid"); right.grid(row=0, column=2, rowspan=2, sticky="n")
        self.plan_lb = tk.Label(right, width=self.pw, height=self.ph, bg="white")
        self.plan_lb.pack()
        t.bind("<space>", lambda e: self.toggle()); t.bind("<Escape>", lambda e: self.clear_sel())
        self.render(force=True)
        self.tick()

    def current_cams(self):
        return [int(v.get().split()[-1]) for v in self.vars]

    def refresh_values(self, z):
        used = {c for i, c in enumerate(self.current_cams()) if i != z}
        self.combos[z]["values"] = [f"Камера {c}" for c in self.res.cameras if c not in used]

    def on_cam(self, z):
        cams = self.current_cams()
        if cams.count(cams[z]) > 1:
            messagebox.showwarning("Выбор камеры", "Эта камера уже выбрана в другой зоне.")
            self.vars[z].set(f"Камера {self.zones[z].cam}")
            return
        self.zones[z].set_cam(cams[z])
        self.render(force=True)


    def toggle(self):
        if self.playing:
            self.playing = False; self.btn.config(text="Пуск")
        else:
            if self.frame >= self.res.last_frame:
                self.frame = self.res.start
            self.playing = True; self.anchor = (time.perf_counter(), self.frame); self.btn.config(text="Пауза")

    def on_scale(self, v):
        if self._slider_lock:
            return
        self.frame = int(float(v)); self.anchor = (time.perf_counter(), self.frame)
        self.render(force=True)

    def on_select(self, _):
        sel = self.tree.selection()
        self.selected = int(sel[0]) if sel else None
        self.render(force=True)

    def on_click(self, e):
        row = self.tree.identify_row(e.y)
        if row and self.tree.selection() == (row,):
            self.t.after_idle(self.clear_sel)

    def clear_sel(self):
        self.tree.selection_remove(self.tree.selection()); self.selected = None; self.render(force=True)

    def tick(self):
        if self.playing:
            el = time.perf_counter() - self.anchor[0]
            tgt = min(self.anchor[1] + int(el * self.res.fps), self.res.last_frame)
            if tgt != self.frame:
                self.frame = tgt
                self.render()
            if self.frame >= self.res.last_frame:
                self.playing = False; self.btn.config(text="Пуск")
        self.t.after(8, self.tick)

    def render(self, force=False):
        f = self.frame
        if not force and f == self.last_drawn:
            return
        self.last_drawn = f
        for z, zone in enumerate(self.zones):
            img = zone.get(self.res.src_frame(zone.cam, f))
            if img is None:
                continue
            ids, bx = self.res.boxes(f, zone.cam)
            v = draw_video(img, ids, bx, (self.vw, self.vh))
            ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)))
            self.labels[z].configure(image=ph); self.labels[z].image = ph
        p = draw_plan(self.plan, self.res, f, self.selected, (self.pw, self.ph))
        ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
        self.plan_lb.configure(image=ph); self.plan_lb.image = ph
        from common import fmt_time
        self.tlabel.config(text=fmt_time(f - self.res.start, self.res.fps))
        self._slider_lock = True; self.scale.set(f); self._slider_lock = False
        now = time.perf_counter()
        if force or now - self.last_table > 0.2:
            self.last_table = now; self.fill_table(f)

    def fill_table(self, f):
        rows = self.res.table(f)
        now_ids = {r[0] for r in rows}
        for iid in list(self.cache):
            if iid not in now_ids:
                self.tree.delete(str(iid)); del self.cache[iid]
        for i, ids, st, fin, cams in rows:
            vals = (ids, st, fin, *cams)
            if i not in self.cache:
                self.tree.insert("", "end", iid=str(i), values=vals)
            elif self.cache[i] != vals:
                self.tree.item(str(i), values=vals)
            self.cache[i] = vals
        if self.selected is not None and self.selected not in now_ids:
            self.selected = None
        n_act = len(self.res.active_ids(f))
        self.info.config(text=f"Количество треков: {len(rows)}    Активных треков: {n_act}\n"
                              f"Выбранный трек: {self.selected if self.selected is not None else '—'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--result", default="fusion_result.pkl")
    ap.add_argument("--plan", default="officePlan.jpg")
    a = ap.parse_args()
    root = tk.Tk()
    App(root, a)
    root.mainloop()
