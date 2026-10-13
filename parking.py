# parking.py
import math
import os
import statistics
import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Štetje vozil", layout="wide")

col_logo, col_title = st.columns([1, 8], vertical_alignment="center")
with col_logo:
    st.image("./logo.jpg", width=100)
with col_title:
    st.title("Štetje vozil na parkirišču - © Simon Pogorelčnik")


# st.logo("./logo.jpg", size="large", link=None, icon_image=None)
# st.title("Štetje vozil na parkirišču -  ©  Simon Pogorelčnik")
st.caption("Sprotni števec prepoznanih vozil med predvajanjem")

# --- Nastavitve ---
MODEL_PATH = "./Model/visdrone-yolo11n.pt"
VEHICLE_CLASSES = [3, 4, 5, 8]  # car, van, truck, bus (VisDrone)
OUTPUT_DIR = "./frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOX_COLOR = (0, 200, 0)  # BGR
MAX_W = 1280             # največja širina frama za obdelavo
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


class FlowCounter:
    """Prihodi in odhodi vozil na podlagi ID-jev sledilnika.

    Prihod: vozilo, ki se po začetku videa prvič pojavi v robnem pasu slike in se
            nato premakne vsaj MOVE_FRAC širine slike od začetnega položaja.
    Odhod:  vozilo, ki ga `lost_frames` obdelanih framov ni več zaznati, zadnjič
            pa je bilo v robnem pasu slike in se je tik pred tem premikalo.
    Vozila, ki so v sliki že na začetku (prvih WARMUP framov), niso prihodi.
    """
    MIN_HITS = 5
    WARMUP = 10
    MOVE_FRAC = 0.03

    def __init__(self):
        self.tracks = {}
        self.arrivals = 0
        self.departures = 0
        self.frames = 0

    @staticmethod
    def _in_edge(pos, w, h, margin):
        x, y = pos
        mx, my = w * margin, h * margin
        return x < mx or x > w - mx or y < my or y > h - my

    def update(self, ids, centers, w, h, margin, lost_frames):
        self.frames += 1
        move_thr = self.MOVE_FRAC * w
        seen = set()

        for tid, c in zip(ids, centers):
            seen.add(tid)
            t = self.tracks.get(tid)
            if t is None:
                self.tracks[tid] = {
                    "first": c, "last": c, "trail": deque([c], maxlen=15),
                    "hits": 1, "missed": 0,
                    "initial": self.frames <= self.WARMUP,
                    "arrived": False, "departed": False,
                }
                continue
            if t["departed"]:  # vozilo se je vrnilo -> prekliči odhod
                t["departed"] = False
                self.departures -= 1
            t["last"] = c
            t["trail"].append(c)
            t["hits"] += 1
            t["missed"] = 0
            if (not t["initial"] and not t["arrived"] and t["hits"] >= self.MIN_HITS
                    and self._in_edge(t["first"], w, h, margin)
                    and math.dist(t["first"], c) >= move_thr):
                t["arrived"] = True
                self.arrivals += 1

        for tid, t in self.tracks.items():
            if tid in seen or t["departed"]:
                continue
            t["missed"] += 1
            if (t["missed"] == lost_frames and t["hits"] >= self.MIN_HITS
                    and self._in_edge(t["last"], w, h, margin)
                    and math.dist(t["trail"][0], t["trail"][-1]) >= move_thr):
                t["departed"] = True
                self.departures += 1


def resize_frame(frame):
    h, w = frame.shape[:2]
    if w > MAX_W:
        frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
    return frame


def find_video_dir():
    """Mapa Video v repozitoriju (ne glede na velikost začetnice)."""
    for d in BASE_DIR.iterdir():
        if d.is_dir() and d.name.lower() == "video":
            return d
    return None


VIDEO_DIR = find_video_dir()
REPO_VIDEOS = (
    sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if VIDEO_DIR
    else []
)

# Glavno okno (levo) in nastavitve (desno)
col_main, col_settings = st.columns([4, 1])

with col_settings:
    st.header("Nastavitve")
    conf = st.slider("Prag zaupanja (conf)", 0.10, 0.90, 0.30, 0.01)
    imgsz = st.selectbox("Ločljivost inference", [640, 960, 1280], index=1)
    show_boxes = st.checkbox("Prikaži škatle", value=True)
    show_ids = st.checkbox("Prikaži ID vozila", value=True)
    skip = st.slider("Obdelaj vsak N-ti frame", 1, 10, 1)
    smooth_n = st.slider("Glajenje števca (št. framov)", 1, 15, 7)
    font_scale = st.slider("Velikost pisave oznak", 0.3, 1.5, 0.5, 0.1)
    edge_pct = st.slider(
        "Robni pas prihodov/odhodov (%)", 0, 50, 15,
        help="Širina pasu ob robu slike, kjer vozila vstopajo ali izstopajo. "
             "0 = štetje izklopljeno, 50 = šteje se kjerkoli v sliki.",
    )
    lost_frames = st.slider(
        "Zakasnitev odhoda (framov)", 10, 90, 30,
        help="Koliko obdelanih framov mora vozilo manjkati, preden se šteje kot odšlo.",
    )
    save_every = st.number_input(
        "Samodejno shrani frame na vsakih N obdelanih (0 = izklopljeno)",
        min_value=0, value=0, step=10,
    )
    st.caption("Razredi modela:")
    st.json(model.names)


def annotate(frame, boxes, ids, n):
    """Nariše škatle (samo ID, brez razredov) in števec na sliko."""
    out = frame.copy()
    if show_boxes:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 1)
            if show_ids and i < len(ids):
                cv2.putText(
                    out,
                    f"#{int(ids[i])}",
                    (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    BOX_COLOR,
                    1,
                    cv2.LINE_AA,
                )

    # HUD števec direktno na sliki
    cv2.putText(
        out,
        f"Vozila: {n}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        4,
        cv2.LINE_AA,
    )
    return out


with col_main:
    # Vir: video iz mape Video v repozitoriju ali lastna naložena datoteka
    source = "Naloži datoteko"
    if REPO_VIDEOS:
        source = st.radio(
            "Vir", ["Video iz mape Video", "Naloži datoteko"], horizontal=True
        )

    uploaded = None
    repo_path = None
    if source == "Video iz mape Video":
        chosen = st.selectbox("Izberi video", REPO_VIDEOS)
        if st.button("▶ Začni obdelavo", type="primary"):
            repo_path = VIDEO_DIR / chosen
    else:
        uploaded = st.file_uploader(
            "Naloži video ali sliko",
            type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp", "webp"],
        )

    if uploaded is not None or repo_path is not None:
        is_image = uploaded is not None and uploaded.name.lower().endswith(IMAGE_EXT)

        col_video, col_stats = st.columns([3, 1])
        with col_video:
            frame_placeholder = st.empty()
        with col_stats:
            st.subheader("Števec vozil")
            now_placeholder = st.empty()
            in_placeholder = st.empty()
            out_placeholder = st.empty()
            frame_info = st.empty()
            progress = st.progress(0)

        last_annotated = None

        # Ponastavi sledilnik, da se ID-ji začnejo znova
        model.predictor = None

        if is_image:
            # ---------- SLIKA (brez sledenja) ----------
            img = cv2.imdecode(
                np.frombuffer(uploaded.read(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                st.error("Slike ni mogoče prebrati.")
            else:
                img = resize_frame(img)
                results = model.predict(
                    img,
                    classes=VEHICLE_CLASSES,
                    conf=conf,
                    imgsz=imgsz,
                    agnostic_nms=True,
                    verbose=False,
                )
                boxes = results[0].boxes
                n = len(boxes)
                last_annotated = annotate(img, boxes, [], n)

                frame_placeholder.image(
                    cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    output_format="JPEG",
                    use_container_width=True,
                )
                now_placeholder.metric("Trenutno", n)
                in_placeholder.metric("Prihodi", "–")
                out_placeholder.metric("Odhodi", "–")
                progress.progress(1.0)
                st.success(f"Slika obdelana. Število vozil: {n}")

        else:
            # ---------- VIDEO (sledenje + prihodi/odhodi) ----------
            if repo_path is not None:
                video_file = str(repo_path)
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read())
                tfile.flush()
                tfile.close()
                video_file = tfile.name

            cap = cv2.VideoCapture(video_file)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

            frame_idx = 0
            processed = 0
            flow = FlowCounter()
            recent = deque(maxlen=smooth_n)  # za glajenje števca

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % skip != 0:
                    continue

                frame = resize_frame(frame)

                # Sledenje: agnostic_nms = ena škatla na vozilo, ne glede na razred
                results = model.track(
                    frame,
                    classes=VEHICLE_CLASSES,
                    conf=conf,
                    imgsz=imgsz,
                    agnostic_nms=True,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                boxes = results[0].boxes
                n = len(boxes)  # vsa vozila, razred ni pomemben
                processed += 1

                ids = boxes.id.tolist() if boxes.id is not None else []
                centers = [(b[0], b[1]) for b in boxes.xywh.tolist()]
                fh, fw = frame.shape[:2]
                flow.update(
                    [int(i) for i in ids], centers, fw, fh,
                    edge_pct / 100, lost_frames,
                )

                recent.append(n)
                n_smooth = int(statistics.median(recent))

                annotated = annotate(frame, boxes, ids, n_smooth)
                last_annotated = annotated

                # Prikaz v Streamlitu (OpenCV je BGR, Streamlit RGB)
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    output_format="JPEG",
                    use_container_width=True,
                )
                now_placeholder.metric("Trenutno", n_smooth)
                in_placeholder.metric("Prihodi", flow.arrivals)
                out_placeholder.metric("Odhodi", flow.departures)
                frame_info.caption(f"Frame {frame_idx} / {total}")
                progress.progress(min(frame_idx / total, 1.0))

                if save_every and processed % int(save_every) == 0:
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg"),
                        annotated,
                    )

            cap.release()
            if repo_path is None:  # začasno datoteko brišemo samo pri naloženih videih
                try:
                    os.unlink(video_file)
                except OSError:
                    pass

            progress.progress(1.0)
            st.success(
                f"Končano. Prihodov: {flow.arrivals}, odhodov: {flow.departures}"
            )

        # Gumb za shranjevanje je izven zanke, ker klik na gumb v Streamlitu
        # ponovno požene skript in bi prekinil predvajanje.
        if last_annotated is not None:
            ok, buf = cv2.imencode(".jpg", last_annotated)
            if ok:
                st.download_button(
                    "💾 Prenesi zadnji frame",
                    data=buf.tobytes(),
                    file_name="zadnji_frame.jpg",
                    mime="image/jpeg",
                )
