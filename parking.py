# parking.py
import os
import statistics
import tempfile
from collections import deque

import cv2
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


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# Glavno okno (levo) in nastavitve (desno)
col_main, col_settings = st.columns([4, 1])

with col_settings:
    st.header("Nastavitve")
    conf = st.slider("Prag zaupanja (conf)", 0.1, 0.9, 0.30, 0.05)
    imgsz = st.selectbox("Ločljivost inference", [640, 960, 1280], index=1)
    show_boxes = st.checkbox("Prikaži škatle", value=True)
    show_ids = st.checkbox("Prikaži ID vozila", value=True)
    skip = st.slider("Obdelaj vsak N-ti frame", 1, 10, 1)
    smooth_n = st.slider("Glajenje števca (št. framov)", 1, 15, 7)
    font_scale = st.slider("Velikost pisave oznak", 0.3, 1.5, 0.5, 0.1)
    save_every = st.number_input(
        "Samodejno shrani frame na vsakih N obdelanih (0 = izklopljeno)",
        min_value=0, value=0, step=10,
    )
    st.caption("Razredi modela:")
    st.json(model.names)

with col_main:
    uploaded = st.file_uploader("Naloži video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        col_video, col_stats = st.columns([3, 1])
        with col_video:
            frame_placeholder = st.empty()
        with col_stats:
            st.subheader("Števec vozil")
            metric_placeholder = st.empty()
            unique_placeholder = st.empty()
            max_placeholder = st.empty()
            frame_info = st.empty()
            progress = st.progress(0)

        frame_idx = 0
        processed = 0
        max_count = 0
        last_annotated = None
        seen_ids = set()                 # unikatna vozila (po ID-ju, ne po razredu)
        recent = deque(maxlen=smooth_n)  # za glajenje števca

        # Ponastavi sledilnik, da se ID-ji začnejo znova za nov video
        model.predictor = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

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
            seen_ids.update(int(i) for i in ids)

            recent.append(n)
            n_smooth = int(statistics.median(recent))
            max_count = max(max_count, n_smooth)

            # Risanje škatel (brez razredov, samo ID)
            annotated = frame.copy()
            if show_boxes:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 1)
                    if show_ids and i < len(ids):
                        cv2.putText(
                            annotated,
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
                annotated,
                f"Vozila: {n_smooth}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )
            last_annotated = annotated

            # Prikaz v Streamlitu (OpenCV je BGR, Streamlit RGB)
            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                output_format="JPEG",
                use_container_width=True,
            )
            metric_placeholder.metric("Vozil na sliki", n_smooth)
            unique_placeholder.metric("Unikatnih vozil", len(seen_ids))
            max_placeholder.metric("Največ vozil hkrati", max_count)
            frame_info.caption(f"Frame {frame_idx} / {total}")
            progress.progress(min(frame_idx / total, 1.0))

            if save_every and processed % int(save_every) == 0:
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg"), annotated
                )

        cap.release()
        try:
            os.unlink(tfile.name)
        except OSError:
            pass

        progress.progress(1.0)
        st.success(
            f"Končano. Unikatnih vozil: {len(seen_ids)}, "
            f"največ hkrati: {max_count}"
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
