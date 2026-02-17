import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from pose_checker import analyze_pose_pa, analyze_pose_lateral
import av
from PIL import Image, ImageDraw, ImageFont

# Configure MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Real-time Chest X-ray Pose Checker", layout="wide")
st.title("📹 ตรวจจับท่าถ่าย X-ray ทรวงอกแบบเรียลไทม์")

# WebRTC configuration for deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

THAI_FONT_PATH = "fonts/THSarabun.ttf" 

def draw_text_pil(img, text, position, color=(0,255,0), font_size=32):
    """Draw Thai text on OpenCV image using PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(THAI_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.feedback_messages = []
        self.pose_type = "PA"  # Default

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontally to remove mirror effect
        img = cv2.flip(img, 1)
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Get feedback based on pose type
            if self.pose_type == "PA":
                feedback = analyze_pose_pa(results.pose_landmarks.landmark)
            else:  # Lateral
                feedback = analyze_pose_lateral(results.pose_landmarks.landmark)
            
            self.feedback_messages = feedback
            
            # Draw pose type indicator
            pose_label = f"ท่า: {self.pose_type}"
            img = draw_text_pil(img, pose_label, (10, 10), color=(255, 255, 0), font_size=28)
            
            # Draw feedback text on image
            y = 50
            for msg in feedback:
                color = (0, 255, 0) if "✅" in msg else (0, 0, 255)
                img = draw_text_pil(img, msg, (10, y), color=color, font_size=32)
                y += 40
        else:
            self.feedback_messages = ["❌ ไม่พบการตรวจจับท่าทาง"]
            img = draw_text_pil(img, "ไม่พบการตรวจจับท่าทาง", (10, 30), color=(0, 0, 255), font_size=32)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar for pose selection
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    pose_type = st.radio(
        "เลือกท่าที่ต้องการตรวจสอบ:",
        options=["PA", "Lateral"],
        index=0,
        help="เลือกท่าถ่าย X-ray ที่ต้องการตรวจสอบ"
    )
    
    st.markdown("---")
    st.markdown("### 📋 เกณฑ์การตรวจสอบ")
    
    if pose_type == "PA":
        st.markdown("""
        **ท่า PA (Posterior-Anterior)**
        -  ไหล่เท่ากัน
        -  ศีรษะอยู่ตรงกลาง
        -  วางมือบริเวณสะโพก
        -  โน้มไหล่ไปด้านหน้า
        -  SID 180 cm
        -  References point = Inferior angle of scapula
        -  Center ray = T6
        """)
    else:
        st.markdown("""
        **ท่า Lateral (ด้านข้าง)**
        -  ไหล่ซ้อนกัน
        -  ยกมือขึ้นเหนือหัว
        -  ศีรษะอยู่ตรงกลาง
        -  SID 100 cm
        -  References point = Inferior angle of scapula
        -  Center ray = T6
        """)

# Instructions
st.markdown("""
### 📋 วิธีใช้งาน:
1. เลือกท่าที่ต้องการตรวจสอบจากแถบด้านซ้าย (PA หรือ Lateral)
2. คลิก "Start" เพื่อเปิดกล้อง
3. อนุญาตการเข้าถึงกล้องเมื่อเบราว์เซอร์ถาม
4. วางตัวให้อยู่ในกรอบกล้อง
5. ดูผลการวิเคราะห์ท่าทางแบบเรียลไทม์
""")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="pose-detection",
    video_processor_factory=PoseVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Update pose type in the video processor
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.pose_type = pose_type

# Troubleshooting section
st.markdown("""
---
### 🔧 หากมีปัญหา:
- **กล้องไม่เปิด**: ตรวจสอบว่าอนุญาตการเข้าถึงกล้องแล้ว
- **ช้า**: ลองรีเฟรชหน้าเว็บ
- **ไม่ตรวจจับท่า**: ตรวจสอบแสงและตำแหน่งในกรอบกล้อง
- **เปลี่ยนท่า**: หยุดกล้องแล้วเลือกท่าใหม่จาก Sidebar แล้วกด Start อีกครั้ง
""")

# Add CSS for better styling
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.main-header {
    text-align: center;
    color: #1e3a8a;
    font-size: 2rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)