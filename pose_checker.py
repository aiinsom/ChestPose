import mediapipe as mp

mp_pose = mp.solutions.pose

def analyze_pose_pa(landmarks):
    """วิเคราะห์ท่า Chest PA"""
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    messages = []

    # ✅ ไหล่เท่ากัน
    if abs(left_shoulder.y - right_shoulder.y) < 0.02:
        messages.append("✅ ไหล่เท่ากัน")
    else:
        messages.append("❌ ไหล่ไม่เท่ากัน")

    # ✅ ศีรษะอยู่ตรงกลาง
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    if abs(nose.x - mid_x) < 0.05:
        messages.append("✅ ศีรษะอยู่ตรงกลาง")
    else:
        messages.append("❌ ศีรษะเอียงหรือหมุน")

    # ✅ มือเท้าเอว (วางใกล้สะโพก)
    lw_close = abs(left_wrist.y - left_hip.y) < 0.07
    rw_close = abs(right_wrist.y - right_hip.y) < 0.07
    if lw_close and rw_close:
        messages.append("✅ วางมือบริเวณสะโพก")
    else:
        messages.append("❌ ยังไม่ได้วางมือบริเวณสะโพก")

    # ✅ โน้มไหล่ไปข้างหน้า (z ของไหล่ < สะโพก = ยื่นไปข้างหน้า)
    left_lean = left_shoulder.z < left_hip.z - 0.05
    right_lean = right_shoulder.z < right_hip.z - 0.05
    if left_lean and right_lean:
        messages.append("✅ โน้มไหล่ไปด้านหน้า")
    else:
        messages.append("❌ ยังไม่โน้มไหล่ไปด้านหน้า")

    return messages


def analyze_pose_lateral(landmarks):
    """วิเคราะห์ท่า Chest Lateral"""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    messages = []

    # ✅ ไหล่ซ้อนกัน (ดูจาก x-coordinate - มองจากด้านข้าง)
    if abs(left_shoulder.x - right_shoulder.x) < 0.1:
        messages.append("✅ ไหล่ซ้อนกัน")
    else:
        messages.append("❌ ไหล่ไม่ซ้อนกัน")

    # ✅ ยกมือขึ้นเหนือหัว
    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        messages.append("✅ ยกมือขึ้นเหนือหัว")
    else:
        messages.append("❌ ยังไม่ได้ยกมือขึ้น")

    # ✅ ศีรษะอยู่ตรงกลาง
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    if abs(nose.x - mid_x) < 0.05:
        messages.append("✅ ศีรษะอยู่ตรงกลาง")
    else:
        messages.append("❌ ศีรษะเอียงหรือหมุน")

    return messages