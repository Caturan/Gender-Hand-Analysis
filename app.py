import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
import pandas as pd

# --- Feature Data ---
FEMALE_BRAIN_FEATURES = [
    "Duygusal farkındalık yüksek",
    "Empati kurma becerisi güçlü",
    "Sözel iletişim becerisi gelişmiş",
    "Yüz ifadeleri ve beden dilini anlama konusunda başarılı",
    "Multitasking (çoklu görev yürütme) becerisi yüksek",
    "Duygusal hafıza güçlü",
    "Stres altında sosyal destek arama eğilimi ('tend-and-befriend')",
    "Detaylara dikkat yüksek",
    "Sosyal ilişkilere hassaslık",
    "Yargılayıcı olmadan dinleme becerisi",
    "Zamanlama ve organizasyon becerileri gelişmiş",
    "Dil merkezleri her iki yarımkürede yaygın",
    "İçsel konuşma (self-talk) daha aktif"
]

MALE_BRAIN_FEATURES = [
    "Uzamsal zekâ güçlü",
    "Problem çözme odaklı yaklaşım",
    "Tek işe odaklanma (mono-tasking) yeteneği yüksek",
    "Sistematik düşünme eğilimi",
    "Rekabetçi davranış eğilimi",
    "Risk alma eğilimi daha yüksek",
    "Analitik düşünme",
    "Duygusal ifadeyi bastırma eğilimi",
    "Görsel işleme becerisi güçlü",
    "Kısa süreli hedeflere odaklanma eğilimi",
    "Motor koordinasyon ve el-göz uyumu güçlü",
    "Stres altında içe kapanma ('fight-or-flight') tepkisi baskın olabilir"
]

# --- Hand Analysis Class ---
class HandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def analyze_frame(self, frame):
        """Analyzes a single frame, draws landmarks, and returns gender."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        gender = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Index finger length (landmarks 5 to 8)
                index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_length = np.linalg.norm([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])

                # Ring finger length (landmarks 13 to 16)
                ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_finger_length = np.linalg.norm([ring_tip.x - ring_mcp.x, ring_tip.y - ring_mcp.y])

                gender = "Kadın" if index_finger_length > ring_finger_length else "Erkek"
        
        return frame, gender

# Create a single instance of the analyzer
analyzer = HandAnalyzer()

# --- Gradio Processing Function ---
def process_image(image):
    """
    Takes a webcam image, processes it, and returns the processed image
    and the corresponding feature table.
    """
    if image is None:
        return None, "Lütfen elinizi kameraya gösterin."

    # Flip the image for a more natural webcam view
    frame = cv2.flip(image, 1)
    
    processed_frame, gender = analyzer.analyze_frame(frame)
    
    feature_table_md = ""
    if gender:
        if gender == "Kadın":
            title = "Kadın Beyni Özellikleri"
            features = FEMALE_BRAIN_FEATURES
            df = pd.DataFrame({"Özellikler": features})
            feature_table_md = f"## {title}\n" + df.to_markdown(index=False)
        else: # Erkek
            title = "Erkek Beyni Özellikleri"
            features = MALE_BRAIN_FEATURES
            df = pd.DataFrame({"Özellikler": features})
            feature_table_md = f"## {title}\n" + df.to_markdown(index=False)
    else:
        feature_table_md = "El algılanmadı. Lütfen elinizi kameraya daha net gösterin."

    return processed_frame, feature_table_md


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Parmak Uzunluğuna Göre Cinsiyet ve Beyin Özellikleri Analizi")
    gr.Markdown(
        "Bu uygulama, elinizin işaret ve yüzük parmağı uzunluklarını karşılaştırarak cinsiyet tahmini yapar "
        "ve ilgili cinsiyetin genel beyin işlevsel özelliklerini gösterir."
    )
    with gr.Row():
        webcam_input = gr.Image(sources="webcam", streaming=True, label="Webcam")
        with gr.Column():
            processed_output = gr.Image(label="İşlenmiş Görüntü")
            feature_output = gr.Markdown(label="Özellikler Tablosu")

    webcam_input.stream(
        process_image,
        inputs=[webcam_input],
        outputs=[processed_output, feature_output]
    )

if __name__ == "__main__":
    demo.launch() 