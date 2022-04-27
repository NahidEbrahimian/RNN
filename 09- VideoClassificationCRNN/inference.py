from models import Models
import numpy as np
import cv2
import os
import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd
from load_video import load_video

def preparing_input(video_path, resize):

  input_frames = []

  frames, num_frames = load_video(video_path, resize)
  input_masks = np.zeros(shape=(1, num_frames), dtype="bool")
  
  frames = np.array(frames).astype("float32") / 255. # Normalize video frames
  frames = frames[np.newaxis, ...]

  input_masks[0, :] = 1

  return frames, input_masks, num_frames



parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
args = parser.parse_args()

input_path = args.input_path
frame_width = 110
frame_height = 110


if not os.path.exists("outputs"): # Create output directory
  os.makedirs("outputs")


input_frames, input_masks, num_frames = preparing_input(input_path, (frame_width, frame_height)) # prepairing input
class_name = ["Natrual", "Elahi Shokr"]

# Load model and weights
models = Models((frame_height, frame_width), num_frames)
model = models.GRU_model()

# if not os.path.exists("weights"):
#     os.makedirs("weights")
#         # os.makedirs(model_path)
# gdd.download_file_from_google_drive(file_id="1Aw7zEKmI4rdsEtVwhYOqLDC_0KnTLb-7",
#                                         dest_path='weights/gru_model.h5')

model.load_weights("weights/gru_model.h5")
pred = model.predict([input_frames, input_masks])

predicted_class = np.argmax(pred)
label = class_name[predicted_class]


# Write video
color = (0, 0, 0)
video = cv2.VideoCapture(input_path)

height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("outputs/out_put.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while True:
    ret, frame = video.read()
    
    if ret == True:
      cv2.putText(frame, label, (width // 10, height // 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1,
                  cv2.LINE_AA)

      video_writer.write(frame)
        
    else:
      break

video.release()
video_writer.release()
cv2.destroyAllWindows()

