import ffmpeg
import numpy as np
from PIL import Image

# Config Data

EFFECT_DIRECTORY = ""
THUMBS_DIRECTORY = ""
IMPORT_DIRECTORY = ""


def save_eff_as_image(np_array, filename):
  image = Image.fromarray(np_array.astype(np.uint8))
  image.save("./thumbs/" + filename, format="png")

def convert_gif_to_np(image, func='scale', axis=0):
  effect_name = image.split('.')[0]
  import_url = './imports/' + image 
  try:
    probe = ffmpeg.probe(import_url)
  except Exception as e:
    print(e)

  gif_x = probe['streams'][0]['width']
  gif_y = probe['streams'][0]['height']
  center_x = gif_x // 2
  center_y = gif_y // 2 - 30

  try:
    gif_input = ffmpeg.input(import_url)
  except Exception as e:
      print(e)

  if func == 'crop':
    if axis == 0:
      crop_x = gif_x
      crop_y = gif_y // 3 + 120
      x = 0
      y = center_y - (crop_y // 2)
      try:
        gif_mod = ffmpeg.crop(gif_input, x, y, crop_x, crop_y)
      except Exception as e:
        print(e)
  elif func == 'scale':
    gif_mod = gif_input

  out, _ = gif_mod.filter('scale', 108, 36).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
  video = np.frombuffer(out, np.uint8).reshape([-1, 36, 108, 3])

  # NOTE without crop
  # out, _ = ffmpeg.input('./imports/' + image).filter('scale', 108, 36).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
  # video = np.frombuffer(out, np.uint8).reshape([-1, 36, 108, 3])
    

  for i, frame in enumerate(video):
    np.save(f"../matrix_data/eff-{effect_name}/eff-{effect_name}{str(i).zfill(3)}", frame, allow_pickle=False)
    save_eff_as_image(frame, effect_name + str(i).zfill(3) + ".png")
  print(f"Frames: {i}")

if __name__ == "__main__":
  np.set_printoptions(threshold=sys.maxsize)

  # Good test - needs v_crop
  convert_gif_to_np('hearts1.gif')