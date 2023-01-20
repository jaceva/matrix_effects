import sys
import math
import time
import numpy as np
from PIL import Image

# RGB effects have a single foreground RGB value super imposed over the effect
# Static effects have RGB values built in and can only be dimmed

def create_default_array(value=1, dt=np.uint8):
  '''Create a numpy array of proper dimensions'''
  np_array = np.ones((36, 108, 3), dtype=dt)
  np_array[:,:,:] *= value

  return np_array

def create_level_array(value=1):
  '''Abstracts the effect vs color'''
  np_array = create_default_array(value=value, dt=np.half)
  return np_array

def add_foreground_color(np_array, fg_color):
  np_array[:,:,0][np_array[:,:,0] > 0] *= fg_color["green"]
  np_array[:,:,1][np_array[:,:,1] > 0] *= fg_color["red"]
  np_array[:,:,2][np_array[:,:,2] > 0] *= fg_color["blue"]

  return np_array

def add_background_color(np_array, bg_color):
  np_array[:,:,0][np_array[:,:,0] < 0] *= -bg_color["green"]
  np_array[:,:,1][np_array[:,:,1] < 0] *= -bg_color["red"]
  np_array[:,:,2][np_array[:,:,2] < 0] *= -bg_color["blue"]

  return np_array

def save_array_as_image(np_array, filename, 
                        fg_color={"red":0, "green":0, "blue":0}, 
                        bg_color={"red":0, "green":0, "blue":0}):
  # TODO saving as np.uint8 is necessary for this function
  # Is this the case for sending data to the pico
  '''Save a numpy array as an image'''
  
  np_array = add_foreground_color(np_array, fg_color)
  np_array = add_background_color(np_array, bg_color)
  np_array = color_convert(np_array)
  image = Image.fromarray(np_array.astype(np.uint8))
  image.save(filename, format="png")

def color_convert(np_array):
  '''Data sent to wall is GRB, image data is RGB'''
  convert_array = create_default_array(dt=np.short)
  convert_array[:,:,0] = np_array[:,:,1]
  convert_array[:,:,1] = np_array[:,:,0]
  convert_array[:,:,2] = np_array[:,:,2]

  return convert_array

def set_box_to_value(np_array, value, top_left=(0,0), bottom_right=(0,0), fill=True):
  '''Set a box of coords to a value'''
  # 0=green, 1=red, 2=blue
  if fill:
    np_array[top_left[1]:bottom_right[1]+1,top_left[0]:bottom_right[0]+1,0] = value
    np_array[top_left[1]:bottom_right[1]+1,top_left[0]:bottom_right[0]+1,1] = value
    np_array[top_left[1]:bottom_right[1]+1,top_left[0]:bottom_right[0]+1,2] = value
  else:
    # top line
    np_array[top_left[1],top_left[0]:bottom_right[0]+1,0] = value
    np_array[top_left[1],top_left[0]:bottom_right[0]+1,1] = value
    np_array[top_left[1],top_left[0]:bottom_right[0]+1,2] = value

    # bottom line
    np_array[bottom_right[1],top_left[0]:bottom_right[0]+1,0] = value
    np_array[bottom_right[1],top_left[0]:bottom_right[0]+1,1] = value
    np_array[bottom_right[1],top_left[0]:bottom_right[0]+1,2] = value

    # left side
    np_array[top_left[1]+1:bottom_right[1],top_left[0],0] = value
    np_array[top_left[1]+1:bottom_right[1],top_left[0],1] = value
    np_array[top_left[1]+1:bottom_right[1],top_left[0],2] = value

    # right side
    np_array[top_left[1]+1:bottom_right[1],bottom_right[0],0] = value
    np_array[top_left[1]+1:bottom_right[1],bottom_right[0],1] = value
    np_array[top_left[1]+1:bottom_right[1],bottom_right[0],2] = value

  return np_array

def set_line_to_value(np_array, value, top_left=(0,0), bottom_right=(0,0)):
  '''Set a line to a value'''
  delta_x = bottom_right[0] - top_left[0]
  delta_y = bottom_right[1] - top_left[1]
  print(delta_x, delta_y)
  if delta_x == 0 or delta_y ==0:
    set_box_to_value(np_array, value, top_left=top_left, bottom_right=bottom_right)
  elif delta_x == delta_y:
    x, y = top_left
    while x < bottom_right[0]:
      np_array[y,x,0] = value
      np_array[y,x,1] = value
      np_array[y,x,2] = value
      x+=1
      y+=1
  elif delta_x > delta_y:
    n_sections = delta_y + 1
    displacement = delta_x + 1
    section_lengths = [math.floor(displacement/n_sections), math.ceil(displacement/n_sections),]
    x, y = top_left
    for i in range(n_sections):
      dx = x + section_lengths[i % 2]
      set_box_to_value(np_array, value, top_left=(x, y), bottom_right=(dx-1, y))
      x, y = dx, y + 1

  else: #delta_x < delta_y
    n_sections = delta_x + 1
    displacement = delta_y + 1
    section_lengths = [math.floor(displacement/n_sections), math.ceil(displacement/n_sections),]
    x, y = top_left
    for i in range(n_sections):
      dy = y + section_lengths[i % 2]
      set_box_to_value(np_array, value, top_left=(x, y), bottom_right=(x, dy-1))
      x, y = x + 1, dy

  return np_array

def square_vortex(vortex_depth=1):
  np_array = create_level_array(value=-1)
  level=1
  step = vortex_depth / 18
  for i in range(0, 18, 2):
    np_array = set_box_to_value(np_array, level, (i,i), (107-i,35-i), fill=False)
    level -= step*2
    print(level)
  
  return np_array

if __name__ == "__main__":
  np.set_printoptions(threshold=sys.maxsize)
  fg_color = {"red":0, "green":0, "blue":255}
  bg_color = {"red":0, "green":0, "blue":0}
  
  for i in range(100):
    depth = i / 100
    color_array = square_vortex(vortex_depth=depth)
    np.save(f"rgb-square-vortex/rgb-square-vortex{str(i).zfill(3)}", color_array)
    save_array_as_image(color_array, f"test{str(i).zfill(3)}.png", fg_color=fg_color, bg_color=bg_color)
  # default_array = create_default_array(dt=np.short)

  # step_level = 0
  # step = 0.01
  # while True:
  #   print(step_level, step)
  #   color_array = square_vortex(default_array, color, step_level=step_level)
  #   step_level += step
  #   if step_level == 1 or step_level == 0:
      
  #     step *= -1

  # for i in range(0, 18, 2):
  #   print(color)
  #   color_array = set_box_to_value(default_array, (i,i), (107-i,35-i), color, fill=False)
  #   color["blue"] =  color["blue"] - 31
  # color_array = set_box_to_value(default_array, (0,0), (107,35), color, fill=False)
  # color_array = set_line_to_value(default_array, (0,0), (100,20), color)
    