import sys
import math
import time
import numpy as np
from PIL import Image
from random import shuffle, randint

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

# def pulse_2by2(np_array, top_left, step=0.1):
def pulse_2by2(step=0.1):
  '''2 pixel square that pulses up/down once with `step` at a given x, y coordinate'''
  value = step
  # set_box_to_value(np_array, value, 
                    # top_left=(top_left[0], top_left[1]), 
                    # bottom_right=(top_left[0]+1, top_left[1]+1))
  while value + step > 0:
    yield round(value, 2)
    value += step
    # set_box_to_value(np_array, value,  
    #                   top_left=(top_left[0], top_left[1]), 
    #                 bottom_right=(top_left[0]+1, top_left[1]+1))
    # print(value)
    if value + step > 1:
      step *= -0.5


def pulse_mini_square_4by4(top_left, overlap=0):
  pulses = []
  for j in range(4):
    for i in range(4):
      # pulses.append(pulse_2by2(np_array, top_left=(top_left[0]+i*2, top_left[1]+j*2)))
      pulses.append(((i, j), pulse_2by2()))
  
  for pulse in pulses:
    pulse_coords = tuple(map(sum, zip(top_left, pulse[0])))
    for value in pulse[1]:
      yield (pulse_coords, value)
  
  return False


def pulse_mini_square_frame(fg_color, bg_color, overlap=0):
  square_coords = [(x, y) for x in range(0, 108, 4) for y in range(0, 36, 4)]
  shuffle(square_coords)
  for i in range(8):
    square_coords.append(square_coords[i])
  # print(square_coords)
  frame = 0
  # active_squares = [pulse_mini_square_4by4(np_array, top_left=square_coords.pop(0))]
  active_squares = [pulse_mini_square_4by4(top_left=square_coords.pop(0))]
  # while len(square_coords) > 0:
  while True:
    pulse_data = []
    for square in active_squares:
      try:
        pulse_data.append(next(square))
      except StopIteration:
        pass
    if len(pulse_data) == 0:
      break
    # print(pulse_data)

    np_array = create_level_array(value=-1)
    for pulse in pulse_data:
      pulse_coords = pulse[0]
      pulse_value = pulse[1]
      set_box_to_value(np_array, pulse_value,  
                      top_left=pulse_coords, 
                    bottom_right=(pulse_coords[0]+1, pulse_coords[1]+1))
    np.save(f"rgb-mini-square-chase/rgb-mini-square-chase{str(frame).zfill(3)}", np_array)
    save_array_as_image(np_array, f"mini-square-chase/test{str(frame).zfill(3)}.png", fg_color=fg_color, bg_color=bg_color)
    frame += 1
    if frame % 10 == 0 and len(square_coords) > 0:
      print(f"add square at frame: {frame}")
      active_squares.append(pulse_mini_square_4by4(top_left=square_coords.pop(0)))


  


if __name__ == "__main__":
  np.set_printoptions(threshold=sys.maxsize)
  fg_color = {"red":255, "green":255, "blue":255}
  bg_color = {"red":0, "green":0, "blue":0}
  pulse_mini_square_frame(fg_color, bg_color)
  # np_array = create_level_array(value=-1)
  # mini_square_generator = pulse_mini_square_4by4(np_array, top_left=(0, 0))
  # np.save(f"rgb-pulse-mini-square/rgb-pulse-mini-square{str(0).zfill(3)}", np_array)
  # save_array_as_image(np_array, f"pulse-mini-square/test{str(0).zfill(3)}.png", fg_color=fg_color, bg_color=bg_color)
  # for i, _ in enumerate(mini_square_generator):
  #   np.save(f"rgb-pulse-mini-square/rgb-pulse-mini-square{str(i).zfill(3)}", np_array)
  #   save_array_as_image(np_array, f"pulse-mini-square/test{str(i).zfill(3)}.png", fg_color=fg_color, bg_color=bg_color)


  # for i in range(100):
  #   depth = i / 100
  #   color_array = square_vortex(vortex_depth=depth)
  #   np.save(f"rgb-square-vortex/rgb-square-vortex{str(i).zfill(3)}", color_array)
  #   save_array_as_image(color_array, f"test{str(i).zfill(3)}.png", fg_color=fg_color, bg_color=bg_color)
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
    