## import_image.py
Create a more customizable import feature to be used with a flask server

#### Design Additions
- Config File
  - Effect directory location/name
    - git directory - prefer not a sub directory - for now
  - Thumbs location/name
    - Used by the web app
  - Import location/name
    - Can we save file to RPi or should all imports be url?
- Directory Management
  - Add directory if missing
- File Management
- Path constants
- Ability to change save file location
- separate all ffmpeg functions
  - crop
  - scale
  - offset
  - flip
  - fill borders
- import from
  - file
  - url
- Support image thumb preview
