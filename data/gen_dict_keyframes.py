import os
import json
from pathlib import Path
import sys

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

def main():
    keyframes_id_save = open(os.path.join(ROOT, 'dicts/keyframes_id_search.json'), 'w')
    folder_keyframes = os.path.join(WORK_DIR, "data", "keyframes")
    list_image_paths = []

    for keyframes in sorted(os.listdir(folder_keyframes)):
        if 'Keyframes' not in keyframes:
            continue  # Process all relevant directories

        print(keyframes)
        keyframes_path = os.path.join(folder_keyframes, keyframes)
        
        if os.path.isdir(keyframes_path):
            sorted_LOs = sorted(os.listdir(keyframes_path))
            for LO in sorted_LOs:
                LO_path = os.path.join(keyframes_path, LO)
                if os.path.isdir(LO_path):
                    sorted_image_paths = sorted(os.listdir(LO_path))
                    for image_path in sorted_image_paths:
                        if os.path.splitext(image_path)[1].lower() == '.jpg':
                            full_path = os.path.join(LO_path, image_path)
                            # Generate relative path from `folder_keyframes` and prepend `keyframes/`
                            relative_path = os.path.relpath(full_path, start=folder_keyframes)
                            formatted_path = os.path.join('keyframes', relative_path).replace(os.path.sep, '/')
                            list_image_paths.append(formatted_path)

    json.dump(list_image_paths, keyframes_id_save, indent=6)
    keyframes_id_save.close()

if __name__ == '__main__':
    main()
