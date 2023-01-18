"""Make a zip file for submission."""

import os
import zipfile

SRC_EXT = '.py'
PYNB_EXT = '.ipynb'
JSON_EXT = '.json'

def make_zip():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create zip file
    zip_path = os.path.join(script_dir, 'submission.zip')
    print('Creating {}'.format(zip_path))
    with zipfile.ZipFile(zip_path, 'w') as zip_fh:
        for base_path, dir_names, file_names in os.walk('.'):
            for file_name in file_names:
                if file_name.endswith(SRC_EXT) \
                        or file_name.endswith(PYNB_EXT) \
                        or file_name.endswith(JSON_EXT):
                    # Read file
                    file_path = os.path.join(base_path, file_name)
                    rel_path = os.path.relpath(file_path, script_dir)
                    print('Writing {} to {}'.format(rel_path, zip_path))
                    zip_fh.write(file_path, rel_path)


if __name__ == '__main__':
    make_zip()
