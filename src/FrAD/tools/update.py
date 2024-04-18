import requests
import os, sys

def fetch_git(url, dir_path, download_ffmpeg_portables=False):
    res = requests.get(url, params={'ref': 'main'})

    if res.status_code != 200:
        sys.stderr.write(f'STATUS CODE: {res.status_code}, Failed to update FrAD\n')
        sys.stderr.write(f'{res.json()['message']}\n')
        return

    for content in res.json():
        if content['type'] == 'dir':
            new_dir_path = os.path.join(dir_path, content['name'])
            if content['name'] == 'res' and not download_ffmpeg_portables: continue
            os.makedirs(new_dir_path, exist_ok=True)
            fetch_git(content['url'], new_dir_path)
        else: open(os.path.join(dir_path, content['name']), 'wb').write(requests.get(content['download_url']).content)
