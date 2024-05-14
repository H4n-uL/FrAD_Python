import hashlib, os, requests, sys

def fetch_git(url, dir_path):
    res = requests.get(url, params={'ref': 'main'})

    if res.status_code != 200:
        print(f'STATUS CODE: {res.status_code}, Failed to update FrAD')
        print(f'{res.json()['message']}')
        sys.exit(1)

    for content in res.json():
        if content['type'] == 'dir':
            new_dir_path = os.path.join(dir_path, content['name'])
            os.makedirs(new_dir_path, exist_ok=True)
            fetch_git(content['url'], new_dir_path)
        else:
            try: data = open(os.path.join(dir_path, content['name']), 'rb').read()
            except: data = b''
            sha = hashlib.sha1((f'blob {len(data)}\x00').encode() + data).hexdigest()
            if content['sha'] != sha:
                print(f'Updating {content['name']} from {sha[:8]}... to {content['sha'][:8]}...')
                open(os.path.join(dir_path, content['name']), 'wb').write(requests.get(content['download_url']).content)
