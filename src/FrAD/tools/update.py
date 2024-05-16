import hashlib, os, requests, sys

def fetch_git(url, dir, dref='/src', ref='main'):
    res = requests.get(url, params={'ref': ref})

    if res.status_code != 200:
        print(f'STATUS CODE: {res.status_code}, Failed to update FrAD')
        print(f'{res.json()['message']}')
        sys.exit(1)

    for content in res.json():
        newref = os.path.join(dref, content['name'])
        if content['type'] == 'dir':
            newdir = os.path.join(dir, content['name'])
            os.makedirs(newdir, exist_ok=True)
            return fetch_git(content['url'], newdir, dref=newref)
        else:
            try:
                data = open(os.path.join(dir, content['name']), 'rb').read()
                sha = hashlib.sha1((f'blob {len(data)}\x00').encode() + data).hexdigest()
            except: sha = None
            if content['sha'] != sha:
                print(f'Updating {newref} from {sha is not None and f"{sha[:8]}..." or "null"} to {content['sha'][:8]}...')
                open(os.path.join(dir, content['name']), 'wb').write(requests.get(content['download_url']).content)
                return True
            return False
