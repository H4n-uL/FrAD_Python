import hashlib, os, requests, sys

def terminal(*args: object, sep: str | None = ' ', end: str | None = '\n', flush: False = False):
    sys.stderr.write(sep.join(args)+end)
    if flush: sys.stderr.flush()

def getsha1(file):
    sha = hashlib.sha1()
    sha.update(f'blob {os.path.getsize(file)}\x00'.encode())
    with open(file, 'rb') as f:
        while True:
            data = f.read(2**30)
            if not data: break
            sha.update(data)
    return sha.hexdigest()

def fetch_git(url, dir, dref='/src', ref='main'):
    res = requests.get(url, params={'ref': ref})

    if res.status_code != 200:
        terminal(f'STATUS CODE: {res.status_code}, Failed to update FrAD')
        terminal(f'{res.json()['message']}')
        sys.exit(1)

    for content in res.json():
        newref = os.path.join(dref, content['name'])
        if content['type'] == 'dir':
            newdir = os.path.join(dir, content['name'])
            os.makedirs(newdir, exist_ok=True)
            fetch_git(content['url'], newdir, dref=newref)
        else:
            try:    sha = getsha1(os.path.join(dir, content['name']))
            except: sha = None
            if content['sha'] != sha:
                terminal(f'Updating {newref} from {sha is not None and f"{sha[:8]}..." or "null"} to {content['sha'][:8]}...')
                open(os.path.join(dir, content['name']), 'wb').write(requests.get(content['download_url']).content)
