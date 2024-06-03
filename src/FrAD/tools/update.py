import hashlib, os, requests, sys

def terminal(*args: object, sep: str | None = ' ', end: str | None = '\n'):
    sys.stderr.buffer.write(str(sep.join(map(str,args))+end).encode())
    sys.stderr.buffer.flush()

def getsha1(file):
    sha = hashlib.sha1()
    sha.update(f'blob {os.path.getsize(file)}\x00'.encode())
    with open(file, 'rb') as f:
        while True:
            data = f.read(2**30)
            if not data: break
            sha.update(data)
    return sha.hexdigest()

def fetch_git(dir, gitdir='src', ref='main'):
    url = os.path.join('https://api.github.com/repos/h4n-ul/Fourier_Analogue-in-Digital/contents', gitdir)
    res = requests.get(url, params={'ref': ref})

    if res.status_code != 200:
        terminal(f'STATUS CODE: {res.status_code}, Failed to update FrAD')
        terminal(f'{res.json()['message']}')
        sys.exit(1)

    for content in res.json():
        file = os.path.join(gitdir, content['name'])
        if content['type'] == 'dir':
            newdir = os.path.join(dir, content['name'])
            os.makedirs(newdir, exist_ok=True)
            if content['name'] != 'src': fetch_git(newdir, file)
        else:
            try:    sha = getsha1(os.path.join(dir, content['name']))
            except: sha = None
            if content['sha'] != sha:
                terminal(f'Updating {file} from {sha is not None and f"{sha[:8]}..." or "null"} to {content['sha'][:8]}...')
                open(os.path.join(dir, content['name']), 'wb').write(requests.get(content['download_url']).content)
