import os

TOKEN = os.environ.get('GITHUB_PETER_SELS_GIST_TOKEN')
if not TOKEN:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

GIST_ID = '3050485675280486e5c800009e37fb06' # Peter Sels can see it here:
# https://gist.github.com/PeterSels/3050485675280486e5c800009e37fb06
