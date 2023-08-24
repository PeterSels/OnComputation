import requests
import json

from GitHubGistInfo import HEADERS, GIST_ID


def download_remote_github_gist_to_local_file(gist_id: str, save_to: str):
    """
    Download the content of a specified Gist into a local file.
    
    Parameters:
    - gist_id: The ID of the Gist to download.
    - save_to: The name of the local file to save the Gist content.
    
    Returns:
    - None
    """
    response = requests.get(f'https://api.github.com/gists/{gist_id}', headers=HEADERS)
    
    if response.status_code == 200:
        gist_data = response.json()
        gist_content = next(iter(gist_data['files'].values()))['content']
        
        with open(save_to, 'w') as file:
            file.write(gist_content)
    else:
        raise ValueError(f"Error {response.status_code}: Unable to download Gist.")

download_remote_github_gist_to_local_file(GIST_ID, 'BeachBiz26.py.gist')


