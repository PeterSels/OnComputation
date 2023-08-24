import requests
import json
from GitHubGistInfo import HEADERS, GIST_ID


def update_remote_github_gist_from_local_file(gist_id: str, filename: str) -> str:
    """
    Update the content of a specified Gist with the content of a given file.
    
    Parameters:
    - gist_id: The ID of the Gist to update.
    - filename: The name of the file containing updated content.
    
    Returns:
    - The URL of the updated Gist, or an error message.
    """
    with open(filename, 'r') as file:
        content = file.read()
    
    data = {
        'files': {
            filename: {
                'content': content
            }
        }
    }

    response = requests.patch(f'https://api.github.com/gists/{gist_id}', headers=HEADERS, data=json.dumps(data))

    if response.status_code == 200:
        return response.json().get('html_url')
    else:
        return f"Error {response.status_code}: Unable to update Gist."

update_remote_github_gist_from_local_file(GIST_ID, 'BeachBiz26.py')  # update remote gist from local file


