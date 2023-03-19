import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", "-a", type=str, dest="name", required=True)
parser.add_argument("--description", "-n", type=str, dest="is_description", default="nono", required=False)
parser.add_argument("--private", "-p", dest="is_private", #action="store_true", 
                    required=False, type=str , default="false" )
args = parser.parse_args()

repo_name = args.name
is_private = args.is_private
description = args.is_description

### load github_token
## load func to connect to postgress database
from urllib.request import urlopen
target_url="https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py"
exec(urlopen(target_url).read().decode('utf-8'))
## load func to sql query token table 
target_url="https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/get_github_key.py"
exec(urlopen(target_url).read().decode('utf-8'))
## get GITHUB_TOKEN in postgress database
GITHUB_TOKEN = func_github_key()


API_URL = "https://api.github.com"
paylod = '{"name": "%s", "private":  %s , "description": "%s"}' %(repo_name, is_private, description)
headers = {
    "Authorization": "token " + GITHUB_TOKEN,
    "Accept": "application/vnd.github.v3+json",
}

try :
    r= requests.post(API_URL + "/user/repos", data = paylod, headers = headers)
    r.raise_for_status()
except Exception as error :
    raise SystemExit(error)

# exemple run :
# python create_repos_on_gith.py --name tools_private --private true --description "begin tools_private"
# python create_repos_on_gith.py --name tools_public --private false --description "begin tools_public"


