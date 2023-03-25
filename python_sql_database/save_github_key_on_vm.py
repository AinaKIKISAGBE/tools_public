


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


with open('/GITHUB_TOKEN', 'w') as f:
    f.write(str(GITHUB_TOKEN))
	
