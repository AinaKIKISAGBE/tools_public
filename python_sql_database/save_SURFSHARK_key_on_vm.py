


## load func to connect to postgress database
from urllib.request import urlopen
target_url="https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py"
exec(urlopen(target_url).read().decode('utf-8'))


import json 

def func_SURFSHARK_key():
	conn, cursor = create_connection()
	cursor.execute("SELECT keys_value FROM tb1keys WHERE keys_source='SURFSHARK_key1'")
	#cursor.fetchall()
	SURFSHARK_key = cursor.fetchone()[0]
	return SURFSHARK_key

SURFSHARK_TOKEN = func_SURFSHARK_key()
SURFSHARK_TOKEN_dict = json.loads(SURFSHARK_TOKEN)

# ['SURFSHARK_USER', 'SURFSHARK_PASSWORD', 'SURFSHARK_COUNTRY', 'SURFSHARK_CITY', 'OPENVPN_OPTS', 'CONNECTION_TYPE', 'LAN_NETWORK', 'CREATE_TUN_DEVICE', 'OVPN_CONFIGS', 'ENABLE_KILL_SWITCH']
for key in SURFSHARK_TOKEN_dict.keys(): 
    with open('/SURFSHARK_TOKEN/{}'.format(key), 'w') as f:
        f.write(str(SURFSHARK_TOKEN_dict[key]))
    
