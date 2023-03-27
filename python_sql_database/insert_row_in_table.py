

# import https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py


### load 2capcha
## load func to connect to postgress database
from urllib.request import urlopen
target_url="https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py"
exec(urlopen(target_url).read().decode('utf-8'))

## load func to sql query token_2capcha table 
def func_2capcha_key():
	conn, cursor = create_connection()
	cursor.execute("SELECT keys_value FROM tb1keys WHERE keys_source='2capcha_key1'")
	#cursor.fetchall()
	key_2capcha = cursor.fetchone()[0]
	return key_2capcha

## get TOKEN_2capcha in postgress database
#TOKEN_2capcha = func_2capcha_key()

def insert_row():
    conn, cursor = create_connection()
    ## nb line 
    cursor.execute("SELECT MAX(id) FROM tb1keys")
    maxipID = cursor.fetchone()[0]
    new_maxipID = int(str(maxipID)) + 1
    
    conn, cursor = create_connection()
    code_sql = "INSERT INTO tb1keys (id, keys_source, keys_value, keys_url) VALUES({new_maxipID},'2capcha_key1','380f27f8cc4e39807913fc6494f743cf','https://2captcha.com/');".format(new_maxipID=new_maxipID)   
    cursor.execute(code_sql)
    conn.commit() 
    conn.close()
