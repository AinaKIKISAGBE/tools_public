

# import https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py

def func_github_key():
	conn, cursor = create_connection()

	cursor.execute("SELECT keys_value FROM tb1keys WHERE keys_source='github_key1'")
	#cursor.fetchall()
	github_key = cursor.fetchone()[0]
	return github_key
	
