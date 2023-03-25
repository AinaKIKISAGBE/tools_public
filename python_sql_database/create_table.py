# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:10:20 2023

@author: ksami
"""

######## python sql : create table ip jobs and one master

## load func to connect to postgress database
from urllib.request import urlopen
target_url="https://raw.githubusercontent.com/AinaKIKISAGBE/tools_public/main/python_sql_database/connect_to_postgresql.py"
exec(urlopen(target_url).read().decode('utf-8'))

def create_table():
	try:
		# Get the cursor object from the connection object
		conn, curr = create_connection()
		try:
			# Fire the CREATE query
			curr.execute("CREATE TABLE IF NOT EXISTS \
			iptable(ipID INTEGER, ip TEXT , category TEXT, available boolean, dateUpdate timestamp, jobRunFile BYTEA, jobRunFormat TEXT, jobRunFormulaire TEXT)")
			#      ( 1 2 3 4 5 6, 10.0.2.1, worker master, true false    , 2403/2023 06:19:54 , bookingxxx.py   ,  .py             , Formulaire User)")
			
		except(Exception, psycopg2.Error) as error:
			# Print exception
			print("Error while creating cartoon table", error)
		finally:
			# Close the connection object
			conn.commit()
			conn.close()
	finally:
		# Since we do not have to do anything here we will pass
		pass
create_table()
		
#conn, cursor = create_connection()
#cursor.execute("SELECT keys_value FROM tb1keys WHERE keys_source='github_key1'")
#cursor.fetchall()
 