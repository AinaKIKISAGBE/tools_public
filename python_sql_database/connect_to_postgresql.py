
import psycopg2

def create_connection():
	# Connect to the database
	# using the psycopg2 adapter.
	# Pass your database name ,# username , password ,
	### keep pc_category
	pc_category=$(cat < /pc_id/pc_category)
	
	if pc_category = "master" : # master
		ip_db = "192.168.1.47"
	elif pc_category = "worker_docker": # worker docker
		ip_db = "10.0.0.5"
	elif pc_category = "worker": # worker VM
		ip_db = "192.168.1.47"	
		
	# hostname and port number
	conn = psycopg2.connect(dbname='keysdb',
							user='postgres',
							password='postgres',
							host= ip_db , # '192.168.1.164',
							port='5432')
	# Get the cursor object from the connection object
	curr = conn.cursor()
	
	
	
	
	return conn, curr



