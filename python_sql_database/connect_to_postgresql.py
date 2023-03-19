
import psycopg2

def create_connection():
	# Connect to the database
	# using the psycopg2 adapter.
	# Pass your database name ,# username , password ,
	# hostname and port number
	conn = psycopg2.connect(dbname='keysdb',
							user='postgres',
							password='postgres',
							host='192.168.1.164',
							port='5432')
	# Get the cursor object from the connection object
	curr = conn.cursor()
	return conn, curr



