import os
import pandas as pd
import psycopg2
import sql
import yaml
import sys
from io import StringIO
from sqlalchemy import create_engine


### the db_df_loader initiates a connection to the selected database and allows the user to load a dataframe 
class db_df_loader:
    def __init__(self, db):
        self.db = db   

        with open(r'../config/secrets.yaml') as file:
            secrets = yaml.load(file, Loader=yaml.FullLoader)
        
        self.user = secrets['pg_user']
        self.pwd = secrets['pg_pwd']

        """ Connect to the PostgreSQL database server """
        conn = None
        try:
            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect("dbname={} user={} password={}".format(self.db, self.user, self.pwd))
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            sys.exit(1) 
        print("Connection successful")
        self.conn = conn
    
    def pd_to_db(self, df, table):
        """
        Here we are going save the dataframe in memory 
        and use copy_from() to copy it to the table
        """
        # save dataframe to an in memory buffer
        buffer = StringIO()
        df.to_csv(buffer, index_label='id', header=False)
        buffer.seek(0)

        cursor = self.conn.cursor()

        try:
            engine = create_engine('postgresql://{}:{}@localhost:5432/{}'.format(self.user, self.pwd, self.db))
            df.to_sql(table, engine, if_exists = 'replace')
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            cursor.close()
            return 1
        print("copy_from_stringio() done")
        cursor.close()
    def query_df(self, query):
        cursor = self.conn.cursor()

        try:
            engine = create_engine('postgresql://{}:{}@localhost:5432/{}'.format(self.user, self.pwd, self.db))
            df = pd.read_sql_query(query,con=engine)
            return df
        
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            cursor.close()
            return 1
        
        cursor.close()
        
        
    def conn_close():
        self.conn.close()