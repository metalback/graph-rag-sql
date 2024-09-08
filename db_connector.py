import sqlite3
import os
import pandas as pd

class DatabaseConnector:
  def __init__(self):
      self.databases = {
          'patients': 'cache/patients.db',
          'encounters': 'cache/encounters.db',
          'labwork': 'cache/labwork.db'
      }
      self.cache_dir = 'cache'
      # Create cache directory if it doesn't exist
      if not os.path.exists(self.cache_dir):
        os.makedirs(self.cache_dir)
        # Create sample databases
        self.create_sample_databases()
      
  def connect_and_cache(self):
      """
      Connect to all databases, check if they're cached, and cache if necessary.
      """
      for db_name, db_file in self.databases.items():
          conn = sqlite3.connect(db_file)
          cursor = conn.cursor()
          
          # Get all tables in the database
          cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
          tables = cursor.fetchall()
          
          for table in tables:
              table_name = table[0]
              cache_file = os.path.join(self.cache_dir, f"{db_name}_{table_name}.json")
              
              if not os.path.exists(cache_file):
                  # Cache the first 200 rows
                  df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 200", conn)
                  df.to_json(cache_file, index=False)
                  print(f"Cached {cache_file}")
          
          conn.close()

  def create_sample_databases(self):
      """
      Create sample SQLite databases with synthetic data for demonstration purposes.
      """
      # patients database
      conn_patients = sqlite3.connect('cache/patients.db')
      c_patients = conn_patients.cursor()
      c_patients.execute('''CREATE TABLE IF NOT EXISTS patients
                   (id INTEGER PRIMARY KEY, name TEXT, mrn TEXT)''')
      for i in range(1000):
          c_patients.execute("INSERT INTO patients VALUES (?, ?, ?)",
                              (i, f"patient {i}", f"MRN000{i}"))
      conn_patients.commit()
      conn_patients.close()

      # encounters database
      conn_encounters = sqlite3.connect('cache/encounters.db')
      c_encounters = conn_encounters.cursor()
      c_encounters.execute('''CREATE TABLE IF NOT EXISTS encounters
                   (id INTEGER PRIMARY KEY, patient_id INTEGER, clinic_id INTEGER, time INTEGER)''')
      for i in range(5000):
          c_encounters.execute("INSERT INTO encounters VALUES (?, ?, ?, ?)",
                           (i, i, i % 100, (i % 10) + 1))
      conn_encounters.commit()
      conn_encounters.close()

      # labwork database
      conn_labwork = sqlite3.connect('cache/labwork.db')
      c_labwork = conn_labwork.cursor()
      c_labwork.execute('''CREATE TABLE IF NOT EXISTS labwork
                   (id INTEGER PRIMARY KEY, mrn TEXT, result REAL)''')
      for i in range(100):
          c_labwork.execute("INSERT INTO labwork VALUES (?, ?, ?)",
                             (i, f"MRN000{i}", (i % 100) + 0.99))
      conn_labwork.commit()
      conn_labwork.close()

      print("Sample databases created successfully.")