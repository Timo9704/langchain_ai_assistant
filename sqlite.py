import sqlite3

# Verbindung zur SQLite-Datenbank herstellen (erstellt die DB, falls sie nicht existiert)
conn = sqlite3.connect('ai_planner/aquarium.db')
c = conn.cursor()

# Tabelle "Aquarium" erstellen
c.execute('''CREATE TABLE IF NOT EXISTS Aquarium (
                name TEXT,
                length INTEGER,
                liter INTEGER,
                price INTEGER
             )''')

# Tabelle "Aquarium" erstellen
c.execute('''CREATE TABLE fish (
  common_name TEXT,
  latin_name TEXT,
  origin TEXT,
  min_temperature INTEGER,
  max_temperature INTEGER,
  min_pH REAL,
  max_pH REAL,
  min_GH INTEGER,
  max_GH INTEGER,
  min_KH INTEGER,
  max_KH INTEGER,
  min_liters INTEGER,
  max_liters INTEGER
)''')


# Daten, die eingefügt werden sollen
data = [
    ("Juwel Rio 125", 120, 80, 200),
    ("Juwel Rio 250", 120, 250, 550),
    ("Juwel Rio 450", 250, 450, 800),
    ("Eheim clearscape 73", 60, 73, 200),
    ("Eheim clearscape 175", 73, 175, 300)
]

# Daten in die Tabelle einfügen
c.executemany('INSERT INTO Aquarium (name, liter, length, price) VALUES (?, ?, ?, ?)', data)

# Daten, die eingefügt werden sollen
data = [
    ('Neonsalmler', 'Paracheirodon innesi', 'Südamerika', 20, 27, 5.0, 7.5, 2, 20, 0, 8, 54, 9999),
    ('Schwertträger Marygold', 'Xiphophorus helleri', 'Südamerika', 22, 28, 7.0, 8.5, 10, 30, 5, 20, 54, 9999),
    ('Molly schwarz', 'Poecilia sphenops', 'Südamerika', 24, 28, 7.0, 8.5, 12, 30, 5, 20, 80, 9999),
    ('Platy', 'Xiphophorus maculatus', 'Südamerika', 18, 28, 7.0, 8.5, 5, 25, 5, 20, 54, 9999),
    ('Diskus', 'Symphysodon discus', 'Südamerika', 29, 30, 5.0, 7.0, 1, 10, 1, 2, 300, 9999)
]

# Daten in die Tabelle einfügen
c.executemany('''
INSERT INTO fish (common_name, latin_name, origin, min_temperature, max_temperature, min_pH, max_pH, min_GH, max_GH, min_KH, max_KH, min_liters, max_liters) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', data)

# Änderungen speichern und Verbindung schließen
conn.commit()
conn.close()

print("Daten erfolgreich in die SQLite-Datenbank eingefügt.")
