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
    ("Juwel Rio 125", 80, 125, 200),
    ("Juwel Rio 180", 100, 180, 550),
    ("Juwel Rio 240", 120, 240, 550),
    ("Juwel Rio 350", 120, 350, 800),
    ("Juwel Rio 450", 250, 450, 800),
    ("Eheim clearscape 73", 60, 73, 200),
    ("Eheim clearscape 175", 71, 175, 300),
    ("Eheim clearscape 200", 90, 200, 200),
    ("Eheim clearscape 300", 120, 300, 200),
    ("Eheim proxima 175", 70, 175, 200),
    ("Eheim proxima 250", 100, 250, 200),
    ("Eheim proxima 325", 130, 325, 200),
    ("Eheim proxima scape", 71, 175, 200),
    ("Eheim vivalineLED 126", 80, 126, 200),
    ("Eheim vivalineLED 150", 60, 150, 200),
    ("Eheim vivalineLED 180", 100, 180, 200),
    ("Clear Garden Mini M", 100, 180, 200),
    ("Clear Garden 45P", 36, 20, 70),
    ("Clear Garden 60P", 60, 64, 130),
    ("Clear Garden 90P", 90, 182, 479),
    ("Clear Garden 120P", 120, 240, 550),
    ("Dennerle Scapers Tank 35", 40, 35, 70),
    ("Dennerle Scapers Tank 55", 45, 55, 90),
    ("Dennerle Scapers Tank 70", 50, 70, 120)
]

# Daten in die Tabelle einfügen
c.executemany('INSERT INTO Aquarium (name, liter, length, price) VALUES (?, ?, ?, ?)', data)

# Daten, die eingefügt werden sollen
data = [
    ('Neonsalmler', 'Paracheirodon innesi', 'Südamerika', 20, 27, 5.0, 7.5, 2, 20, 0, 8, 54, 9999),
    ('Schwertträger Marygold', 'Xiphophorus helleri', 'Südamerika', 22, 28, 7.0, 8.5, 10, 30, 5, 20, 54, 9999),
    ('Molly schwarz', 'Poecilia sphenops', 'Südamerika', 24, 28, 7.0, 8.5, 12, 30, 5, 20, 80, 9999),
    ('Platy', 'Xiphophorus maculatus', 'Südamerika', 18, 28, 7.0, 8.5, 5, 25, 5, 20, 54, 9999),
    ('Leuchtaugenbärbling', 'Rasbora dorsiocellata macrophthalma', 'Asien', 20, 28, 5.5, 7.5, 5, 10, 1, 6, 60, 9999),
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
