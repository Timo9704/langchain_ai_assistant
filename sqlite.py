import sqlite3

# Verbindung zur SQLite-Datenbank herstellen (erstellt die DB, falls sie nicht existiert)
conn = sqlite3.connect('ai_planner/app.db')
c = conn.cursor()

# Tabelle "Aquarium" erstellen
c.execute('''CREATE TABLE IF NOT EXISTS aquarium (
                name TEXT,
                length INTEGER,
                depth INTEGER,
                height INTEGER,
                liter INTEGER,
                glasstype TEXT,
                price INTEGER
             )''')

# Daten, die eingefügt werden sollen
data = [
    ("Juwel Rio 125", 80, 36, 50, 125, "Floatglas", 360),
    ("Juwel Rio 180", 100, 41, 50, 180, "Floatglas", 399),
    ("Juwel Rio 240", 120, 41, 55, 240, "Floatglas", 554),
    ("Juwel Rio 350", 120, 50, 66, 350, "Floatglas", 770),
    ("Juwel Rio 450", 150, 50, 66, 450, "Floatglas", 980),
    ("Eheim clearscape 73", 60, 35, 35, 73, "Weißglas", 449),
    ("Eheim clearscape 175", 71, 35, 35, 175, "Weißglas", 630),
    ("Eheim clearscape 200", 90, 50, 45, 200, "Weißglas", 689),
    ("Eheim clearscape 300", 120, 50, 50, 300, "Weißglas", 999),
    ("Eheim proxima 175", 70, 50, 57, 175, "Floatglas", 569),
    ("Eheim proxima 250", 100, 50, 57, 250, "Floatglas", 929),
    ("Eheim proxima 325", 130, 57, 57, 325, "Floatglas", 999),
    ("Eheim proxima scape", 71, 50, 50, 175, "Weißglas", 470),
    ("Eheim vivalineLED 126", 80, 35, 45, 126, "Weißglas", 380),
    ("Eheim vivalineLED 150", 60, 50, 50, 150, "Weißglas", 423),
    ("Eheim vivalineLED 180", 100, 40, 45, 180, "Weißglas", 470),
    ("Clear Garden Mini M", 36, 22, 26, 180, "Weißglas", 70),
    ("Clear Garden 45P", 45, 30, 30, 40, "Weißglas", 90),
    ("Clear Garden 60P", 60, 30, 36, 64, "Weißglas", 129),
    ("Clear Garden 90P", 90, 45, 45, 182, "Weißglas", 479),
    ("Clear Garden 120P", 120, 45, 45, 240, "Weißglas", 550),
    ("Dennerle Scapers Tank 35", 40, 32, 28, 35, "Weißglas", 70),
    ("Dennerle Scapers Tank 55", 45, 36, 34, 55, "Weißglas", 90),
    ("Dennerle Scapers Tank 70", 50, 39, 36, 70, "Weißglas", 120)
]

c.executemany('INSERT INTO aquarium (name, length, depth, height, liter, glasstype, price) VALUES (?, ?, ?, ?, ?, ?, ?)', data)

# Tabelle "Fish" erstellen
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

data = [
    ('Neonsalmler', 'Paracheirodon innesi', 'Südamerika', 20, 27, 5.0, 7.5, 2, 20, 0, 8, 54, 9999),
    ('Schwertträger Marygold', 'Xiphophorus helleri', 'Südamerika', 22, 28, 7.0, 8.5, 10, 30, 5, 20, 54, 9999),
    ('Molly schwarz', 'Poecilia sphenops', 'Südamerika', 24, 28, 7.0, 8.5, 12, 30, 5, 20, 80, 9999),
    ('Platy', 'Xiphophorus maculatus', 'Südamerika', 18, 28, 7.0, 8.5, 5, 25, 5, 20, 54, 9999),
    ('Leuchtaugenbärbling', 'Rasbora dorsiocellata macrophthalma', 'Asien', 20, 28, 5.5, 7.5, 5, 10, 1, 6, 60, 9999),
    ('Diskus', 'Symphysodon discus', 'Südamerika', 29, 30, 5.0, 7.0, 1, 10, 1, 2, 300, 9999),
    ('Kupfersalmler', 'Hasemania nana', 'Südamerika', 22, 27, 6.0, 8.0, 1, 20, 1, 8, 80, 9999),
    ('Funkensalmler', 'Hyphessobrycon amandae', 'Südamerika', 20, 28, 5.0, 6.5, 1, 5, 1, 5, 54, 9999),
    ('Guppy', 'Poecilia Reticulata', 'Südamerika', 20, 28, 7, 8, 5, 10, 5, 20, 54, 9999)
]

c.executemany('''
INSERT INTO fish (common_name, latin_name, origin, min_temperature, max_temperature, min_pH, max_pH, min_GH, max_GH, min_KH, max_KH, min_liters, max_liters) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', data)


c.execute('''CREATE TABLE IF NOT EXISTS plants (
    name TEXT,
    type TEXT,
    growth_rate TEXT,
    light_demand TEXT,
    co2_demand TEXT
)''')

data = [
    ('Hygrophila corymbosa Siamensis 53B', 'Hintergrund', 'mittel', 'niedrig', 'niedrig'),
    ('Hygrophila pinnatifida', 'Hintergrund', 'mittel', 'mittel', 'mittel'),
    ('Rotala rotundifolia', 'Hintergrund', 'mittel', 'niedrig', 'niedrig'),
    ('Rotala wallichii', 'Hintergrund', 'mittel', 'hoch', 'hoch'),
    ('Vallisneria spiralis Tiger', 'Hintergrund', 'hoch', 'niedrig', 'niedrig'),
    ('Pogostemon stellatus', 'Hintergrund', 'mittel', 'hoch', 'hoch'),
    ('Hygrophila costata', 'Hintergrund', 'hoch', 'mittel', 'niedrig'),
    ('Cryptocoryne beckettii Petchii', 'Mittelgrund', 'mittel', 'niedrig', 'niedrig'),
    ('Cryptocoryne wendtii Green', 'Mittelgrund', 'mittel', 'niedrig', 'niedrig'),
    ('Cryptocoryne wendtii Tropica', 'Mittelgrund', 'mittel', 'niedrig', 'niedrig'),
    ('Cryptocoryne x willisii', 'Mittelgrund', 'langsam', 'niedrig', 'niedrig'),
    ('Hottonia palustris', 'Mittelgrund', 'langsam', 'niedrig', 'niedrig'),
    ('Eleocharis acicularis', 'Vordergrund', 'mittel', 'niedrig', 'niedrig'),
    ('Glossostigma elatinoides', 'Vordergrund', 'hoch', 'hoch', 'hoch'),
    ('Helanthium tenellum', 'Vordergrund', 'langsam', 'niedrig', 'niedrig'),
    ('Hydrocotyle tripartita', 'Vordergrund', 'hoch', 'mittel', 'niedrig'),
    ('Lilaeopsis brasiliensis', 'Vordergrund', 'langsam', 'mittel', 'mittel'),
    ('Marsilea hirsuta', 'Vordergrund', 'mittel', 'niedrig', 'niedrig'),
    ('Micranthemum callitrichoides', 'Vordergrund', 'mittel', 'hoch', 'hoch'),

]

c.executemany('INSERT INTO plants (name, type, growth_rate, light_demand, co2_demand) VALUES (?, ?, ?, ?, ?)', data)

conn.commit()
conn.close()

print("Daten erfolgreich in die SQLite-Datenbank eingefügt.")
