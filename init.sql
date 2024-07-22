CREATE DATABASE aquabott;
\connect aquabott

CREATE TABLE public.aquarium (
    name VARCHAR(255),
    liter INT,
    laenge INT
);

INSERT INTO public.aquarium (name, liter, laenge) VALUES ('Juwel Rio 125', 120, 80);