# Règles et limites des données

## Statuts des commandes et des articles
Valeurs possibles : `Processing` (en préparation), `Shipped` (expédiée), `Complete`
(livrée), `Cancelled` (annulée), `Returned` (retournée).
Les commandes annulées sont exclues de toutes les métriques. Les articles retournés sont
exclus du chiffre d'affaires et des articles vendus, mais comptent dans le taux de retour.

## Devise, taxes et remises
Tous les montants sont en euros TTC, y compris pour la Suisse. Il n'y a pas de données de
marge nette, de frais de livraison ni de TVA détaillée. Des remises de 10 à 30 % s'appliquent
pendant les soldes de janvier et de juillet : `sale_price` est alors inférieur au prix
catalogue.

## Calendrier et périodes
Les trimestres sont civils : T1 janvier-mars, T2 avril-juin, T3 juillet-septembre, T4
octobre-décembre. L'exercice fiscal correspond à l'année civile. Les semaines commencent le
lundi (norme ISO). « Le mois dernier » désigne le mois civil complet précédent.

## Fraîcheur et historique des données
L'historique commence le 1er janvier 2024. Les données sont à jour jusqu'au 24 octobre 2026
inclus : une période qui inclut des dates postérieures est incomplète.
Le pic d'activité annuel a lieu en novembre et décembre.

## Périmètre géographique
ModaNova vend dans six pays : France, Allemagne (`Germany`), Espagne (`Spain`), Italie
(`Italy`), Belgique (`Belgium`) et Suisse (`Switzerland`). Les noms de pays sont stockés en
anglais. La France représente environ 40 % des clients.

## Données non disponibles
L'entreprise ne dispose pas dans cette base : des stocks et ruptures, des dépenses
marketing (coût des campagnes publicitaires Facebook, Google ou e-mail, budget publicité,
coût d'acquisition), du trafic du site et des taux de conversion, des avis clients, des
données des concurrents, ni de prévisions. Une question sur ces sujets ne peut pas recevoir
de réponse chiffrée. Le canal d'acquisition d'un client (`traffic_source`) est connu, mais
pas ce qu'il a coûté.
