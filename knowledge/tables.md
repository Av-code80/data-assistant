# Documentation des tables

## Table users (clients)
Une ligne par client inscrit sur le site.
Colonnes : `id` (identifiant), `gender` (`F` ou `M`), `age` (âge en années, de 18 à 75),
`country` (pays de livraison), `city` (ville), `traffic_source` (canal d'acquisition :
`Search`, `Organic`, `Email`, `Facebook`, `Display`), `created_at` (date d'inscription).
Les tranches d'âge (18-24, 25-34, 35-44, 45-54, 55+) se calculent à partir de `age`.

## Table products (produits)
Une ligne par produit du catalogue.
Colonnes : `id`, `name`, `category` (catégorie : `Jeans`, `Sweaters`, `Outerwear & Coats`,
`Dresses`, `Tops & Tees`, `Shoes`, `Accessories`, `Swim`, `Active`), `brand` (marque),
`department` (`Women` ou `Men`), `cost` (coût d'achat en euros), `retail_price` (prix
catalogue en euros TTC).
Les noms de catégories sont en anglais : « pulls » correspond à `Sweaters`, « manteaux » à
`Outerwear & Coats`, « robes » à `Dresses`, « chaussures » à `Shoes`, « maillots de bain »
à `Swim`, « sport » à `Active`.

## Table orders (commandes)
Une ligne par commande.
Colonnes : `order_id`, `user_id` (client), `status` (statut de la commande), `created_at`
(date de commande), `shipped_at` (expédition), `delivered_at` (livraison), `returned_at`
(retour), `num_of_item` (nombre d'articles).
La date de référence d'une commande pour toutes les analyses est `created_at`.

## Table order_items (lignes de commande)
Une ligne par article commandé : c'est la table de référence pour le chiffre d'affaires.
Colonnes : `id`, `order_id`, `user_id`, `product_id`, `status` (statut de l'article),
`created_at` (date de commande), `returned_at` (date de retour de l'article), `sale_price`
(prix payé en euros TTC, remises incluses).
Le statut d'un article peut différer de celui de sa commande : un article seul peut être
retourné dans une commande `Complete`.
