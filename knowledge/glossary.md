# Glossaire métier ModaNova

## Chiffre d'affaires (revenue)
Synonymes : CA, chiffre d'affaires, ventes, revenus, recettes, GMV, montant vendu.
Le chiffre d'affaires est la somme des prix de vente (`order_items.sale_price`) des articles
dont le statut n'est ni `Cancelled` ni `Returned`. Il s'exprime en euros TTC.
Convention ModaNova : quand un utilisateur parle de « ventes » sans préciser, il s'agit du
chiffre d'affaires. Quand il demande « ce qui se vend le plus » ou « le plus vendu » sans
parler d'articles, de pièces ou d'unités, on classe aussi par chiffre d'affaires.

## Nombre de commandes (order_count)
Synonymes : commandes, nombre de commandes, volume de commandes, paniers validés.
Nombre de commandes distinctes (`orders.order_id`) dont le statut n'est pas `Cancelled`.
Une commande retournée compte comme une commande passée.

## Panier moyen (average_order_value)
Synonymes : panier moyen, valeur moyenne de commande, AOV, montant moyen par commande.
Chiffre d'affaires divisé par le nombre de commandes, sur le même périmètre et la même période.

## Articles vendus (units_sold)
Synonymes : articles vendus, unités, pièces vendues, volume, quantités.
Nombre de lignes de `order_items` dont le statut n'est ni `Cancelled` ni `Returned`.
Chaque ligne correspond à une pièce. À utiliser quand l'utilisateur parle explicitement
d'articles, de pièces ou d'unités.

## Clients (customer_count)
Synonymes : clients, acheteurs, nombre de clients, clientèle.
Nombre de clients distincts (`orders.user_id`) ayant passé au moins une commande non annulée
sur la période. Un client inscrit qui n'a jamais commandé n'est pas compté.

## Taux de retour (return_rate)
Synonymes : taux de retour, retours, articles retournés, articles renvoyés, remboursements,
produits rendus.
Nombre d'articles au statut `Returned` divisé par le nombre d'articles non annulés, en
pourcentage. Les chaussures et les robes ont les taux de retour les plus élevés.
