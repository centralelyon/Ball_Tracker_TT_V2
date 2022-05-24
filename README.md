
<h1>Ball_Tracker_TT_V2 : Outil de tracking de tennis de table</h1>

Cet outil est la deuxième version d'un tracker pour tennis de table, il a pour objectif de donner <b>les coordonées d'une balle de tennis de table</b> à partir d'une vidéo. 

<h2>Dataset utilisé</h2>

Pour entrainer ce modèle j'ai utilisé le match Flore vs Bolle que nous avions à disposition, celui-ci n'étant pas annoté pour ce genre de tâche il a fallu annoté des images pour se constituer un jeu de données suffisant (ici <b>6000 images annotées sur 59000</b>)

<h2>Preprocessing</h2>

Pour entrainer notre algorithme on utilise 3 types d'images différentes (data augmentation par distorsion de couleurs) : La première image est en RGB (normal), en deuxième on utilise le Background Suppression d'OpenCV, enfin on utilise Canny et on enlève tout pixel d'une couleur inférieur à 220 (valeur seuil).

Les images sont séparées en tiles : en effet l'image de base est de 1920x1080 pixels ce qui n'est pas possible de calculer (nombre énorme de neurones), pour ne pas perdre d'informations et pour se concentrer sur les frames importantes on divise la zone de jeux en petite image (256x144) et on entraine un réseau de neurones sur chacune de ces tiles. On fusionne ensuite les résultats en regardant quelle prédiction est la plus sûre (intérêt du fonctionnement en 2 réseaux de neurones).

<h2>Architecture de l'algorithme</h2>

L'algorithme se décompose en 2 réseaux de neurones : Le premier permet de détecter si une image contient une balle ou non (vision globale), il s'agit alors d'un problème de classification.
Le deuxième réseau de neurones permet de trouver les coordonées de la balle dans l'image (vision locale), il s'agit alors d'un problème de regresssion.
