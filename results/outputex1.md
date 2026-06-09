Device utilise : cuda
GPU : NVIDIA GeForce GTX 1650 Ti
Nombre d'exemples : 20640
Nombre de features : 8
Noms des features : ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

Cinq premieres lignes :
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25        3.413
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25        3.422

Statistiques descriptives :
             MedInc      HouseAge      AveRooms     AveBedrms  ...      AveOccup      Latitude     Longitude   MedHouseVal
count  20640.000000  20640.000000  20640.000000  20640.000000  ...  20640.000000  20640.000000  20640.000000  20640.000000
mean       3.870671     28.639486      5.429000      1.096675  ...      3.070655     35.631861   -119.569704      2.068558
std        1.899822     12.585558      2.474173      0.473911  ...     10.386050      2.135952      2.003532      1.153956
min        0.499900      1.000000      0.846154      0.333333  ...      0.692308     32.540000   -124.350000      0.149990
25%        2.563400     18.000000      4.440716      1.006079  ...      2.429741     33.930000   -121.800000      1.196000
50%        3.534800     29.000000      5.229129      1.048780  ...      2.818116     34.260000   -118.490000      1.797000
75%        4.743250     37.000000      6.052381      1.099526  ...      3.282261     37.710000   -118.010000      2.647250
max       15.000100     52.000000    141.909091     34.066667  ...   1243.333333     41.950000   -114.310000      5.000010

[8 rows x 9 columns]

Commentaire Q2 :
La cible est asymetrique a droite et plafonnee a 5.0, ce qui cree un effet de censure.
Le boxplot montre des valeurs elevees proches du plafond artificiel.

Commentaire Q3 :
Feature la plus correlee a la cible : MedInc (0.688)
Paire de features la plus colineaire : Latitude / Longitude

Verification DataLoader :
X batch : torch.Size([64, 8]) | y batch : torch.Size([64, 1])
X mean : -0.0247 | X std : 0.8201
Val batch size : torch.Size([256, 8])
Test batch size : torch.Size([256, 8])

Commentaire Q5 :
Le scaler est fitte seulement sur train pour eviter le data leakage vers validation/test.
PS C:\Users\HP\Desktop\deeplearning> 

