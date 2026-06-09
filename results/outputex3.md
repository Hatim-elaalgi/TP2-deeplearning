PS C:\Users\HP\Desktop\deeplearning> py -X utf8 ex3_ablation.py
>> 
Device utilise : cuda
GPU : NVIDIA GeForce GTX 1650 Ti
A_baseline         | val_MSE=0.3796 | temps=29.9s
B_sans_bn          | val_MSE=0.2596 | temps=97.3s
C_sans_dropout     | val_MSE=0.3297 | temps=33.6s
D_sans_l2          | val_MSE=0.3555 | temps=50.3s
E_aucune_reg       | val_MSE=0.2687 | temps=48.1s
sans_clipping      | val_MSE=0.3397 | temps=70.2s
clip_0.1           | val_MSE=0.3402 | temps=31.7s
clip_0.5           | val_MSE=0.3685 | temps=26.4s
clip_1.0           | val_MSE=0.3796 | temps=34.9s
clip_5.0           | val_MSE=0.3039 | temps=68.3s
clip_10.0          | val_MSE=0.3161 | temps=58.5s
raw_sans_clipping  | val_MSE=0.3770 | temps=71.5s
raw_clip_1         | val_MSE=0.4228 | temps=55.3s

Commentaire : BatchNorm stabilise souvent le plus l'apprentissage.
Dropout et L2 reduisent le surapprentissage, surtout lorsque les courbes train/val divergent.
Un clip_value autour de 1 ou 5 est generalement un bon compromis sur donnees standardisees.
PS C:\Users\HP\Desktop\deeplearning> 
