      (linear): Linear(in_features=8, out_features=128, bias=True)
      (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (1): ModuleDict(
      (linear): Linear(in_features=128, out_features=64, bias=True)
      (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (2): ModuleDict(
      (linear): Linear(in_features=64, out_features=32, bias=True)
      (bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
    )
  )
  (output_layer): Linear(in_features=32, out_features=1, bias=True)
)
Parametres entrainables : 11,969
Sortie : torch.Size([64, 1])
Plage : [-2.752, 4.216]
Ep  20 | tr_MSE=0.3943 | val_MSE=0.3323 | R2=0.7507 | lr=0.001000
Ep  40 | tr_MSE=0.3506 | val_MSE=0.7262 | R2=0.4550 | lr=0.001000
Ep  60 | tr_MSE=0.3466 | val_MSE=0.2966 | R2=0.7774 | lr=0.000500
  Early stopping a l'epoch 76
Baseline best val MSE : 0.2790
Temps entrainement : 95.9 s
Modele sauvegarde : C:\Users\HP\Desktop\deeplearning\models\baseline_best.pth
PS C:\Users\HP\Desktop\deeplearning> 