Device utilise : cuda
GPU : NVIDIA GeForce GTX 1650 Ti
Total configurations Grid Search : 288
Grid Search : 48 configurations x 80 epochs max
------------------------------------------------------------
  [  1/48] val_MSE=0.3268 (21.9s)
  [  2/48] val_MSE=0.3142 (29.0s)
  [  3/48] val_MSE=0.3185 (42.7s)
  [  4/48] val_MSE=0.3068 (44.6s)
  [  5/48] val_MSE=0.3348 (47.1s)
  [  6/48] val_MSE=0.3280 (46.8s)
  [  7/48] val_MSE=0.3397 (55.6s)
  [  8/48] val_MSE=0.3440 (49.7s)
  [  9/48] val_MSE=0.3181 (33.8s)
  [ 10/48] val_MSE=0.3060 (35.8s)
  [ 11/48] val_MSE=0.3131 (44.0s)
  [ 12/48] val_MSE=0.3081 (43.6s)
  [ 13/48] val_MSE=0.3397 (49.9s)
  [ 14/48] val_MSE=0.3477 (32.6s)
  [ 15/48] val_MSE=0.3480 (44.1s)
  [ 16/48] val_MSE=0.3451 (43.4s)
  [ 17/48] val_MSE=0.4167 (18.3s)
  [ 18/48] val_MSE=0.3990 (20.8s)
  [ 19/48] val_MSE=0.4167 (21.6s)
  [ 20/48] val_MSE=0.4242 (21.8s)
  [ 21/48] val_MSE=0.3012 (70.6s)
  [ 22/48] val_MSE=0.3062 (52.2s)
  [ 23/48] val_MSE=0.2981 (81.5s)
  [ 24/48] val_MSE=0.2895 (81.5s)
  [ 25/48] val_MSE=0.4154 (18.2s)
  [ 26/48] val_MSE=0.4341 (16.7s)
  [ 27/48] val_MSE=0.4265 (21.0s)
  [ 28/48] val_MSE=0.4039 (21.3s)
  [ 29/48] val_MSE=0.3112 (62.7s)
  [ 30/48] val_MSE=0.3072 (49.2s)
  [ 31/48] val_MSE=0.2986 (97.9s)
  [ 32/48] val_MSE=0.2883 (85.8s)
  [ 33/48] val_MSE=0.3882 (18.8s)
  [ 34/48] val_MSE=0.4122 (22.5s)
  [ 35/48] val_MSE=0.3803 (20.0s)
  [ 36/48] val_MSE=0.3746 (20.1s)
  [ 37/48] val_MSE=0.3757 (32.2s)
  [ 38/48] val_MSE=0.4029 (23.7s)
  [ 39/48] val_MSE=0.4062 (46.2s)
  [ 40/48] val_MSE=0.4055 (49.8s)
  [ 41/48] val_MSE=0.3854 (19.9s)
  [ 42/48] val_MSE=0.3859 (19.9s)
  [ 43/48] val_MSE=0.3924 (19.6s)
  [ 44/48] val_MSE=0.4090 (20.0s)
  [ 45/48] val_MSE=0.3544 (33.0s)
  [ 46/48] val_MSE=0.3765 (36.0s)
  [ 47/48] val_MSE=0.3992 (46.0s)
  [ 48/48] val_MSE=0.4264 (34.5s)

=== TOP 10 configurations Grid Search ===
  hidden_dims activation  dropout_rate     lr  weight_decay  val_mse
[128, 64, 32] leaky_relu           0.3 0.0005        0.0010 0.288338
[128, 64, 32]       relu           0.3 0.0005        0.0010 0.289547
[128, 64, 32]       relu           0.3 0.0005        0.0001 0.298148
[128, 64, 32] leaky_relu           0.3 0.0005        0.0001 0.298649
[128, 64, 32]       relu           0.3 0.0010        0.0001 0.301156
     [64, 32] leaky_relu           0.1 0.0010        0.0010 0.305982
[128, 64, 32]       relu           0.3 0.0010        0.0010 0.306186
     [64, 32]       relu           0.1 0.0005        0.0010 0.306754
[128, 64, 32] leaky_relu           0.3 0.0010        0.0010 0.307178
     [64, 32] leaky_relu           0.1 0.0005        0.0010 0.308103

Impact individuel estime :
hyperparametre  std_des_moyennes_val_mse
   hidden_dims                  0.032392
  dropout_rate                  0.020680
            lr                  0.002215
  weight_decay                  0.001190
    activation                  0.000897
    clip_value                       NaN