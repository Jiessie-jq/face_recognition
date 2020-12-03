configs = {
    # ------------ Basic Configuration ------------
    "batch_size": 5,
    "input_size": [112, 112],
    # ------------ Training Configuration ------------
    "learning_rate": 0.1 / 8,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    # ------------ IO Configuration ------------
    "base_dir": "/OCR/face_recognition/model_save/v0.1",
    "dataset_dir": "/OCR/face_recognition/full_data.csv",
    "log_interval": 200,
    # ------------ Dataset Configuration ------------
    "dataset": "dummy",
    "num_class": 7,
    "learning_rate_milestons": [3],
    "learning_rate_gamma": 0.1,
    "num_epoch": 10,
    # ------------ Model Configuration ------------
    "use_stn": False,
    "backbone": "resnet18",
    "output_head": "bn_dropout_gap_fc_bn",
    "feature_dim": 512,
    # ------------ Loss Configuration ------------
    # loss function: margined_logit = s * (cos(m1 * theta + m2) - m3)
    # m1 != 1.0, m2 == 0.0, m3 == 0.0 is used in SphereFace, which is not implemented in this codebase
    # m1 == 1.0, m2 != 0.0, m3 == 0.0 is used in ArcFace
    # m1 == 1.0, m2 == 0.0, m3 != 0.0 is used in CosFace
    # other combinations of (m1, m2, m3) are also welcomed.
    # "loss_type": "cosface",
    # "loss_scale": 30,
    # "loss_m1": 1,
    # "loss_m2": 0,
    # "loss_m3": 0.35,
    "loss_type": "arcface",
    "loss_scale": 30,
    "loss_m1": 1,
    "loss_m2": 0.5,
    "loss_m3": 0,
}
