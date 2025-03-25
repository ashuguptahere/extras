from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="/home/qulith-jr/Desktop/QL/datasets/dataset-rfdetr",
    epochs=500,
    batch_size=16,
    grad_accum_steps=4,
    lr=1e-4,
)
