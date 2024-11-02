# MVTec-Reconstruct - KLA Project Submission

This is a submission for KLA's Project statement. This project aims to develop a DL algorithm to restore degraded images without obscuring defects of interest. The performance of the algorithm is benchmarked using a validation dataset, with metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM). Iterative refinement is conducted based on evaluation results to improve image clarity and defect detection accuracy.

## Dataset Overview
The MVTec Anomaly Detection (AD) dataset is widely used for benchmarking image anomaly detection tasks. For this project, we introduced noise and blur to degrade the images in the dataset, and our model aims to use these degraded images as input for restoration.

## Our Approach
We employed the SADNet model, a state-of-the-art denoising model, to reconstruct the degraded images. Since preserving defective regions is a key requirement, we used a weighted combination of MSELoss and L1Loss to ensure that more emphasis is exerted on regions masked as defective. Specifically, hyperparameters α (alpha) and β (beta) were introduced to create weights for each pixel such that more importance is given to defective regions (α + β * mask).

![SADNet Model Block Diagram](https://github.com/sujay-2001/MVTec-Reconstruct/blob/assets/image.png)

Below is a brief overview of the process flow used in the model:
- **Input**: Degraded images from the MVTec AD dataset.
- **Restoration Network**: SADNet model reconstructs the degraded images.
- **Loss Function**: Combination of weighted MSELoss, weighted L1Loss and PSNRLoss to emphasize defective regions.
- **Hyperparameters**: α and β are adjusted to apply the appropriate weight to defective regions, ensuring that defects are not obscured.

## Evaluation Metrics
The performance of the model was evaluated using two key metrics: PSNR and SSIM. These metrics are computed for both the overall image and the defective regions to gauge restoration quality and defect detection accuracy.

### Average Metrics Over All Items
| Category          | PSNR       | SSIM       |
|-------------------|------------|------------|
| Overall           | 30.01      | 0.821      |
| Defective Regions | 43.53      | 0.989      |

### Average Metrics for Each Item
| Item        | PSNR (Overall) | PSNR (Defective Regions) | SSIM (Overall) | SSIM (Defective Regions) |
|-------------|----------------|--------------------------|----------------|--------------------------|
| bottle      | 31.08          | 38.52                    | 0.884          | 0.986                    |
| capsule     | 35.65          | 53.51                    | 0.908          | 0.999                    |
| hazelnut    | 32.21          | 38.02                    | 0.903          | 0.983                    |
| pill        | 35.76          | 48.23                    | 0.926          | 0.983                    |
| grid        | 28.08          | 44.47                    | 0.850          | 0.997                    |
| screw       | 33.27          | 50.51                    | 0.924          | 0.999                    |
| carpet      | 22.31          | 39.50                    | 0.591          | 0.988                    |
| cable       | 27.82          | 42.39                    | 0.873          | 0.993                    |
| metal_nut   | 29.61          | 41.68                    | 0.810          | 0.988                    |
| leather     | 27.04          | 46.85                    | 0.659          | 0.996                    |
| wood        | 26.42          | 40.29                    | 0.646          | 0.991                    |
| zipper      | 28.56          | 41.39                    | 0.720          | 0.992                    |
| tile        | 25.05          | 36.01                    | 0.676          | 0.967                    |
| transistor  | 29.77          | 40.88                    | 0.909          | 0.980                    |
| toothbrush  | 30.57          | 43.07                    | 0.900          | 0.994                    |

## Results and Iterative Refinement
Based on the evaluation results, we iteratively adjusted the hyperparameters and loss weights to improve the model's ability to restore degraded images while preserving the clarity of defective regions. The weighted loss function proved effective in ensuring that the reconstructed images retained defect details, leading to more accurate anomaly detection.


## Get Started 

### Install required modules
Install the required modules to run the model using:

```bash
pip install -r requirements.txt
```
### Update `config.yaml`
Update the `config.yaml` with the training/evaluation parameters to either fine-tune the model or evaluate the model.

### Evaluation
After updating the `config.yaml` with params: `data_path`, `model_checkpoint`, `interpolate`, `input_size`, `batch_size` `output_save_path`, you can run the following bash command to evaluate the model on the validation dataset:
```bash
python3 eval.py
```

### Training
Similarly you can update the `config.yaml` with desired params, you can run the following bash command to train the model on the training dataset:
```bash
python3 train.py
```

## Model weights
You can download the model weights from the following links:
| Input Size          | Link       | 
|-------------------|------------|
| 512x512 | [Download](https://drive.google.com/file/d/1Fw89b_zFvieYskDVl7DL9KEeDTw9zza4/view?usp=sharing)   |
| 640x640 | [Download](https://drive.google.com/file/d/1zBuHLehW0v1hvsCnEM8ndchPYw7fWPY4/view?usp=sharing)      |
| 800x800 | [Download](https://drive.google.com/file/d/1X3vwF9FE8houLr6OpR0zt30H6Wxh1ny1/view?usp=sharing)       |


## Repository Organization

Our repository organization is shown below:

```
MVTec-Reconstruct
|── assets
├── dcn
├── model
│   ├── sadnet.py
│   ├── ...
├── eval.py
├── train.py
├── losses.py
├── metrics.py
├── utils.py
├── config.yaml
├── requirements.txt
├── README.md

```

## References
```
@article{chang2020spatial,
  title={Spatial-Adaptive Network for Single Image Denoising},
  author={Chang, Meng and Li, Qi and Feng, Huajun and Xu, Zhihai},
  journal={arXiv preprint arXiv:2001.10291},
  year={2020}
}
```
## Acknowledgements
The SADNet model in our code adopts from [JimmyChame's implementation](https://github.com/JimmyChame/SADNet).