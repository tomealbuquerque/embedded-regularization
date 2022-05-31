# Embedded Regularization for classification of colposcopic images

https://ieeexplore.ieee.org/document/9433871

by Tomé Albuquerque and Jaime S. Cardoso

## Abstract
Cervical cancer ranks as the fourth most common cancer among females worldwide with roughly 528,000 new cases yearly. Significant progress in the realm of artificial intelligence particularly in neural networks and deep learning help physicians to diagnose cervical cancer more accurately. In this paper, we address a classification problem with the widely used VGG16 architecture. In addition to classification error, our model considers a regularization part during tuning of the weights, acting as prior knowledge of the colposcopic image. This embedded regularization approach, using a 2D Gaussian kernel, has enabled the model to learn which sections of the medical images are more crucial for the classification task. The experimental results show an improvement compared with standard transfer learning and multimodal approaches of cervical cancer classification in literature.

<img src="https://github.com/tomealbuquerque/embedded-regularization/blob/main/figures/fig1.PNG" width="400">

## Results
<img src="https://github.com/tomealbuquerque/embedded-regularization/blob/main/figures/tab1.PNG" width="500">

## Citation
If you find this work useful for your research, please cite our paper:
```
@INPROCEEDINGS{9433871,
  author={Albuquerque, Tomé and Cardoso, Jaime S.},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Embedded Regularization For Classification Of Colposcopic Images}, 
  year={2021},
  volume={},
  number={},
  pages={1920-1923},
  doi={10.1109/ISBI48211.2021.9433871}}
```

If you have any questions about our work, please do not hesitate to contact <tome.albuquerque@gmail.com>
