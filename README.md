# Introduction
Thank you for becoming a sponsor, your support makes my work on the satellite-image-deep-learning repository and newsletter possible 🙏

This repository is divided into three main parts:

- Datasets
- Software for working with remote sensing data
- Model deployment

# Datasets
This section contains a short list of datasets relevant to deep learning, particularly those which come up regularly in the literature. **Warning** satellite image files can be LARGE, and even a small datasets may comprise 50GB+ of imagery

## Lists of datasets
<!-- markdown-link-check-disable -->
* [Earth Observation Database](https://eod-grss-ieee.com/)
<!-- markdown-link-check-enable -->
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)
* [Awesome_Satellite_Benchmark_Datasets](https://github.com/Seyed-Ali-Ahmadi/Awesome_Satellite_Benchmark_Datasets)
* [Callisto-Dataset-Collection](https://github.com/Agri-Hub/Callisto-Dataset-Collection) -> datasets that use Copernicus/sentinel data

## Sentinel
As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions)
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel) -> a curated list of awesome tools, tutorials and APIs related to data from the Copernicus Sentinel Satellites.
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) and [Sentinel-2 L2A 120m Mosaic](https://registry.opendata.aws/sentinel-s2-l2a-mosaic-120/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F)
* Paid access to Sentinel & Landsat data via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py)
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels. Also available [in torchgeo](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#bigearthnet)
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* [Analyzing Sentinel-2 satellite data in Python with Keras](https://github.com/jensleitloff/CNN-Sentinel)
* [Xarray backend to Copernicus Sentinel-1 satellite data products](https://github.com/bopen/xarray-sentinel)
* [SEN2VENµS](https://zenodo.org/record/6514159#.YoRxM5PMK3I) -> a dataset for the training of Sentinel-2 super-resolution algorithms
* [SEN12MS](https://github.com/zhu-xlab/SEN12MS) -> A Curated Dataset of Georeferenced Multi-spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion. Checkout [SEN12MS toolbox](https://github.com/schmitt-muc/SEN12MS) and many referenced uses on [paperswithcode.com](https://paperswithcode.com/dataset/sen12ms)
* [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) -> A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning, with [website](https://www.sen4agrinet.space.noa.gr/) and [models](https://github.com/Orion-AI-Lab/S4A-Models)
* [earthspy](https://github.com/AdrienWehrle/earthspy) -> Monitor and study any place on Earth and in Near Real-Time (NRT) using the Sentinel Hub services developed by the EO research team at Sinergise
* [Space2Ground](https://github.com/Agri-Hub/Space2Ground) -> dataset with Space (Sentinel-1/2) and Ground (street-level images) components, annotated with crop-type labels for agriculture monitoring.
* [sentinel2tools](https://github.com/QuantuMobileSoftware/sentinel2tools) -> downloading & basic processing of Sentinel 2 imagesry. Read [Sentinel2tools: simple lib for downloading Sentinel-2 satellite images](https://medium.com/geekculture/sentinel2tools-simple-lib-for-downloading-sentinel-2-satellite-images-f8a6be3ee894)
* [open-sentinel-map](https://github.com/VisionSystemsInc/open-sentinel-map) -> The OpenSentinelMap dataset contains Sentinel-2 imagery and per-pixel semantic label masks derived from OpenStreetMap
* [MSCDUnet](https://github.com/Lihy256/MSCDUnet) -> change detection datasets containing VHR, multispectral (Sentinel-2) and SAR (Sentinel-1)
* [OMBRIA](https://github.com/geodrak/OMBRIA) -> Sentinel-1 & 2 dataset for adressing the flood mapping problem
* [Canadian-cropland-dataset](https://github.com/bioinfoUQAM/Canadian-cropland-dataset) -> a novel patch-based dataset compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2
* [Sentinel-2 Cloud Cover Segmentation Dataset](https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1) on Radiant mlhub
* [The Azavea Cloud Dataset](https://www.azavea.com/blog/2021/08/02/the-azavea-cloud-dataset/) which is used to train this [cloud-model](https://github.com/azavea/cloud-model)
* [fMoW-Sentinel](https://purl.stanford.edu/vg497cb6002) -> The Functional Map of the World - Sentinel-2 corresponding images (fMoW-Sentinel) dataset consists of image time series collected by the Sentinel-2 satellite, corresponding to locations from the Functional Map of the World (fMoW) dataset across several different times. Used in [SatMAE](https://github.com/sustainlab-group/SatMAE)
* [Earth Surface Water Dataset](https://zenodo.org/record/5205674#.Y4iEFezP1hE) -> a dataset for deep learning of surface water features on Sentinel-2 satellite images. See [this ref using it in torchgeo](https://towardsdatascience.com/artificial-intelligence-for-geospatial-analysis-with-pytorchs-torchgeo-part-1-52d17e409f09)
* [Ship-S2-AIS dataset](https://zenodo.org/record/7229756#.Y5GsgOzP1hE) -> 13k tiles extracted from 29 free Sentinel-2 products. 2k images showing ships in Denmark sovereign waters: one may detect cargos, fishing, or container ships

## Landsat
Long running US program -> see [Wikipedia](https://en.wikipedia.org/wiki/Landsat_program)
* 8 bands, 15 to 60 meters, 185km swath, the temporal resolution is 16 days
* [Landsat 4, 5, 7, and 8 imagery on Google](https://cloud.google.com/storage/docs/public-datasets/landsat), see [the GCP bucket here](https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/), with Landsat 8 imagery in COG format analysed in [this notebook](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb)
* [Landsat 8 imagery on AWS](https://registry.opendata.aws/landsat-8/), with many tutorials and tools listed
* https://github.com/kylebarron/landsat-mosaic-latest -> Auto-updating cloudless Landsat 8 mosaic from AWS SNS notifications
* [Visualise landsat imagery using Datashader](https://examples.pyviz.org/landsat/landsat.html#landsat-gallery-landsat)
* [Landsat-mosaic-tiler](https://github.com/kylebarron/landsat-mosaic-tiler) -> This repo hosts all the code for landsatlive.live website and APIs.

## Maxar
Satellites owned by Maxar (formerly DigitalGlobe) include [GeoEye-1](https://en.wikipedia.org/wiki/GeoEye-1), [WorldView-2](https://en.wikipedia.org/wiki/WorldView-2), [3](https://en.wikipedia.org/wiki/WorldView-3) & [4](https://en.wikipedia.org/wiki/WorldView-4)
* Maxar ARD (COG plus data masks, with STAC) [sample data in S3](https://ard.maxar.com/docs/sdk/examples/outputs/)
* [Dataset on AWS](https://spacenet.ai/datasets/) -> see [this getting started notebook](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53) and this notebook on the [off-Nadir dataset](https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3)
* [cloud_optimized_geotif here](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* [WorldView cloud optimized geotiffs](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* For more Worldview imagery see Kaggle DSTL competition.

## Planet
* [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/), supported through Norway’s International Climate & Forests Initiative. [BBC coverage](https://www.bbc.co.uk/news/science-environment-54651453)
* Planet have made imagery available via kaggle competitions

## UC Merced
Land use classification dataset with 21 classes and 100 RGB TIFF images for each class. Each image measures 256x256 pixels with a pixel resolution of 1 foot
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* Available as a Tensorflow dataset -> https://www.tensorflow.org/datasets/catalog/uc_merced
* Also [available as a multi-label dataset](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d)
* Read [Vision Transformers for Remote Sensing Image Classification](https://www.mdpi.com/2072-4292/13/3/516/htm) where a Vision Transformer classifier achieves 98.49% classification accuracy on Merced

## EuroSAT
Land use classification dataset of Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples. Available in RGB and 13 band versions
* [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT) -> publication where a CNN achieves a classification accuracy 98.57%
* Repos using fastai [here](https://github.com/shakasom/Deep-Learning-for-Satellite-Imagery) and [here](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html)
* [evolved_channel_selection](http://matpalm.com/blog/evolved_channel_selection/) -> explores the trade off between mixed resolutions and whether to use a channel at all, with [repo](https://github.com/matpalm/evolved_channel_selection)
* RGB version available as [dataset in pytorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT) with the 13 band version [in torchgeo](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat). Checkout the tutorial on [data augmentation with this dataset](https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html)
* RGB and 13 band versions [in tensorflow](https://www.tensorflow.org/datasets/catalog/eurosat)

## PatternNet
Land use classification dataset with 38 classes and 800 RGB JPG images for each class
* https://sites.google.com/view/zhouwx/dataset?authuser=0
* Publication: [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424)

## Million-AID
A large-scale benchmark dataset containing million instances for RS scene classification, 51 scene categories organized by the hierarchical category
* https://captain-whu.github.io/DiRS/
* [Pretrained models](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)
* Also see [AID](https://captain-whu.github.io/AID/), [AID-Multilabel-Dataset](https://github.com/Hua-YS/AID-Multilabel-Dataset) & [DFC15-multilabel-dataset](https://github.com/Hua-YS/DFC15-Multilabel-Dataset)

## DIOR object detection dataset
A large-scale benchmark dataset for object detection in optical remote sensing images, which consists of 23,463 images and 192,518 object instances annotated with horizontal bounding boxes
* https://gcheng-nwpu.github.io/
* https://arxiv.org/abs/1909.00133
* [ors-detection](https://github.com/Vlad15lav/ors-detection) -> Object Detection on the DIOR dataset using YOLOv3
* [dior_detect](https://github.com/hm-better/dior_detect) -> benchmarks for object detection on DIOR dataset
* [Tools](https://github.com/CrazyStoneonRoad/Tools) -> for dealing with the DIOR

## Multiscene
MultiScene dataset aims at two tasks: Developing algorithms for multi-scene recognition & Network learning with noisy labels
* https://multiscene.github.io/ & https://github.com/Hua-YS/Multi-Scene-Recognition

## FAIR1M object detection dataset
A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery
* [arxiv papr](https://arxiv.org/abs/2103.05569)
* Download at gaofen-challenge.com
* [2020Gaofen](https://github.com/AICyberTeam/2020Gaofen) -> 2020 Gaofen Challenge data, baselines, and metrics

## DOTA object detection dataset
A Large-Scale Benchmark and Challenges for Object Detection in Aerial Images
* https://captain-whu.github.io/DOTA/index.html
* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for loading dataset
* [Arxiv paper](https://arxiv.org/abs/1711.10398)
* [Pretrained models in mmrotate](https://github.com/open-mmlab/mmrotate)
* [DOTA2VOCtools](https://github.com/Complicateddd/DOTA2VOCtools) -> dataset split and transform to voc format
* Segmentation annotations available in iSAID dataset

## iSAID instance segmentation dataset
A Large-scale Dataset for Instance Segmentation in Aerial Images
* https://captain-whu.github.io/iSAID/dataset.html
* Uses images from the DOTA dataset
* [Object Detection in Aerial Imagery](https://arxiv.org/abs/2211.15479) -> shows the performance of two-stage, one-stage and attention based object detectors on the iSAID dataset

## HRSC RGB ship object detection dataset
* https://www.kaggle.com/datasets/guofeng/hrsc2016
* [Pretrained models in mmrotate](https://github.com/open-mmlab/mmrotate)
* [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch)

## SAR Ship Detection Dataset (SSDD)
* https://github.com/TianwenZhang0825/Official-SSDD
* [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch)

## High-Resolution SAR Rotation Ship Detection Dataset (SRSDD)
* [Github](https://github.com/HeuristicLU/SRSDD-V1.0)
* [A Lightweight Model for Ship Detection and Recognition in Complex-Scene SAR Images](https://www.mdpi.com/2072-4292/14/23/6053)

## LEVIR ship dataset
A dataset for tiny ship detection under medium-resolution remote sensing images. Annotations in bounding box format
* [LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship)
<!-- markdown-link-check-disable -->
* Hosted on [Nucleus](https://dashboard.scale.com/nucleus/ds_cbsghny30nf00b1x3w7g?utm_source=open_dataset&utm_medium=github&utm_campaign=levir_ships)
<!-- markdown-link-check-enable -->

## SAR Aircraft Detection Dataset
2966 non-overlapped 224×224 slices are collected with 7835 aircraft targets
* https://github.com/hust-rslab/SAR-aircraft-data

## xView1: Objects in context for overhead imagery
A fine-grained object detection dataset with 60 object classes along an ontology of 8 class types. Over 1,000,000 objects across over 1,400 km^2 of 0.3m resolution imagery. Annotations in bounding box format
* [Official website](http://xviewdataset.org/)
* [arXiv paper](https://arxiv.org/abs/1802.07856).
* [paperswithcode](https://paperswithcode.com/dataset/xview)

## xView2: xBD building damage assessment
Annotated high-resolution satellite imagery for building damage assessment, precise segmentation masks and damage labels on a four-level spectrum, 0.3m resolution imagery
* [Official website](https://xview2.org/)
* [arXiv paper](https://arxiv.org/abs/1911.09296)
* [paperswithcode](https://paperswithcode.com/paper/xbd-a-dataset-for-assessing-building-damage)

## xView3: Detecting dark vessels in SAR
Detecting dark vessels engaged in illegal, unreported, and unregulated (IUU) fishing activities on synthetic aperture radar (SAR) imagery. With human and algorithm annotated instances of vessels and fixed infrastructure across 43,200,000 km^2 of Sentinel-1 imagery, this multi-modal dataset enables algorithms to detect and classify dark vessels
* [Official website](https://iuu.xview.us/)
* [arXiv paper](https://arxiv.org/abs/2206.00897)
* [Github](https://github.com/DIUx-xView) -> all reference code, dataset processing utilities, and winning model codes + weights
* [paperswithcode](https://paperswithcode.com/dataset/xview3-sar)

## Vehicle Detection in Aerial Imagery (VEDAI)
Vehicle Detection in Aerial Imagery. Bounding box annotations
* https://downloads.greyc.fr/vedai/
* [pytorch-vedai](https://github.com/MichelHalmes/pytorch-vedai)

## Cars Overhead With Context (COWC)
Large set of annotated cars from overhead. Established baseline for object detection and counting tasks. Annotations in bounding box format
* http://gdo152.ucllnl.org/cowc/
* https://github.com/LLNL/cowc
* [Detecting cars from aerial imagery for the NATO Innovation Challenge](https://arthurdouillard.com/post/nato-challenge/)

## AI-TOD - tiny object detection
The mean size of objects in AI-TOD is about 12.8 pixels, which is much smaller than other datasets. Annotations in bounding box format
* https://github.com/jwwangchn/AI-TOD
* [NWD](https://github.com/jwwangchn/NWD) -> code for 2021 [paper](https://arxiv.org/abs/2110.13389): A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. Uses AI-TOD dataset
* [AI-TOD-v2](https://chasel-tsui.github.io/AI-TOD-v2/) -> meticulously relabelling of the v1 dataset

## Counting from Sky
A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method
* https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method

## AIRS (Aerial Imagery for Roof Segmentation)
Public dataset for roof segmentation from very-high-resolution aerial imagery (7.5cm). Covers almost the full area of Christchurch, the largest city in the South Island of New Zealand.
* [On Kaggle](https://www.kaggle.com/atilol/aerialimageryforroofsegmentation)
* [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset

## Inria building/not building segmentation dataset
RGB GeoTIFF at spatial resolution of 0.3 m. Data covering Austin, Chicago, Kitsap County, Western & Easter Tyrol, Innsbruck, San Francisco & Vienna
* https://project.inria.fr/aerialimagelabeling/contest/
* [SemSegBuildings](https://github.com/SharpestProjects/SemSegBuildings) -> Project using fast.ai framework for semantic segmentation on Inria building segmentation dataset
* [UNet_keras_for_RSimage](https://github.com/loveswine/UNet_keras_for_RSimage) -> keras code for binary semantic segmentation

## AICrowd Mapping Challenge: building segmentation dataset
300x300 pixel RGB images with annotations in COCO format. Imagery appears to be global but with significant fraction from North America
* Dataset release as part of the [mapping-challenge](https://www.aicrowd.com/challenges/mapping-challenge)
* Winning solution published by neptune.ai [here](https://github.com/neptune-ai/open-solution-mapping-challenge), achieved precision 0.943 and recall 0.954 using Unet with Resnet.
* [mappingchallenge](https://github.com/krishanr/mappingchallenge) -> YOLOv5 applied to the AICrowd Mapping Challenge dataset

## BONAI - building footprint dataset
BONAI (Buildings in Off-Nadir Aerial Images) is a dataset for building footprint extraction (BFE) in off-nadir aerial images
* https://github.com/jwwangchn/BONAI

## GID15 large scale semantic segmentation dataset
* https://captain-whu.github.io/GID15/

## LEVIR-CD building change detection dataset
* https://justchenhao.github.io/LEVIR/
* [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) -> pytorch implemention of FCCDN for change detection task
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset uses LEVIR-CD imagery

## ISPRS
Semantic segmentation dataset. 38 patches of 6000x6000 pixels, each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic, and a DSM. Resolution 5 cm
* https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

## SpaceNet
SpaceNet is a series of competitions with datasets and utilities provided. The challenges covered are: (1 & 2) building segmentation, (3) road segmentation, (4) off-nadir buildings, (5) road network extraction, (6) multi-senor mapping, (7) multi-temporal urban change, (8) Flood Detection Challenge Using Multiclass Segmentation
* [spacenet.ai](https://spacenet.ai/) is an online hub for data, challenges, algorithms, and tools
* [The SpaceNet 7 Multi-Temporal Urban Development Challenge: Dataset Release](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-dataset-release-9e6e5f65c8d5)
* [spacenet-three-topcoder](https://github.com/snakers4/spacenet-three-topcoder) solution
* [official utilities](https://github.com/SpaceNetChallenge/utilities) -> Packages intended to assist in the preprocessing of SpaceNet satellite imagery dataset to a format that is consumable by machine learning algorithms
* [andraugust spacenet-utils](https://github.com/andraugust/spacenet-utils) -> Display geotiff image with building-polygon overlay & label buildings using kNN on the pixel spectra
* [Spacenet-Building-Detection](https://github.com/IdanC1s2/Spacenet-Building-Detection) -> uses keras and [Spacenet 1 dataset](https://spacenet.ai/spacenet-buildings-dataset-v1/)
* [Spacenet 8 winners blog post](https://medium.com/@SpaceNet_Project/spacenet-8-a-closer-look-at-the-winning-approaches-75ff4033bf53)

## WorldStrat Dataset
Nearly 10,000 km² of free high-resolution satellite imagery of unique locations which ensure stratified representation of all types of land-use across the world: from agriculture to ice caps, from forests to multiple urbanization densities.
* https://github.com/worldstrat/worldstrat
* [Quick tour of the WorldStrat Dataset](https://www.satellite-image-deep-learning.com/p/quick-tour-of-the-worldstrat-dataset)
* Each high-resolution image (1.5 m/pixel) comes with multiple temporally-matched low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites (10 m/pixel)
* Several super-resolution benchmark models trained on it

## Satlas
A Large-Scale, Multi-Task Dataset for Remote Sensing Image Understanding. Annotates all modalities (classification, segmentation, object detection etc)
* https://satlas.allenai.org/
* Dataset release in January 2023

## RF100: object detection benchmark
RF100 is compiled from 100 real world datasets that straddle a range of domains. The aim is that performance evaluation on this dataset will enable a more nuanced guide of how a model will perform in different domains. Contains 10k aerial images
* https://www.rf100.org/
* https://github.com/roboflow-ai/roboflow-100-benchmark

## Tensorflow datasets
* [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) -> RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) -> EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet) -> a large-scale Sentinel-2 land use classification dataset, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels. Official website includes version of the dataset with Sentinel 1 & 2 chips
* [so2sat](https://www.tensorflow.org/datasets/catalog/so2sat) -> a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by Sentinel 1 & 2

## AWS datasets
* [Earth on AWS](https://aws.amazon.com/earth/) is the AWS equivalent of Google Earth Engine
* Currently 36 satellite datasets on the [Registry of Open Data on AWS](https://registry.opendata.aws)

## Microsoft datasets
* [US Building Footprints](https://github.com/Microsoft/USBuildingFootprints) -> building footprints in all 50 US states, GeoJSON format, generated using semantic segmentation. Also [Australia](https://github.com/microsoft/AustraliaBuildingFootprints), [Canadian](https://github.com/Microsoft/CanadianBuildingFootprints), [Uganda-Tanzania](https://github.com/microsoft/Uganda-Tanzania-Building-Footprints), [Kenya-Nigeria](https://github.com/microsoft/KenyaNigeriaBuildingFootprints) and [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints) are available. Checkout [RasterizingBuildingFootprints](https://github.com/mehdiheris/RasterizingBuildingFootprints) to convert vector shapefiles to raster layers
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) is a Dask-Gateway enabled JupyterHub deployment focused on supporting scalable geospatial analysis, [source repo](https://github.com/microsoft/planetary-computer-hub)
* [landcover-orinoquia](https://github.com/microsoft/landcover-orinoquia) -> Land cover mapping of the Orinoquía region in Colombia, in collaboration with Wildlife Conservation Society Colombia. An #AIforEarth project
* [RoadDetections dataset by Microsoft](https://github.com/microsoft/RoadDetections)

## Google datasets
* [open-buildings](https://sites.research.google/open-buildings/) -> A dataset of building footprints to support social good applications covering 64% of the African continent. Read [Mapping Africa’s Buildings with Satellite Imagery](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html)

## Google Earth Engine (GEE)
Since there is a whole community around GEE I will not reproduce it here but list very select references. Get started at https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* Supports large scale processing with classical algorithms, e.g. clustering for land use. For deep learning, you export datasets from GEE as tfrecords, train on your preferred GPU platform, then upload inference results back to GEE
* [awesome-google-earth-engine](https://github.com/gee-community/awesome-google-earth-engine)
* [Awesome-GEE](https://github.com/giswqs/Awesome-GEE)
* [awesome-earth-engine-apps](https://github.com/philippgaertner/awesome-earth-engine-apps)
* [How to Use Google Earth Engine and Python API to Export Images to Roboflow](https://blog.roboflow.com/how-to-use-google-earth-engine-with-roboflow/) -> to acquire training data
* [ee-fastapi](https://github.com/csaybar/ee-fastapi) is a simple FastAPI web application for performing flood detection using Google Earth Engine in the backend.
* [How to Download High-Resolution Satellite Data for Anywhere on Earth](https://towardsdatascience.com/how-to-download-high-resolution-satellite-data-for-anywhere-on-earth-5e6dddee2803)
* [wxee](https://github.com/aazuspan/wxee) -> Export data from GEE to xarray using wxee then train with pytorch or tensorflow models. Useful since GEE only suports tfrecord export natively

## Radiant Earth
* https://www.radiant.earth/
* Datasets and also models on https://mlhub.earth/

## Image captioning datasets
* [RSICD](https://github.com/201528014227051/RSICD_optimal) -> 10921 images with five sentences descriptions per image. Used in  [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd), models at [this repo](https://github.com/arampacha/CLIP-rsicd)
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset contains 10077 pairs of bi-temporal remote sensing images and 50385 sentences describing the differences between images. Uses LEVIR-CD imagery

## Weather Datasets
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data
* [DeepWeather](https://github.com/adamhazimeh/DeepWeather) -> improve weather forecasting accuracy by analyzing satellite images

## Forest datasets
* [awesome-forests](https://github.com/blutjens/awesome-forests) -> A curated list of ground-truth forest datasets for the machine learning and forestry community
* [ReforesTree](https://github.com/gyrrei/ReforesTree) -> A dataset for estimating tropical forest biomass based on drone and field data
* [yosemite-tree-dataset](https://github.com/nightonion/yosemite-tree-dataset) -> a benchmark dataset for tree counting from aerial images

## Geospatial datasets
* [Resource Watch](https://resourcewatch.org/data/explore) provides a wide range of geospatial datasets and a UI to visualise them

## Time series & change detection datasets
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) -> A Time Series Dataset for Crop Type Mapping
* The SeCo dataset contains image patches from Sentinel-2 tiles captured at different timestamps at each geographical location. [Download SeCo here](https://github.com/ElementAI/seasonal-contrast)
* [Onera Satellite Change Detection Dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018
* [SYSU-CD](https://github.com/liumency/SYSU-CD) -> The dataset contains 20000 pairs of 0.5-m aerial images of size 256×256 taken between the years 2007 and 2014 in Hong Kong

### DEM (digital elevation maps)
* Shuttle Radar Topography Mission, search online at usgs.gov
* Copernicus Digital Elevation Model (DEM) on S3, represents the surface of the Earth including buildings, infrastructure and vegetation. Data is provided as Cloud Optimized GeoTIFFs. [link](https://registry.opendata.aws/copernicus-dem/)
* [Awesome-DEM](https://github.com/DahnJ/Awesome-DEM)

## UAV & Drone datasets
* Many on https://www.visualdata.io
* [AU-AIR dataset](https://bozcani.github.io/auairdataset) -> a multi-modal UAV dataset for object detection.
* [ERA](https://lcmou.github.io/ERA_Dataset/) ->  A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos.
* [Aerial Maritime Drone Dataset](https://public.roboflow.ai/object-detection/aerial-maritime) -> bounding boxes
* [RetinaNet for pedestrian detection](https://towardsdatascience.com/pedestrian-detection-in-aerial-images-using-retinanet-9053e8a72c6) -> bounding boxes
* [Dataset of thermal and visible aerial images for multi-modal and multi-spectral image registration and fusion](https://www.sciencedirect.com/science/article/pii/S2352340920302201) -> The dataset consists of 30 visible images and their metadata, 80 thermal images and their metadata, and a visible georeferenced orthoimage.
* [BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos](https://ieeexplore.ieee.org/document/9093284) -> Thermal IR videos of humans and animals. With [Github repo](https://github.com/exb7900/BIRDSAI)
* [ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos](https://lcmou.github.io/ERA_Dataset/)
* [DroneVehicle](https://github.com/VisDrone/DroneVehicle) -> Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning. Annotations are rotated bounding boxes. With [Github repo](https://github.com/SunYM2020/UA-CMDet)
* [UAVOD10](https://github.com/weihancug/10-category-UAV-small-weak-object-detection-dataset-UAVOD10) -> 10 class of objects at 15 cm resolution. Classes are; building, ship, vehicle, prefabricated house, well, cable tower, pool, landslide, cultivation mesh cage, and quarry. Bounding boxes
* [Busy-parking-lot-dataset---vehicle-detection-in-UAV-video](https://github.com/zhu-xlab/Busy-parking-lot-dataset---vehicle-detection-in-UAV-video) -> Vehicle instance segmentation. Unsure format of annotations, possible Matlab specific
* [dd-ml-segmentation-benchmark](https://github.com/dronedeploy/dd-ml-segmentation-benchmark) -> DroneDeploy Machine Learning Segmentation Benchmark
* [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) -> Vision Benchmark for Maritime Search and Rescue. Bounding box object detection, single-object tracking and multi-object tracking annotations

## Other datasets
* [land-use-land-cover-datasets](https://github.com/r-wenger/land-use-land-cover-datasets)
* [EORSSD-dataset](https://github.com/rmcong/EORSSD-dataset) -> Extended Optical Remote Sensing Saliency Detection (EORSSD) Dataset
* [RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU) -> 46 scene classes for image classification, free for education, research and commercial use
* [RSOD-Dataset](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) -> dataset for object detection in PASCAL VOC format. Aircraft, playgrounds, overpasses & oiltanks
* [VHR-10_dataset_coco](https://github.com/chaozhong2010/VHR-10_dataset_coco) -> Object detection and instance segmentation dataset based on NWPU VHR-10 dataset. RGB & SAR
* [HRSID](https://github.com/chaozhong2010/HRSID) -> high resolution sar images dataset for ship detection, semantic segmentation, and instance segmentation tasks
* [MAR20](https://gcheng-nwpu.github.io/) -> Military Aircraft Recognition dataset
* [RSSCN7](https://github.com/palewithout/RSSCN7) -> Dataset of the article “Deep Learning Based Feature Selection for Remote Sensing Scene Classification”
* [Sewage-Treatment-Plant-Dataset](https://github.com/peijinwang/Sewage-Treatment-Plant-Dataset) -> object detection
* [TGRS-HRRSD-Dataset](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset) -> High Resolution Remote Sensing Detection (HRRSD)
* [MUSIC4HA](https://github.com/gistairc/MUSIC4HA) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Hot Area
* [MUSIC4GC](https://github.com/gistairc/MUSIC4GC) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Golf Course
* [MUSIC4P3](https://github.com/gistairc/MUSIC4P3) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Photovoltaic Power Plants (solar panels)
* [ABCDdataset](https://github.com/gistairc/ABCDdataset) -> damage detection dataset to identify whether buildings have been washed-away by tsunami
* [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) -> Oil and Gas Tank Dataset
* [LS-SSDD-v1.0-OPEN](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN) -> Large-Scale SAR Ship Detection Dataset
* [S2Looking](https://github.com/S2Looking/Dataset) -> A Satellite Side-Looking Dataset for Building Change Detection, [paper](https://arxiv.org/abs/2107.09244)
* [Zurich Summer Dataset](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset) -> Semantic segmentation of urban scenes
* [AISD](https://github.com/RSrscoder/AISD) -> Aerial Imagery dataset for Shadow Detection
* [Awesome-Remote-Sensing-Relative-Radiometric-Normalization-Datasets](https://github.com/ArminMoghimi/Awesome-Remote-Sensing-Relative-Radiometric-Normalization-Datasets)
* [SearchAndRescueNet](https://github.com/michaelthoreau/SearchAndRescueNet) -> Satellite Imagery for Search And Rescue Dataset, with example Faster R-CNN model
* [geonrw](https://ieee-dataport.org/open-access/geonrw) -> orthorectified aerial photographs, LiDAR derived digital elevation models and segmentation maps with 10 classes. With [repo](https://github.com/gbaier/geonrw)
* [Thermal power plans dataset](https://github.com/wenxinYin/AIR-TPPDD)
* [University1652-Baseline](https://github.com/layumi/University1652-Baseline) -> A Multi-view Multi-source Benchmark for Drone-based Geo-localization
* [benchmark_ISPRS2021](https://github.com/whuwuteng/benchmark_ISPRS2021) -> A new stereo dense matching benchmark dataset for deep learning
* [WHU-SEN-City](https://github.com/whu-csl/WHU-SEN-City) -> A paired SAR-to-optical image translation dataset which covers 34 big cities of China
* [SAR_vehicle_detection_dataset](https://github.com/whu-csl/SAR_vehicle_detection_dataset) -> 104 SAR images for vehicle detection, collected from Sandia MiniSAR/FARAD SAR images and MSTAR images
* [ERA-DATASET](https://github.com/zhu-xlab/ERA-DATASET) -> A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos
* [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) -> a large-scale dataset for self-supervised learning in Earth observation
* [UBC-dataset](https://github.com/AICyberTeam/UBC-dataset) -> a dataset for building detection and classification from very high-resolution satellite imagery with the focus on object-level interpretation of individual buildings
* [AIR-CD](https://github.com/AICyberTeam/AIR-CD) -> a challenging cloud detection data set called AIR-CD, with higher spatial resolution and more representative landcover types
* [AIR-PolSAR-Seg](https://github.com/AICyberTeam/AIR-PolSAR-Seg) -> a challenging PolSAR terrain segmentation dataset
* [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) -> High-Resolution Cloud Detection Dataset comprising 150 RGB images and a resolution varying from 0.5 to 15 m in different global regions
* [AeroRIT](https://github.com/aneesh3108/AeroRIT) -> A New Scene for Hyperspectral Image Analysis
* [Building_Dataset](https://github.com/QiaoWenfan/Building_Dataset) -> High-speed Rail Line Building Dataset Display
* [Haiming-Z/MtS-WH-reference-map](https://github.com/Haiming-Z/MtS-WH-reference-map) -> a reference map for change detection based on MtS-WH
* [MtS-WH-Dataset](https://github.com/rulixiang/MtS-WH-Dataset) -> Multi-temporal Scene WuHan (MtS-WH) Dataset
* [Multi-modality-image-matching](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods) -> image matching dataset including several remote sensing modalities
* [RID](https://github.com/TUMFTM/RID) -> Roof Information Dataset for CV-Based Photovoltaic Potential Assessment. With [paper](https://www.mdpi.com/2072-4292/14/10/2299)
* [APKLOT](https://github.com/langheran/APKLOT) -> A dataset for aerial parking block segmentation
* [QXS-SAROPT](https://github.com/yaoxu008/QXS-SAROPT) -> Optical and SAR pairing dataset from the [paper](https://arxiv.org/abs/2103.08259): The QXS-SAROPT Dataset for Deep Learning in SAR-Optical Data Fusion
* [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD) -> SAR-ACD consists of 4322 aircraft clips with 6 civil aircraft categories and 14 other aircraft categories
* [SODA](https://shaunyuan22.github.io/SODA/) -> A large-scale Small Object Detection dataset. SODA-A comprises 2510 high-resolution images of aerial scenes, which has 800203 instances annotated with oriented rectangle box annotations over 9 classes.
* [Data-CSHSI](https://github.com/YuxiangZhang-BIT/Data-CSHSI) -> Open source datasets for Cross-Scene Hyperspectral Image Classification, includes Houston, Pavia & HyRank datasets
* [SynthWakeSAR](https://data.bris.ac.uk/data/dataset/30kvuvmatwzij2mz1573zqumfx) -> A Synthetic SAR Dataset for Deep Learning Classification of Ships at Sea, with [paper](https://www.mdpi.com/2072-4292/14/16/3999)
* [SAR2Opt-Heterogeneous-Dataset](https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset) -> SAR-optical images to be used as a benchmark in change detection and image transaltion on remote sensing images
* [urban-tree-detection-data](https://github.com/jonathanventura/urban-tree-detection-data) -> Dataset for training and evaluating tree detectors in urban environments with aerial imagery
* [Landsat 8 Cloud Cover Assessment Validation Data](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)
* [Attribute-Cooperated-Classification-Datasets](https://github.com/CrazyStoneonRoad/Attribute-Cooperated-Classification-Datasets) -> Three datasets based on AID, UCM, and Sydney. For each image, there is a label of scene classification and a label vector of attribute items.
* [RarePlanes](https://www.cosmiqworks.org/rareplanes-public-user-guide/) is a dataset of real (Maxar) and simulated images of planes. Utility functions at [VisionSystemsInc - RarePlanes](https://github.com/VisionSystemsInc/RarePlanes)
* [dynnet](https://github.com/aysim/dynnet) -> DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
* [open_earth_map](https://github.com/bao18/open_earth_map) -> a benchmark dataset for global high-resolution land cover mapping
* [Satellite imagery datasets containing ships](https://github.com/NaLiu613/Satellite-Imagery-Datasets-Containing-Ships) -> A list of radar and optical satellite datasets for ship detection, classification, semantic segmentation and instance segmentation tasks
* [SolarDK](https://arxiv.org/abs/2212.01260) -> A high-resolution urban solar panel image classification and localization dataset

## Kaggle
Kaggle hosts over > 200 satellite image datasets, [search results here](https://www.kaggle.com/search?q=satellite+image+in%3Adatasets).
The [kaggle blog](http://blog.kaggle.com) is an interesting read.

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images from planet Dove satellite constellation
* 12 classes including - **cloudy, primary + waterway** etc
* [1st place winner interview - used 11 custom CNN](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)
* [Understanding the Amazon Rainforest with Multi-Label Classification + VGG-19, Inceptionv3, AlexNet & Transfer Learning](https://towardsdatascience.com/understanding-the-amazon-rainforest-with-multi-label-classification-vgg-19-inceptionv3-5084544fb655)
* [amazon-classifier](https://github.com/mikeskaug/amazon-classifier) -> compares random forest with CNN
* [multilabel-classification](https://github.com/muneeb706/multilabel-classification) -> compares various CNN architecutres
* [Planet-Amazon-Kaggle](https://github.com/Skumarr53/Planet-Amazon-Kaggle) -> uses fast.ai
* [deforestation_deep_learning](https://github.com/schumanzhang/deforestation_deep_learning)
* [Track-Human-Footprint-in-Amazon-using-Deep-Learning](https://github.com/sahanasub/Track-Human-Footprint-in-Amazon-using-Deep-Learning)
* [Amazon-Rainforest-CNN](https://github.com/cldowdy/Amazon-Rainforest-CNN) -> uses a 3-layer CNN in Tensorflow
* [rainforest-tagging](https://github.com/minggli/rainforest-tagging) -> Convolutional Neural Net and Recurrent Neural Net in Tensorflow for satellite images multi-label classification

### Kaggle - DSTL segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* Rating - medium, many good examples (see the Discussion as well as kernels), but as this competition was run a couple of years ago many examples use python 2
* WorldView 3 - 45 satellite images covering 1km x 1km in both 3 (i.e. RGB) and 16-band (400nm - SWIR) images
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**
* [Interview with 1st place winner who used segmentation networks](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/) - 40+ models, each tweaked for particular target (e.g. roads, trees)
* [ZF_UNET_224_Pretrained_Model 2nd place solution](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) ->
* [3rd place soluton](https://github.com/osin-vladimir/kaggle-satellite-imagery-feature-detection) -> which explored pansharpening & calculating reflectance indices, with [arxiv paper](https://arxiv.org/abs/1706.06169) 
* [Deepsense 4th place solution](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
* [Entry by lopuhin](https://github.com/lopuhin/kaggle-dstl) using UNet with batch-normalization
* [Multi-class semantic segmentation of satellite images using U-Net](https://github.com/rogerxujiang/dstl_unet) using DSTL dataset, tensorflow 1 & python 2.7. Accompanying [article](https://towardsdatascience.com/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40)
* [Deep-Satellite-Image-Segmentation](https://github.com/antoine-spahr/Deep-Satellite-Image-Segmentation)
* [Dstl-Satellite-Imagery-Feature-Detection-Improved](https://github.com/dsp6414/Dstl-Satellite-Imagery-Feature-Detection-Improved)
* [Satellite-imagery-feature-detection](https://github.com/ArangurenAndres/Satellite-imagery-feature-detection)
* [Satellite_Image_Classification](https://github.com/aditya-sawadh/Satellite_Image_Classification) -> using XGBoost and ensemble classification methods
* [Unet-for-Satellite](https://github.com/justinishikawa/Unet-for-Satellite)
* [building-segmentation](https://github.com/jimpala/building-segmentation) -> TensorFlow U-Net implementation trained to segment buildings in satellite imagery

### Kaggle - DeepSat land cover classification
* https://www.kaggle.com/datasets/crawford/deepsat-sat4 & https://www.kaggle.com/datasets/crawford/deepsat-sat6
* [DeepSat-Kaggle](https://github.com/athulsudheesh/DeepSat-Kaggle) -> uses Julia
* [deepsat-aws-emr-pyspark](https://github.com/hellosaumil/deepsat-aws-emr-pyspark) -> Using PySpark for Image Classification on Satellite Imagery of Agricultural Terrains

### Kaggle - Airbus ship detection challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* Rating - medium, most solutions using deep-learning, many kernels, [good example kernel](https://www.kaggle.com/kmader/baseline-u-net-model-part-1)
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined
* [Deep Learning for Ship Detection and Segmentation](https://towardsdatascience.com/deep-learning-for-ship-detection-and-segmentation-71d223aca649) -> treated as instance segmentation problem, with [notebook](https://github.com/abhinavsagar/kaggle-notebooks/blob/master/ship_segmentation.ipynb)
* [Lessons Learned from Kaggle’s Airbus Challenge](https://towardsdatascience.com/lessons-learned-from-kaggles-airbus-challenge-252e25c5efac)
* [Airbus-Ship-Detection](https://github.com/kheyer/Airbus-Ship-Detection) -> This solution scored 139 out of 884 for the competition, combines ResNeXt50 based classifier and a U-net segmentation model
* [Ship-Detection-Project](https://github.com/ZTong1201/Ship-Detection-Project) -> uses Mask R-CNN and UNet model
* [Airbus_SDC](https://github.com/WillieMaddox/Airbus_SDC)
* [Airbus_SDC_dup](https://github.com/WillieMaddox/Airbus_SDC_dup) -> Project focused on detecting duplicate regions of overlapping satellite imagery. Applied to Airbus ship detection dataset
* [airbus-ship-detection](https://github.com/jancervenka/airbus-ship-detection) -> CNN with REST API
* [Ship-Detection-from-Satellite-Images-using-YOLOV4](https://github.com/debasis-dotcom/Ship-Detection-from-Satellite-Images-using-YOLOV4) -> uses Kaggle Airbus Ship Detection dataset
* [Image Segmentation: Kaggle experience](https://towardsdatascience.com/image-segmentation-kaggle-experience-9a41cb8924f0) -> Medium article by gold medal winner Vlad Shmyhlo

### Kaggle - Shipsnet classification dataset
* https://www.kaggle.com/rhammell/ships-in-satellite-imagery -> Classify ships in San Franciso Bay using Planet satellite imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, 3 meter pixel size
* [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning

### Kaggle - Ships in Google Earth
* https://www.kaggle.com/tomluther/ships-in-google-earth
* 794 jpegs showing various sized ships in satellite imagery, annotations in Pascal VOC format for object detection models
* [kaggle-ships-in-Google-Earth-yolov5](https://github.com/robmarkcole/kaggle-ships-in-Google-Earth-yolov5)

### Kaggle - Ships in San Franciso Bay
* https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, provided by Planet
* [DeepLearningShipDetection](https://github.com/PenguinDan/DeepLearningShipDetection)
* [Ship-Detection-Using-Satellite-Imagery](https://github.com/Dhruvisha29/Ship-Detection-Using-Satellite-Imagery)

### Kaggle - Swimming pool and car detection using satellite imagery
* https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection
* 3750 satellite images of residential areas with annotation data for swimming pools and cars
* [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5)

### Kaggle - Planesnet classification dataset
* https://www.kaggle.com/rhammell/planesnet -> Detect aircraft in Planet satellite image chips
* 20x20 RGB images, the "plane" class includes 8000 images and the "no-plane" class includes 24000 images
* [Dataset repo](https://github.com/rhammell/planesnet) and [planesnet-detector](https://github.com/rhammell/planesnet-detector) demonstrates a small CNN classifier on this dataset
* [ergo-planes-detector](https://github.com/evilsocket/ergo-planes-detector) -> An ergo based project that relies on a convolutional neural network to detect airplanes from satellite imagery, uses the PlanesNet dataset
* [Using AWS SageMaker/PlanesNet to process Satellite Imagery](https://github.com/kskalvar/aws-sagemaker-planesnet-imagery)
* [Airplane-in-Planet-Image](https://github.com/MaxLenormand/Airplane-in-Planet-Image) -> pytorch model

### Kaggle - CGI Planes in Satellite Imagery w/ BBoxes
* https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes
* 500 computer generated satellite images of planes
* [Faster RCNN to detect airplanes](https://github.com/ShubhankarRawat/Airplane-Detection-for-Satellites)
* [aircraft-detection-from-satellite-images-yolov3](https://github.com/emrekrtorun/aircraft-detection-from-satellite-images-yolov3)

### Kaggle - Draper challenge to place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Rating - hard. Not many useful kernels.
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.
* Kaggle interviews for entrants who [used XGBOOST](http://blog.kaggle.com/2016/09/15/draper-satellite-image-chronology-machine-learning-solution-vicens-gaitan/) and a [hybrid human/ML approach](http://blog.kaggle.com/2016/09/08/draper-satellite-image-chronology-damien-soukhavong/)
* [deep-cnn-sat-image-time-series](https://github.com/MickyDowns/deep-cnn-sat-image-time-series) -> uses LSTM

### Kaggle - Dubai segmentation
* https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery
* 72 satellite images of Dubai, the UAE, and is segmented into 6 classes
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset
* [Multiclass-semantic-segmentation-in-satallite-images](https://github.com/tahirjhan/Multiclass-semantic-segmentation-in-satallite-images) -> uses keras
* [Semantic-Segmentation-using-U-Net](https://github.com/Anay21110/Semantic-Segmentation-using-U-Net) -> uses keras
* [unet_satelite_image_segmentation](https://github.com/nassimaliou/unet_satelite_image_segmentation)

### Kaggle - Massachusetts Roads & Buildings Datasets - segmentation
* https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
* https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset
* [Official published dataset](https://www.cs.toronto.edu/~vmnih/data/)
* [Road_seg_dataset](https://github.com/parth1620/Road_seg_dataset) -> subset of the roads dataset containing only 200 images and masks
* [Road and Building Semantic Segmentation in Satellite Imagery](https://github.com/Paulymorphous/Road-Segmentation) uses U-Net on the Massachusetts Roads Dataset & keras
* [Semantic-segmentation repo by fuweifu-vtoo](https://github.com/fuweifu-vtoo/Semantic-segmentation) -> uses pytorch and the [Massachusetts Buildings & Roads Datasets](https://www.cs.toronto.edu/~vmnih/data/)
* [ssai-cnn](https://github.com/mitmul/ssai-cnn) -> This is an implementation of Volodymyr Mnih's dissertation methods on his Massachusetts road & building dataset
* [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) -> pip installable library to train building footprint segmentation on satellite and aerial imagery, applied to Massachusetts Buildings Dataset and Inria Aerial Image Labeling Dataset
* [Road detection using semantic segmentation and albumentations for data augmention](https://towardsdatascience.com/road-detection-using-segmentation-models-and-albumentations-libraries-on-keras-d5434eaf73a8) using the Massachusetts Roads Dataset, U-net & Keras
* [Image-Segmentation)](https://github.com/mschulz/Image-Segmentation) -> using Massachusetts Road dataset and fast.ai

### Kaggle - Deepsat classification challenge
Not satellite but airborne imagery. Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three**
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**

### Kaggle - High resolution ship collections 2016 (HRSC2016)
* https://www.kaggle.com/guofeng/hrsc2016
* Ship images harvested from Google Earth
* [HRSC2016_SOTA](https://github.com/ming71/HRSC2016_SOTA) -> Fair comparison of different algorithms on the HRSC2016 dataset

### Kaggle - SWIM-Ship Wake Imagery Mass
* https://www.kaggle.com/datasets/lilitopia/swimship-wake-imagery-mass
* An optical ship wake detection benchmark dataset built for deep learning
* [WakeNet](https://github.com/Lilytopia/WakeNet) -> A CNN-based optical image ship wake detector, code for 2021 paper: Rethinking Automatic Ship Wake Detection: State-of-the-Art CNN-based Wake Detection via Optical Images

### Kaggle - Understanding Clouds from Satellite Images
In this challenge, you will build a model to classify cloud organization patterns from satellite images.
* https://www.kaggle.com/c/understanding_cloud_organization/
* [3rd place solution on Github by naivelamb](https://github.com/naivelamb/kaggle-cloud-organization)
* [15th place solution on Github by Soongja](https://github.com/Soongja/kaggle-clouds)
* [69th place solution on Github by yukkyo](https://github.com/yukkyo/Kaggle-Understanding-Clouds-69th-solution)
* [161st place solution on Github by michal-nahlik](https://github.com/michal-nahlik/kaggle-clouds-2019)
* [Solution by yurayli](https://github.com/yurayli/satellite-cloud-segmentation)
* [Solution by HazelMartindale](https://github.com/HazelMartindale/kaggle_understanding_clouds_learning_project) uses 3 versions of U-net architecture
* [Solution by khornlund](https://github.com/khornlund/understanding-cloud-organization)
* [Solution by Diyago](https://github.com/Diyago/Understanding-Clouds-from-Satellite-Images)
* [Solution by tanishqgautam](https://github.com/tanishqgautam/Multi-Label-Segmentation-With-FastAI)

### Kaggle - 38-Cloud Cloud Segmentation
* https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images
* Contains 38 Landsat 8 images and manually extracted pixel-level ground truths
* [38-Cloud Github repository](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) and follow up [95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset) dataset
* [How to create a custom Dataset / Loader in PyTorch, from Scratch, for multi-band Satellite Images Dataset from Kaggle](https://medium.com/analytics-vidhya/how-to-create-a-custom-dataset-loader-in-pytorch-from-scratch-for-multi-band-satellite-images-c5924e908edf)
* [Cloud-Net: A semantic segmentation CNN for cloud detection](https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection) -> an end-to-end cloud detection algorithm for Landsat 8 imagery, trained on 38-Cloud Training Set
* [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> semantic segmentation using a Unet on the Kaggle 38-Cloud dataset

### Kaggle - Airbus Aircraft Detection Dataset
* https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset
* One hundred civilian airports and over 3000 annotated commercial aircrafts
* [detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)
* [pytorch-remote-sensing](https://github.com/miko7879/pytorch-remote-sensing) -> Aircraft detection using the 'Airbus Aircraft Detection' dataset and Faster-RCNN with ResNet-50 backbone in pytorch

### Kaggle - Airbus oil storage detection dataset
* https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset
* [Oil-Storage Tank Instance Segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn/blob/main/mask_rcnn_oiltanks_gpu.ipynb) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f)
* [Oil Storage Detection on Airbus Imagery with YOLOX](https://medium.com/artificialis/oil-storage-detection-on-airbus-imagery-with-yolox-9e38eb6f7e62) -> uses the Kaggle Airbus Oil Storage Detection dataset
* [Oil-Storage-Tanks-Data-Preparation-YOLO-Format](https://github.com/shah0nawaz/Oil-Storage-Tanks-Data-Preparation-YOLO-Format)

### Kaggle - Satellite images of hurricane damage
* https://www.kaggle.com/kmader/satellite-images-of-hurricane-damage
* https://github.com/dbuscombe-usgs/HurricaneHarvey_buildingdamage

### Kaggle - Austin Zoning Satellite Images
* https://www.kaggle.com/franchenstein/austin-zoning-satellite-images
* classify a images of Austin into one of its zones, such as residential, industrial, etc. 3667 satellite images

### Kaggle - Statoil/C-CORE Iceberg Classifier Challenge
* https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
* [Deep Learning for Iceberg detection in Satellite Images](https://towardsdatascience.com/deep-learning-for-iceberg-detection-in-satellite-images-c667acf4bad0)
* [radar-image-recognition](https://github.com/siarez/radar-image-recognition)
* [Iceberg-Classification-Using-Deep-Learning](https://github.com/mankadronit/Iceberg-Classification-Using-Deep-Learning)
* [Deep-Learning-Project](https://github.com/singh-shakti94/Deep-Learning-Project)
* [iceberg-classifier-challenge solution by ShehabSunny](https://github.com/ShehabSunny/iceberg-classifier-challenge)

### Kaggle - Land Cover Classification Dataset from DeepGlobe Challenge - segmentation
* https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)
* [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net, uses DeepGlobe Land Cover Segmentation dataset, with [code](https://github.com/TarunKumar1995-glitch/land_cover_classification_unet)
* [DeepGlobe Land Cover Classification Challenge solution](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)

### Kaggle - Next Day Wildfire Spread
A Data Set to Predict Wildfire Spreading from Remote-Sensing Data
* https://www.kaggle.com/fantineh/next-day-wildfire-spread
* https://arxiv.org/abs/2112.02447

### Kaggle - Satellite Next Day Wildfire Spread
Inspired by the above dataset, using different data sources
* https://www.kaggle.com/satellitevu/satellite-next-day-wildfire-spread
* https://github.com/SatelliteVu/SatelliteVu-AWS-Disaster-Response-Hackathon

## Kaggle - Spacenet 7 Multi-Temporal Urban Change Detection
* https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development
* [SatFootprint](https://github.com/PriyanK7n/SatFootprint) -> building segmentation on the Spacenet 7 dataset

## Kaggle - Satellite Images to predict poverty in Africa
* https://www.kaggle.com/datasets/sandeshbhat/satellite-images-to-predict-povertyafrica
* Uses satellite imagery and nightlights data to predict poverty levels at a local level
* [Predicting-Poverty](https://github.com/jmather625/predicting-poverty-replication) -> Combining satellite imagery and machine learning to predict poverty, in PyTorch

## Kaggle - NOAA Fisheries Steller Sea Lion Population Count
* https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count -> count sea lions from aerial images
* [Sealion-counting](https://github.com/babyformula/Sealion-counting)
* [Sealion_Detection_Classification](https://github.com/yyc9268/Sealion_Detection_Classification)

## Kaggle - Arctic Sea Ice Image Masking
* https://www.kaggle.com/datasets/alexandersylvester/arctic-sea-ice-image-masking
* [sea_ice_remote_sensing](https://github.com/sum1lim/sea_ice_remote_sensing)

## Kaggle - Overhead-MNIST
* A Benchmark Satellite Dataset as Drop-In Replacement for MNIST
* https://www.kaggle.com/datamunge/overheadmnist -> kaggle
* https://arxiv.org/abs/2102.04266 -> paper
* https://github.com/reveondivad/ov-mnist -> github

## Kaggle - Satellite Image Classification
* https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
* [satellite-image-classification-pytorch](https://github.com/dilaraozdemir/satellite-image-classification-pytorch)

## Kaggle - EuroSAT - Sentinel-2 Dataset
* https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset
* RGB Land Cover and Land Use Classification using Sentinel-2 Satellite
* Used in paper [Image Augmentation for Satellite Images](https://arxiv.org/abs/2207.14580)

### Kaggle - miscellaneous
* https://www.kaggle.com/reubencpereira/spatial-data-repo -> Satellite + loan data
* https://www.kaggle.com/towardsentropy/oil-storage-tanks -> Image data of industrial oil tanks with bounding box annotations, estimate tank fill % from shadows
* https://www.kaggle.com/airbusgeo/airbus-wind-turbines-patches -> Airbus SPOT satellites images over wind turbines for classification
* https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes -> CGI planes object detection dataset
* https://www.kaggle.com/atilol/aerialimageryforroofsegmentation -> Aerial Imagery for Roof Segmentation
* https://www.kaggle.com/andrewmvd/ship-detection -> 621 images of boats and ships
* https://www.kaggle.com/alpereniek/vehicle-detection-from-satellite-images-data-set
* https://www.kaggle.com/sergiishchus/maxar-satellite-data -> Example Maxar data at 15 cm resolution
* https://www.kaggle.com/cici118/swimming-pool-detection-algarves-landscape
* https://www.kaggle.com/datasets/donkroco/solar-panel-module -> object detection for solar panels
* https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset -> segment roads

# Software for working with remote sensing data
[A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code) -> lists many useful tools and resources
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery by [CosmiQ Works](https://www.cosmiqworks.org/), similar to Rastervision but with some unique very vool features
* [openSAR](https://github.com/EarthBigData/openSAR) -> Synthetic Aperture Radar (SAR) Tools and Documents from Earth Big Data LLC
* [YMIR](https://github.com/industryessentials/ymir) -> YMIR provides a Rapid Data-centric Development Platform
for Vision Applications. Read the paper [here](https://arxiv.org/abs/2111.10046).
* [qhub](https://qhub.dev) -> QHub enables teams to build and maintain a cost effective and scalable compute/data science platform in the cloud.
* [imagej](https://imagej.net) -> a very versatile image viewer and processing program
* [Geo Data Viewer](https://github.com/RandomFractals/geo-data-viewer) extension for VSCode which enables opening and viewing various geo data formats with nice visualisations
* [Datasette](https://datasette.io/) is a tool for exploring and publishing data as an interactive website and accompanying API, with SQLite backend. Various plugins extend its functionality, for example to allow displaying geospatial info, render images (useful for thumbnails), and add user authentication. Available as a [desktop app](https://datasette.io/desktop). Read [Drawing shapes on a map to query a SpatiaLite database](https://simonwillison.net/2021/Jan/24/drawing-shapes-spatialite/)
* [Photoprism](https://github.com/photoprism/photoprism) is a privately hosted app for browsing, organizing, and sharing your photo collection, with support for tiffs
* [dbeaver](https://github.com/dbeaver/dbeaver) is a free universal database tool and SQL client with [geospatial features](https://github.com/dbeaver/dbeaver/wiki/Working-with-Spatial-GIS-data)
* [Grafana](https://grafana.com/) can be used to make interactive dashboards, checkout [this example showing Point data](https://blog.timescale.com/blog/grafana-variables-101/). Note there is an [AWS managed service for Grafana](https://aws.amazon.com/grafana/)
* [litestream](https://litestream.io/) -> Continuously stream SQLite changes to S3-compatible storage
* [ImageFusion)](https://github.com/JohMast/ImageFusion) -> Temporal fusion of raster image time-Series
* [nvtop](https://github.com/Syllo/nvtop) -> NVIDIA GPUs htop like monitoring tool
* [rgis](https://github.com/frewsxcv/rgis) -> Geospatial data viewer written in Rust
* [aerialbot](https://github.com/doersino/aerialbot) -> A simple yet highly configurable bot that tweets geotagged aerial imagery of a random location in the world
* [SatDump](https://github.com/altillimity/SatDump) -> A generic satellite data processing software.

## General utilities
Scripts and command line applications
* [geospatial-cli](https://github.com/JakobMiksch/geospatial-cli) -> a collection of geospatial programs with commandline interface
* [PyShp](https://github.com/GeospatialPython/pyshp) -> The Python Shapefile Library (PyShp) reads and writes Shapefiles in pure Python
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* [EarthPy](https://github.com/earthlab/earthpy) -> A set of helper functions to make working with spatial data in open source tools easier. read[Exploratory Data Analysis (EDA) on Satellite Imagery Using EarthPy](https://towardsdatascience.com/exploratory-data-analysis-eda-on-satellite-imagery-using-earthpy-c0e186fe4293)
* [pygeometa](https://geopython.github.io/pygeometa/) -> provides a lightweight and Pythonic approach for users to easily create geospatial metadata in standards-based formats using simple configuration files
* [pesto](https://airbusdefenceandspace.github.io/pesto/) -> PESTO is designed to ease the process of packaging a Python algorithm as a processing web service into a docker image. It contains shell tools to generate all the boiler plate to build an OpenAPI processing web service compliant with the Geoprocessing-API. By [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [GEOS](https://geos.readthedocs.io/en/latest/index.html) -> Google Earth Overlay Server (GEOS) is a python-based server for creating Google Earth overlays of tiled maps. Your can also display maps in the web browser, measure distances and print maps as high-quality PDF’s.
* [GeoDjango](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/) intends to be a world-class geographic Web framework. Its goal is to make it as easy as possible to build GIS Web applications and harness the power of spatially enabled data. [Some features of GDAL are supported.](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/gdal/)
* [rasterstats](https://pythonhosted.org/rasterstats/) -> summarize geospatial raster datasets based on vector geometries
* [turfpy](https://turfpy.readthedocs.io/en/latest/index.html) -> a Python library for performing geospatial data analysis which reimplements turf.js
* [rsgislib](https://github.com/remotesensinginfo/rsgislib) -> Remote Sensing and GIS Software Library; python module tools for processing spatial and image data
* [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html) -> seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner. See [eo-learn-examples](https://github.com/sentinel-hub/eo-learn-examples)
* [RStoolbox: Tools for Remote Sensing Data Analysis in R](https://bleutner.github.io/RStoolbox/)
* [nd](https://github.com/jnhansen/nd) -> Framework for the analysis of n-dimensional, multivariate Earth Observation data, built on xarray
* [reverse-geocoder](https://github.com/thampiman/reverse-geocoder) -> a fast, offline reverse geocoder in Python
* [MuseoToolBox](https://github.com/nkarasiak/MuseoToolBox) -> a python library to simplify the use of raster/vector, especially for machine learning and remote sensing
* [py6s](https://py6s.readthedocs.io/en/latest/) -> an interface to the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) atmospheric Radiative Transfer Model
* [timvt](https://github.com/developmentseed/timvt) -> PostGIS based Vector Tile server built on top of the modern and fast FastAPI framework
* [titiler](https://github.com/developmentseed/titiler) -> A dynamic Web Map tile server using FastAPI
* [BRAILS](https://github.com/NHERI-SimCenter/BRAILS) -> an AI-based pipeline for city-scale building information modelling (BIM)
* [color-thief-py](https://github.com/fengsp/color-thief-py) -> Grabs the dominant color or a representative color palette from an image
* [force](https://github.com/davidfrantz/force) -> an all-in-one processing engine for medium-resolution Earth Observation image archives
* [mapwarper](https://github.com/timwaters/mapwarper) -> an open source map geo-rectification, warping and georeferencing application
* [sarpy](https://github.com/ngageoint/sarpy) -> A basic Python library to demonstrate reading, writing, display, and simple processing of complex SAR data using the NGA SICD standard
* [buzzard](https://github.com/earthcube-lab/buzzard) -> Advanced raster and geometry manipulations
* [sentinel1denoised](https://github.com/nansencenter/sentinel1denoised) -> Thermal noise subtraction, scalloping correction, angular correction
* [RStoolbox](https://github.com/bleutner/RStoolbox) -> Remote Sensing Data Analysis in R
* [kart](https://github.com/koordinates/kart) -> Distributed version-control for geospatial and tabular data
* [picogeojson](https://github.com/fortyninemaps/picogeojson) -> a Python library for reading, writing, and working with GeoJSON
* [shareloc](https://github.com/CNES/shareloc) -> a simple remote sensing geometric library, to perform image coordinates projections between sensor and ground and vice versa
* [geoblaze](https://github.com/GeoTIFF/geoblaze) -> Blazing Fast JavaScript Raster Processing Engine
* [nasa-wildfires](https://github.com/datadesk/nasa-wildfires) -> Download wildfire hotspots detected by NASA satellites and the Fire Information for Resource Management System (FIRMS)
* [SSGP-toolbox](https://github.com/Dreamlone/SSGP-toolbox) -> Simple Spatial Gapfilling Processor. Toolbox for filling gaps in spatial datasets
* [imgreg2D](https://github.com/BrancoLab/imgreg2D) -> 2D image registration in python, using napari
* [georust](https://github.com/georust) -> A collection of geospatial tools and libraries written in Rust
* [DataPillager](https://github.com/gdherbert/DataPillager) -> Download data from Esri REST service
* [litexplore](https://github.com/litements/litexplore) -> a Python web app that lets you explore remote SQLite databases over SSH connections
* [tifeatures](https://github.com/developmentseed/tifeatures) -> Simple and Fast Geospatial Features API for PostGIS
* [pyroSAR](https://github.com/johntruckenbrodt/pyroSAR) -> framework for large-scale SAR satellite data processing
* [S1_NRB](https://github.com/SAR-ARD/S1_NRB) -> A prototype processor for the Sentinel-1 Normalised Radar Backscatter product
* [AGBench](https://github.com/gyrrei/AGBench) -> a Python library that benchmarks satellite-based aboveground biomass or carbon estimate maps
* [mbtiles-s3-server](https://github.com/uktrade/mbtiles-s3-server) -> Python server to on-the-fly extract and serve vector tiles from an mbtiles file on S3
* [matico](https://github.com/Matico-Platform/matico) -> a set of tools and services that allow users to manage geospatial datasets, build APIs that use those datasets and full geospatial applications with little to no code
* [gmtsar](https://github.com/mobigroup/gmtsar) -> easy and fast satellite interferometry (InSAR) processing

## Low level numerical & data formats
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391). Checkout [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet) for tiled map plotting and [sklearn-xarray](https://github.com/phausamann/sklearn-xarray) for metadata-aware machine learning. Publish Xarray Datasets via a REST API uisng [xpublish](https://github.com/xarray-contrib/xpublish)
* [wxee](https://github.com/aazuspan/wxee) -> Export data from GEE to xarray using wxee then train with pytorch or tensorflow models. Useful since GEE only suports tfrecord export natively
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [xarray-beam](https://github.com/google/xarray-beam) -> Distributed Xarray with Apache Beam by Google
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates. [Create Land Use Classification using Geowombat & Sklearn](https://pygis.io/docs/f_rs_ml_predict.html)
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy
* [geoparquet](https://github.com/opengeospatial/geoparquet) -> Specification for storing geospatial vector data (point, line, polygon) in Parquet
* [TFRecord reader for PyTorch](https://github.com/vahidk/tfrecord)

## Image processing, handling, manipulation
* [Pillow is the Python Imaging Library](https://pillow.readthedocs.io/en/stable/) -> this will be your go-to package for image manipulation in python
* [opencv-python](https://github.com/opencv/opencv-python) is pre-built CPU-only OpenCV packages for Python
* [kornia](https://github.com/kornia/kornia) is a differentiable computer vision library for PyTorch, like openCV but on the GPU. Perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.
* [tifffile](https://github.com/cgohlke/tifffile) -> Read and write TIFF files
* [xtiff](https://github.com/BodenmillerGroup/xtiff) -> A small Python 3 library for writing multi-channel TIFF stacks
* [geotiff](https://github.com/Open-Source-Agriculture/geotiff) -> A noGDAL tool for reading and writing geotiff files
* [geolabel-maker](https://github.com/makinacorpus/geolabel-maker) -> combine satellite or aerial imagery with vector spatial data to create your own ground-truth dataset in the COCO format for deep-learning models
* [imagehash](https://github.com/JohannesBuchner/imagehash) -> Image hashes tell whether two images look nearly identical
* [fake-geo-images](https://github.com/up42/fake-geo-images) -> A module to programmatically create geotiff images which can be used for unit tests
* [imagededup](https://github.com/idealo/imagededup) -> Finding duplicate images made easy! Uses perceptual hashing
* [duplicate-img-detection](https://github.com/mattpodolak/duplicate-img-detection) -> A basic duplicate image detection service using perceptual image hash functions and nearest neighbor search, implemented using faiss, fastapi, and imagehash
* [rmstripes](https://github.com/DHI-GRAS/rmstripes) -> Remove stripes from images with a combined wavelet/FFT approach
* [activeloopai Hub](https://github.com/activeloopai/hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Scalable data pipelines.
* [sewar](https://github.com/andrewekhalel/sewar) -> All image quality metrics you need in one package
* [Satellite imagery label tool](https://github.com/calebrob6/labeling-tool) -> provides an easy way to collect a random sample of labels over a given scene of satellite imagery
* [Missing-Pixel-Filler](https://github.com/spaceml-org/Missing-Pixel-Filler) -> given images that may contain missing data regions (like satellite imagery with swath gaps), returns these images with the regions filled
* [color_range_filter](https://github.com/developmentseed/color_range_filter) -> a script that allows us to find range of colors in images using openCV, and then convert them into geo vectors
* [eo4ai](https://github.com/ESA-PhiLab/eo4ai) -> easy-to-use tools for preprocessing datasets for image segmentation tasks in Earth Observation
* [rasterix](https://github.com/mogasw/rasterix) -> a cross-platform utility built around the GDAL library and the Qt framework designed to process geospatial raster data
* [datumaro](https://github.com/openvinotoolkit/datumaro) -> Dataset Management Framework, a Python library and a CLI tool to build, analyze and manage Computer Vision datasets
* [sentinelPot](https://github.com/LLeiSong/sentinelPot) -> a python package to preprocess Sentinel 1&2 imagery
* [ImageAnalysis](https://github.com/UASLab/ImageAnalysis) -> Aerial imagery analysis, processing, and presentation scripts.
* [rastertodataframe](https://github.com/mblackgeo/rastertodataframe) -> Convert any GDAL compatible raster to a Pandas DataFrame
* [yeoda](https://github.com/TUW-GEO/yeoda) -> provides lower and higher-level data cube classes to work with well-defined and structured earth observation data
* [tiles-to-tiff](https://github.com/jimutt/tiles-to-tiff) -> Python script for converting XYZ raster tiles for slippy maps to a georeferenced TIFF image
* [telluric](https://github.com/satellogic/telluric) -> a Python library to manage vector and raster geospatial data in an interactive and easy way
* [Sniffer](https://github.com/2320sharon/Sniffer) -> A python application for sorting through geospatial imagery
* [pyjeo](https://github.com/ec-jrc/jeolib-pyjeo) -> a library for image processing for geospatial data implemented in JRC Ispra, with [paper](https://www.mdpi.com/2220-9964/8/10/461)
* [vpv](https://github.com/kidanger/vpv) -> Image viewer designed for image processing experts
* [arop](https://github.com/george-silva/arop) -> Automated Registration and Orthorectification Package
* [satellite_image](https://github.com/dgketchum/satellite_image) -> Python package to process images from Landsat satellites and return geographic information, cloud mask, numpy array, geotiff
* [large_image](https://github.com/girder/large_image) -> Python modules to work with large multiresolution images
* [ResizeRight](https://github.com/assafshocher/ResizeRight) -> The correct way to resize images or tensors. For Numpy or Pytorch (differentiable)
* [pysat](https://github.com/pysat/pysat) -> a package providing a simple and flexible interface for downloading, loading, cleaning, managing, processing, and analyzing scientific measurements
* [plcompositor](https://github.com/planetlabs/plcompositor) -> c++ tool from Planet to create seamless and cloudless image mosaics from deep stacks of satellite imagery

## Image augmentation packages
Image augmentation is a technique used to expand a training dataset in order to improve ability of the model to generalise
* [AugLy](https://github.com/facebookresearch/AugLy) -> A data augmentations library for audio, image, text, and video. By Facebook
* [albumentations](https://github.com/albumentations-team/albumentations) -> Fast image augmentation library and an easy-to-use wrapper around other libraries
* [FoHIS](https://github.com/noahzn/FoHIS) -> Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
* [Kornia](https://kornia.readthedocs.io/en/latest/augmentation.html) provides augmentation on the GPU
* [toolbox by ming71](https://github.com/ming71/toolbox) -> various cv tools, such as label tools, data augmentation, label conversion, etc.
* [AstroAugmentations](https://github.com/mb010/AstroAugmentations) -> augmentations designed around astronomical instruments
* [Chessmix](https://github.com/matheusbarrosp/chessmix) -> data augmentation method for remote sensing semantic segmentation
* [satellite_object_augmentation](https://github.com/LanaLana/satellite_object_augmentation) -> Object-based augmentation for remote sensing images segmentation via CNN
* [hypernet](https://github.com/ESA-PhiLab/hypernet) -> hyperspectral data augmentation

## Image formats, data management and catalogues
* [GeoServer](http://geoserver.org/) -> an open source server for sharing geospatial data
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* https://terria.io/ for pretty catalogues
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be opened using xarray
* [TileDB](https://tiledb.com/) -> a 'Universal Data Engine' to store, analyze and share any data (beyond tables), with any API or tool (beyond SQL) at planet-scale (beyond clusters), open source and managed options.
* Read about [Serverless PostGIS on AWS Aurora](https://blog.addresscloud.com/serverless-postgis/)
* [Hub](https://github.com/activeloopai/Hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Read [Faster Machine Learning Using Hub by Activeloop: A code walkthrough of using the hub package for satellite imagery](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005)
* [A Comparison of Spatial Functions: PostGIS, Athena, PrestoDB, BigQuery vs RedShift](https://ual.sg/post/2020/07/03/a-comparison-of-spatial-functions-postgis-athena-prestodb-bigquery-vs-redshift/)
* [Unfolded Studio](https://studio.unfolded.ai/) -> visualization platform building on open source geospatial technologies including kepler.gl, deck.gl and H3. Processing is performed browser side enabling very responsive visualisations.
* [DroneDB](https://github.com/DroneDB/DroneDB) -> can index and extract useful information from the EXIF/XMP tags of aerial images to display things like image footprint, flight path and image GPS location
* [embeddinghub](https://github.com/featureform/embeddinghub) -> A vector database for machine learning embeddings
* [Resonant GeoData](https://github.com/ResonantGeoData/ResonantGeoData/) -> a Django application well suited for catalogging and searching annotated geospatial imagery, shapefiles, and full motion video datasets
* [fastdup](https://github.com/visualdatabase/fastdup) -> a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images
* [Nucleus](https://dashboard.scale.com/nucleus/) is a platform for image dataset management with advanced features including [autotagging](https://nucleus.scale.com/docs/introduction-to-autotag) and finding [instances with mismatched predictions & annotations](https://nucleus.scale.com/docs/find-inaccurate-predictions)

## Model tracking, versioning, specification & compilation
* [dvc](https://dvc.org/) -> a git extension to keep track of changes in data, source code, and ML models together
* [Weights and Biases](https://wandb.ai/) -> keep track of your ML projects. Log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues
* [geo-ml-model-catalog](https://github.com/radiantearth/geo-ml-model-catalog) -> provides a common metadata definition for ML models that operate on geospatial data
* [hummingbird](https://github.com/microsoft/hummingbird) ->  a library for compiling trained traditional ML models into tensor computations, e.g. scikit learn model to pytorch for fast inference on a GPU
* [deepchecks](https://github.com/deepchecks/deepchecks) -> Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort
* [pachyderm](https://www.pachyderm.com/) -> Data Versioning and Pipelines for MLOps. Read [Pachyderm + Label Studio](https://medium.com/pachyderm-data/pachyderm-label-studio-ecc09f1f9329) which discusses versioning and lineage of data annotations

## Graphing and visualisation
* [hvplot](https://hvplot.holoviz.org/) -> A high-level plotting API for the PyData ecosystem built on HoloViews. Allows overlaying data on map tiles, see [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727)
* [Pyviz](https://examples.pyviz.org/) examples include several interesting geospatial visualisations
* [napari](https://napari.org) -> napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools. Note that to view a 3GB COG I had to install the [napari-tifffile-reader](https://github.com/GenevieveBuckley/napari-tifffile-reader) plugin.
* [pixel-adjust](https://github.com/cisaacstern/pixel-adjust) -> Interactively select and adjust specific pixels or regions within a single-band raster. Built with rasterio, matplotlib, and panel.
* [Plotly Dash](https://plotly.com/dash/) can be used for making interactive dashboards
* [folium](https://python-visualization.github.io/folium/) -> a python wrapper to the excellent [leaflet.js](https://leafletjs.com/) which makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. Also checkout the [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) component for adding folium maps to your streamlit apps
* [ipyearth](https://github.com/davidbrochart/ipyearth) -> An IPython Widget for Earth Maps
* [geopandas-view](https://github.com/martinfleis/geopandas-view) -> Interactive exploration of GeoPandas GeoDataFrames
* [geogif](https://github.com/gjoseph92/geogif) -> Turn xarray timestacks into GIFs
* [leafmap](https://github.com/giswqs/leafmap) -> geospatial analysis and interactive mapping with minimal coding in a Jupyter environment
* [xmovie](https://github.com/jbusecke/xmovie) -> A simple way of creating movies from xarray objects
* [acquisition-time](https://github.com/charlotte-pel/acquisition-time) -> Drawing (Satellite) acquisition dates in a timeline
* [splot](https://github.com/pysal/splot) -> Lightweight plotting for geospatial analysis in PySAL
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Tools to Design or Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)
* [AstronomicAL](https://github.com/grant-m-s/AstronomicAL) -> An interactive dashboard for visualisation, integration and classification of data using Active Learning
* [pyodi](https://github.com/Gradiant/pyodi) -> A simple tool for explore your object detection dataset
* [Interactive-TSNE](https://github.com/spaceml-org/Interactive-TSNE) -> a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability
* [fastgradio](https://github.com/aliabd/fastgradio) -> Build fast gradio demos of fastai learners
* [pysheds](https://github.com/mdbartos/pysheds) -> Simple and fast watershed delineation in python
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [cartoframes](https://github.com/CartoDB/cartoframes) -> integrate CARTO maps, analysis, and data services into data science workflows
* [datashader](https://datashader.org/) -> create meaningful representations of large datasets quickly and flexibly. Read [Creating Visual Narratives from Geospatial Data Using Open-Source Technology Maxar blog post](https://blog.maxar.com/tech-and-tradecraft/2021/creating-visual-narratives-from-geospatial-data-using-open-source-technology)
* [Kaleido](https://github.com/plotly/Kaleido) -> Fast static image export for web-based visualization libraries with zero dependencies
* [Embedding Projector in Wandb](https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector) -> allows users to plot multi-dimensional embeddings on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE
* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) -> Latex code for making neural networks diagrams
* [Damage Assessment Visualizer](https://github.com/microsoft/Nonprofits/tree/master/Damage%20Assessment%20Visualizer) -> leverages satellite imagery from a disaster region to visualize conditions of building and structures before and after a disaster
* [NN-SVG](https://github.com/alexlenail/NN-SVG) -> is a tool for creating Neural Network (NN) architecture drawings parametrically rather than manually
* [bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer) -> Make drawing and labeling bounding boxes easy as cake
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [EOmaps](https://github.com/raphaelquast/EOmaps) -> A library to create interactive maps of geographical datasets
* [H3-Pandas](https://github.com/DahnJ/H3-Pandas) -> Integrates H3 with GeoPandas and Pandas
* [gmplot](https://github.com/gmplot/gmplot) -> a matplotlib-like interface to render all the data you'd like on top of Google Maps
* [NPYViewer](https://github.com/csmailis/NPYViewer) ->  a simple GUI tool that provides multiple ways to view `.npy` files containing 2D NumPy Arrays
* [pyGEOVis](https://github.com/geoyee/pyGEOVis) -> Visualize geo-tiff/json based on folium
* [bokeh-tiler](https://github.com/avanetten/bokeh-tiler) -> Tile large geospatial images for use in Bokeh. Read [Serving up SpaceNet Imagery for Bokeh](https://medium.com/geodesic/serving-up-spacenet-imagery-for-bokeh-e85b8fffe05)
* [torchshow](https://github.com/xwying/torchshow) -> Visualize PyTorch tensor in one-line of code
* [pixels](https://github.com/jwasilgeo/pixels) -> Mapping and charting pixels from remote sensing Earth observation data with JavaScript
* [MulimgViewer](https://github.com/nachifur/MulimgViewer) -> a multi-image viewer that can open multiple images in one interface
* [cnn-explainer](https://github.com/poloclub/cnn-explainer) -> Learning Convolutional Neural Networks with Interactive Visualization
* [Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map](https://github.com/royalosyin/Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map)
* [shapefile2gif](https://github.com/johannesuhl/shapefile2gif) -> Given a shapefile with time-annotated vector objects (e.g., building footprints + construction year), this script will automatically create an animated GIF illustrating the dynamics for a user-specified period of time
* [insat3d_imagen](https://github.com/rupeshs/insat3d_imagen) -> Processes INSAT HDF file and generates satellite images
* [pygieons](https://github.com/pygieons/pygieons) -> A simple package to visualize and keep track of GIS and Earth Observation libraries in Python
* [regionmask](https://github.com/regionmask/regionmask) -> Create masks of geographical regions for arbitrary longitude and latitude grids
* [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

## Algorithms
* [WaterDetect](https://github.com/cordmaur/WaterDetect) -> an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.
* [GatorSense Hyperspectral Image Analysis Toolkit](https://github.com/GatorSense/hsi_toolkit_py) -> This repo contains algorithms for Anomaly Detectors, Classifiers, Dimensionality Reduction, Endmember Extraction, Signature Detectors, Spectral Indices
* [detectree](https://github.com/martibosch/detectree) -> Tree detection from aerial imagery
* [pylandstats](https://github.com/martibosch/pylandstats) -> compute landscape metrics
* [dg-calibration](https://github.com/DHI-GRAS/dg-calibration) -> Coefficients and functions for calibrating DigitalGlobe imagery
* [python-fmask](https://github.com/ubarsc/python-fmask) -> Implementation in Python of the cloud and shadow algorithms known collectively as Fmask
* [pyshepseg](https://github.com/ubarsc/pyshepseg) -> Python implementation of image segmentation algorithm of Shepherd et al (2019) Operational Large-Scale Segmentation of Imagery Based on Iterative Elimination.
* [Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images) -> shadow detection and correction algorithm
* [faiss](https://github.com/facebookresearch/faiss) -> A library for efficient similarity search and clustering of dense vectors, e.g. image embeddings
* [awesome-spectral-indices](https://github.com/davemlz/awesome-spectral-indices) -> A ready-to-use curated list of Spectral Indices for Remote Sensing applications
* [urban-footprinter](https://github.com/martibosch/urban-footprinter) -> A convolution-based approach to detect urban extents from raster datasets
* [ocean_color](https://github.com/marrs-lab/ocean_color) -> Tools and algorithms for drone and satellite based ocean color science
* [poliastro](https://github.com/poliastro/poliastro) -> pure Python library for interactive Astrodynamics and Orbital Mechanics, with a focus on ease of use, speed, and quick visualization
* [acolite](https://github.com/acolite/acolite) -> generic atmospheric correction module
* [pmapper](https://github.com/nasa-jpl/pmapper) -> a super-resolution and deconvolution toolkit for python. PMAP stands for Poisson Maximum A-Posteriori, a highly flexible and adaptable algorithm for these problems
* [pylandtemp](https://github.com/pylandtemp/pylandtemp) -> Algorithms for computing global land surface temperature and emissivity from NASA's Landsat satellite images with Python
* [sarsen](https://github.com/bopen/sarsen) -> Algorithms and utilities for Synthetic Aperture Radar (SAR) sensors
* [sun-position](https://github.com/s-bear/sun-position) -> code for computing sun position
* [simple_ortho](https://github.com/dugalh/simple_ortho) -> Fast and simple orthorectification of images with known DEM and camera model
* [imageResolution](https://github.com/geojames/imageResolution) -> Simple spatial resolution calculator for nadir & oblique aerial imagery
* [Spectral-Clustering](https://github.com/zhangyk8/Spectral-Clustering) -> normalized and unnormalized spectral clustering algorithms
* [Fogpy](https://github.com/pytroll/fogpy) -> nowcasting of fog and low stratus clouds
* [orthorectification](https://github.com/mpfaffenberger/orthorectification) -> Orthorectification in Python. Note that all of this functionality already exists in libraries like GDAL and others. The goal of this codebase was to present and deep dive into these subroutines
* [Flood-Severity-Estimation](https://github.com/jorgemspereira/Flood-Severity-Estimation) -> estimate the height of the water in geo-referenced photos that depict floods using DEMs from JAXA
* [coastline-extraction](https://github.com/Ricardo-C-Oliveira/coastline-extraction) -> Methods to identify and extract coastline from remote sensed data
* [Near real-time shadow detection and removal in remote sensing imagery application](https://github.com/BIT-zhwang/remote-sensing-image-shadow-detection-and-removal)
* [image-registration](https://github.com/satish1901/image-registration) -> using Point Feature Detection, Normalized DLT, RANSAC & Image Warping
* [pyTSEB](https://github.com/hectornieto/pyTSEB) -> A python Two Source Energy Balance model for estimation of evapotranspiration with remote sensing data
* [libpredict](https://github.com/la1k/libpredict) -> satellite orbit prediction library
* [GOTCHA](https://github.com/jveitchmichaelis/gotcha) -> Command line implementation of the GOTCHA stereo matching algorithm
* [SREM](https://github.com/oyam/srem) -> A Simplified and Robust Surface Reflectance Estimation Method for Satellite Imagery
* [kaizen](https://github.com/fuzailpalnak/kaizen) -> A library to map match and help tackle the problem of overlapping/intersecting road and building footprint that arises in the process of map making
* [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope) -> Batch shoreline extraction toolkit for PlanetScope Dove satellite imagery
* [mappymatch](https://github.com/NREL/mappymatch) -> Pure-python package for map matching

## GDAL & Rasterio
So improtant this pair gets their own section. GDAL is THE command line tool for reading and writing raster and vector geospatial data formats. If you are using python you will probably want to use Rasterio which provides a pythonic wrapper for GDAL
* [GDAL](https://gdal.org) and [on twitter](https://twitter.com/gdaltips)
* GDAL is a dependency of Rasterio and can be difficult to build and install. I recommend using conda, brew (on OSX) or docker in these situations
* GDAL docker quickstart: `docker pull osgeo/gdal` then `docker run --rm -v $(pwd):/data/ osgeo/gdal gdalinfo /data/cog.tiff`
* [Even Rouault](https://github.com/rouault) maintains GDAL, please consider [sponsoring him](https://github.com/sponsors/rouault)
* [Rasterio](https://rasterio.readthedocs.io/en/latest/) -> reads and writes GeoTIFF and other raster formats and provides a Python API based on Numpy N-dimensional arrays and GeoJSON. There are a variety of plugins that extend Rasterio functionality.
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [rioxarray](https://github.com/corteva/rioxarray) -> geospatial xarray extension powered by rasterio
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [godal](https://github.com/airbusgeo/godal) -> golang wrapper for GDAL
* [Write rasterio to xarray](https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py)
* [Loam: A Client-Side GDAL Wrapper for Javascript](https://github.com/azavea/loam)
* [Short list of useful GDAL commands](https://github.com/MaxLenormand/Data-Science-for-Remote-Sensing) while working in data science for remote sensing
* [gdal-segment](https://github.com/cbalint13/gdal-segment) -> implements various segmentation algorithms over raster images
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch
* [gdal2tiles](https://github.com/tehamalab/gdal2tiles) -> A python library for generating map tiles based on gdal2tiles.py from GDAL project
* [gdal3.js](https://github.com/bugra9/gdal3.js) -> Convert raster and vector geospatial data to various formats and coordinate systems entirely in the browser

## Cloud Optimised GeoTiff (COG)
A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF that supports HTTP range requests, enabling downloading of specific tiles rather than the full file. COG generally work normally in GIS software such as QGIS, but are larger than regular GeoTIFFs
* https://www.cogeo.org/
* [cog-best-practices](https://github.com/pangeo-data/cog-best-practices)
* [COGs in production](https://sean-rennie.medium.com/cogs-in-production-e9a42c7f54e4)
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [aiocogeo](https://github.com/geospatial-jeff/aiocogeo) -> Asynchronous cogeotiff reader (python asyncio)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article Cloud Native Geoprocessing of Earth Observation Satellite Data with Pangeo](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Working with COGS and STAC in python using geemap](https://geemap.org/notebooks/44_cog_stac/)
* [Load, Experiment, and Download Cloud Optimized Geotiffs (COG) using Python with Google Colab](https://towardsdatascience.com/access-satellite-imagery-with-aws-and-google-colab-4660178444f5) -> short read which covers finding COGS, opening with Rasterio and doing some basic manipulations, all in a Colab Notebook.
* [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727) -> local COG from public AWS bucket, open with rioxarray, visualise with [hvplot](https://hvplot.holoviz.org/). See [the Jupyter notebook](https://nbviewer.jupyter.org/gist/rsignell-usgs/9657896371bb4f38437505146555264c)
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [cogbeam](https://github.com/GoogleCloudPlatform/cogbeam) -> a python based Apache Beam pipeline, optimized for Google Cloud Dataflow, which aims to expedite the conversion of traditional GeoTIFFs into COGs
* [cogserver](https://github.com/rouault/cogserver) -> Expose a GDAL file as a HTTP accessible on-the-fly COG
* [Displaying a gridded dataset on a web-based map - Step by step guide for displaying large GeoTIFFs, using Holoviews, Bokeh, and Datashader](https://towardsdatascience.com/displaying-a-gridded-dataset-on-a-web-based-map-ad6bbe90247f)
* [cog_worker](https://github.com/Vizzuality/cog_worker) -> Scalable arbitrary analysis on COGs

## SpatioTemporal Asset Catalog specification (STAC)
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered.
* Spec at https://github.com/radiantearth/stac-spec
* [STAC 1.0.0: The State of the STAC Software Ecosystem](https://medium.com/radiant-earth-insights/stac-1-0-0-software-ecosystem-updates-da4e800a4973)
* [Planet Disaster Data catalogue](https://planet.stac.cloud/) has the [catalogue source on Github](https://github.com/cholmes/pdd-stac) and uses the [stac-browser](https://github.com/radiantearth/stac-browser)
* [Getting Started with STAC APIs](https://www.azavea.com/blog/2021/04/05/getting-started-with-stac-apis/) intro article
* [SpatioTemporal Asset Catalog API specification](https://github.com/radiantearth/stac-api-spec) -> an API to make geospatial assets openly searchable and crawlable
* [stacindex](https://stacindex.org/) -> STAC Catalogs, Collections, APIs, Software and Tools
* Several useful repos on https://github.com/sat-utils
* [Intake-STAC](https://github.com/intake/intake-stac) -> Intake-STAC provides an opinionated way for users to load Assets from STAC catalogs into the scientific Python ecosystem. It uses the intake-xarray plugin and supports several file formats including GeoTIFF, netCDF, GRIB, and OpenDAP.
* [sat-utils/sat-search](https://github.com/sat-utils/sat-search) -> Sat-search is a Python 3 library and a command line tool for discovering and downloading publicly available satellite imagery using STAC compliant API
* [franklin](https://github.com/azavea/franklin) -> A STAC/OGC API Features Web Service focused on ease-of-use for end-users.
* [stacframes](https://github.com/azavea/stacframes) -> A Python library for working with STAC Catalogs via Pandas DataFrames
* [sat-api-pg](https://github.com/developmentseed/sat-api-pg) -> A Postgres backed STAC API
* [stactools](https://github.com/stac-utils/stactools) -> Command line utility and Python library for STAC
* [pystac](https://github.com/stac-utils/pystac) -> Python library for working with any STAC Catalog
* [STAC Examples for Nightlights data](https://github.com/developmentseed/nightlights_stac_examples) -> minimal example STAC implementation for the [Light Every Night](https://registry.opendata.aws/wb-light-every-night/) dataset of all VIIRS DNB and DMSP-OLS nighttime satellite data
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [stac-fastapi](https://github.com/stac-utils/stac-fastapi) -> STAC API implementation with FastAPI
* [stac-fastapi-elasticsearch](https://github.com/stac-utils/stac-fastapi-elasticsearch) -> Elasticsearch backend for stac-fastapi
* [ml-aoi](https://github.com/stac-extensions/ml-aoi) -> An Item and Collection extension to provide labeled training data for machine learning models
* Discoverable and Reusable ML Workflows for Earth Observation -> [part 1](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-1-e198507b5eaa) and [part 2](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-2-ebe2b4812d5a) with the Geospatial Machine Learning Model Catalog (GMLMC)
* [eoAPI](https://github.com/developmentseed/eoAPI) -> Earth Observation API with STAC + dynamic Raster/Vector Tiler
* [stac-nb](https://github.com/darrenwiens/stac-nb) -> STAC in Jupyter Notebooks
* [xstac](https://github.com/TomAugspurger/xstac) -> Generate STAC Collections from xarray datasets
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [cirrus-geo](https://github.com/cirrus-geo/cirrus-geo) -> a STAC-based processing pipeline
* [stac-interactive-search](https://github.com/calebrob6/stac-interactive-search) -> A simple (browser based) UI for searching STAC APIs
* [easystac](https://github.com/cloudsen12/easystac) -> A Python package for simple STAC queries
* [stacmap](https://github.com/aazuspan/stacmap) -> Explore STAC items with an interactive map
* [odc-stac](https://github.com/opendatacube/odc-stac) -> Load STAC items into xarray Datasets. Process locally or distribute data loading and computation with Dask.
* [AWS Lambda SenCloud Monitoring](https://github.com/ahuarte47/aws-sencloud-monitoring) -> keep up-to-date your own derived data from the Sentinel-2 COG imagery archive using AWS lambda
* [stac-geoparquet](https://github.com/TomAugspurger/stac-geoparquet) -> Convert STAC items to geoparquet

## OpenStreetMap
[OpenStreetMap](https://www.openstreetmap.org/) (OSM) is a map of the world, created by people like you and free to use under an open license. Quite a few publications use OSM data for annotations & ground truth. Note that the data is created by volunteers and the quality can be variable
* [osmnx](https://github.com/gboeing/osmnx) -> Retrieve, model, analyze, and visualize data from OpenStreetMap
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training. This should be used instead of the deprecated [skynet-data](https://github.com/developmentseed/skynet-data)
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”
* [Import OpenStreetMap data into Unreal Engine 4](https://github.com/ue4plugins/StreetMap)
* [OSMDeepOD](https://github.com/geometalab/OSMDeepOD) ->  perform object detection with retinanet
* [Match Bing Map Aerial Imagery with OpenStreetMap roads](https://github.com/whywww/Aerial-Imagery-and-OpenStreetMap-Retrieval)
* [Computer Vision With OpenStreetMap and SpaceNet — A Comparison](https://medium.com/the-downlinq/computer-vision-with-openstreetmap-and-spacenet-a-comparison-cc70353d0ace)
* [url-map](https://simonwillison.net/2022/Jun/12/url-map/) -> A tiny web app to create images from OpenStreetMap maps
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data
* [baremaps](https://github.com/baremaps/baremaps) -> Create custom vector tiles from OpenStreetMap and other data sources with Postgis and Java.
* [osm2streets](https://github.com/a-b-street/osm2streets) -> Convert OSM to street networks with detailed geometry

## QGIS
A popular open source alternative to ArcGIS, QGIS is a desktop appication written in python and extended with plugins which are essentially python scripts
* [QGIS](https://qgis.org/en/site/)
* Create, edit, visualise, analyse and publish geospatial information. Open source alternative to ArcGIS.
* [Python scripting](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console)
* Create your own plugins using the [QGIS Plugin Builder](http://g-sherman.github.io/Qgis-Plugin-Builder/)
* [DeepLearningTools plugin](https://plugins.qgis.org/plugins/DeepLearningTools/) -> aid training Deep Learning Models
* [Mapflow.ai plugin](https://www.gislounge.com/run-ai-mapping-in-qgis-over-high-resolution-satellite-imagery/) -> various models to extract building footprints etc from Maxar imagery
* [dzetsaka plugin](https://github.com/nkarasiak/dzetsaka) -> classify different kind of vegetation
* [Coregistration-Qgis-processing](https://github.com/SMByC/Coregistration-Qgis-processing) -> Qgis processing plugin for image co-registration; projection and pixel alignment based on a target image, uses Arosics
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [buildseg](https://github.com/deepbands/buildseg) -> a building extraction plugin of QGIS based on ONNX
* [deep-learning-datasets-maker](https://github.com/deepbands/deep-learning-datasets-maker) -> a QGIS plugin to make datasets creation easier for raster and vector data
* [Modzy-QGIS-Plugin](https://github.com/modzy/Modzy-QGIS-Plugin) -> demos Vehicle Detection model
* [kart](https://plugins.qgis.org/plugins/kart/) -> provides modern, open source, distributed version-control for geospatial and tabular datasets
* [Plugin for Landcover Classification](https://github.com/atishayjn/QGIS-Plugin) -> capable of implementing machine learning algorithms such as Random forest, SVM and CNN algorithms such as UNET through a simple GUI framework.
* [pg_tileserv])(https://github.com/CrunchyData/pg_tileserv) -> A very thin PostGIS-only tile server in Go. Takes in HTTP tile requests, executes SQL, returns MVT tiles.
* [pg_featureserv](https://github.com/CrunchyData/pg_featureserv) -> Lightweight RESTful Geospatial Feature Server for PostGIS in Go
* [osm-instance-segmentation](https://github.com/mnboos/osm-instance-segmentation) -> QGIS plugin for finding changes in vector data from orthophotos (i.e. aerial imagery) using tensorflow
* [Semi-Automatic Classification Plugin](https://github.com/semiautomaticgit/SemiAutomaticClassificationPlugin) -> supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images
* [chippy-checker-editor](https://github.com/devglobalpartners/chippy-checker-editor) -> QGIS plugin for viewing and editing labeled remote sensing images
* [qgis-plugin-deepness](https://github.com/PUTvision/qgis-plugin-deepness) -> Plugin for neural network inference in QGIS: segmentation, regression and detection

## Parallel procesing with Dask
Dask provides advanced parallelism and distributed out-of-core computation with a `dask.dataframe` module designed to scale pandas.
* [Dask](https://docs.dask.org/en/latest/) works with your favorite PyData libraries to provide performance at scale for the tools you love
* [Coiled](https://coiled.io) is a managed Dask service. Get started by reading [Democratizing Satellite Imagery Analysis with Dask](https://coiled.io/blog/democratizing-satellite-imagery-analysis-with-dask/)
* [Dask with PyTorch for large scale image analysis](https://blog.dask.org/2021/03/29/apply-pretrained-pytorch-model)
* [dask-geopandas](https://github.com/geopandas/dask-geopandas) -> offers geospatial capabilities of GeoPandas backed by Dask
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [dask-geomodeling](https://github.com/nens/dask-geomodeling) -> On-the-fly operations on geographical maps
* [dask-image](https://github.com/dask/dask-image) -> many SciPy ndimage functions implemented
* [Detecting Green Roofs in Toronto](https://toarches.medium.com/geospatial-big-data-processing-with-python-detecting-green-roofs-in-toronto-bd7bf08900f2) -> compares deep learning (Mask R-CNN & fast.ai) and classical approach using NDVI scaled on Dask
* [Analyze terabyte-scale geospatial datasets with Dask and Jupyter on AWS](https://aws.amazon.com/blogs/publicsector/analyze-terabyte-scale-geospatial-datasets-with-dask-and-jupyter-on-aws/)
* [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo) -> A change detection demo for the Austin area using a pre-trained PyTorch model scaled with Dask on Planet imagery

## Web apps
Flask is often used to serve up a simple web app based on templated HTML files
* [FastMap](https://github.com/butlerbt/FastMap) -> Flask deployment of deep learning model performing segmentation task on aerial imagery building footprints
* [Querying Postgres with Python Fastapi Backend and Leaflet-Geoman Frontend](https://geo.rocks/post/leaflet-geoman-fastapi-postgis/)
* [cropcircles](https://github.com/doersino/cropcircles) -> a purely-client-side web app originally designed for accurately cropping circular center pivot irrigation fields from aerial imagery
* [django-large-image](https://github.com/ResonantGeoData/django-large-image) -> Django endpoints for working with large images for tile serving
* [Earth Classification API](https://github.com/conlamon/satellite-classification-flask-api) -> Flask based app that serves a CNN model and interfaces with a React and Leaflet front-end
* [Demo flask map app](https://github.com/kdmayer/flask_tutorial) -> Building Python-based, database-driven web applications (with maps!) using Flask, SQLite, SQLAlchemy and MapBox
* [Building a Web App for Instance Segmentation using Docker, Flask and Detectron2](https://towardsdatascience.com/instance-segmentation-web-app-63016b8ed4ae)
* [greppo](https://github.com/greppo-io/greppo) -> Build & deploy geospatial applications quick and easy. Read [Build a geospatial dashboard in Python using Greppo](https://towardsdatascience.com/build-a-geospatial-dashboard-in-python-using-greppo-60aff44ba6c9)
* [localtileserver](https://github.com/banesullivan/localtileserver) -> image tile server for viewing geospatial rasters with ipyleaflet, folium, or CesiumJS locally in Jupyter or remotely in Flask applications. Checkout [bokeh-tiler](https://github.com/avanetten/bokeh-tiler) 
* [flask-geocoding-webapp](https://github.com/mblackgeo/flask-geocoding-webapp) -> A quick example Flask application for geocoding and rendering a webmap using Folium/Leaflet
* [flask-vector-tiles](https://github.com/mblackgeo/flask-vector-tiles) -> A simple Flask/leaflet based webapp for rendering vector tiles from PostGIS
* [Crash Severity Prediction](https://github.com/SoySauceNZ/web-app) -> using CAS Open Data and Maxar Satellite Imagery, React app
* [wildfire-detection-from-satellite-images-ml](https://github.com/shrey24/wildfire-detection-from-satellite-images-ml) -> simple flask app for classification
* [SlumMappingViaRemoteSensingImagery](https://github.com/hamna-moieez/SlumMappingViaRemoteSensingImagery) -> learning slum segmentation and localization using satellite imagery and visualising on a flask app
* [cloud-removal-deploy](https://github.com/XavierJiezou/cloud-removal-deploy) -> flask app for cloud removal
* [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> research & web-service for clearcut detection

## Jupyter
The [Jupyter](https://jupyter.org/) Notebook is a web-based interactive computing platform. There are many extensions which make it a powerful environment for analysing satellite imagery
* [jupyterlite](https://jupyterlite.readthedocs.io/en/latest/) -> JupyterLite is a JupyterLab distribution that runs entirely in the browser
* [jupyter_compare_view](https://github.com/Octoframes/jupyter_compare_view) -> Blend Between Multiple Images
* [folium](https://python-visualization.github.io/folium/quickstart.html) -> display interactive maps in Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using jupyter notebooks
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) -> create custom image classification annotators within Jupyter notebooks
* [jupyter-innotater](https://github.com/ideonate/jupyter-innotater) -> Inline data annotator for Jupyter notebooks
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [pylabel](https://github.com/pylabel-project/pylabel) -> includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model
* [jupyterlab-s3-browser](https://github.com/IBM/jupyterlab-s3-browser) -> extension for browsing S3-compatible object storage
* [papermill](https://github.com/nteract/papermill) -> Parameterize, execute, and analyze notebooks
* [pretty-jupyter](https://github.com/JanPalasek/pretty-jupyter) -> Creates dynamic html report from jupyter notebook

## Streamlit
[Streamlit](https://streamlit.io/) is an awesome python framework for creating apps with python. Additionally they will host the apps free of charge. Here I list resources which are EO related. Note that a component is an addon which extends Streamlits basic functionality
* [cogviewer](https://github.com/mykolakozyr/cogviewer) -> Simple Cloud Optimized GeoTIFF viewer
* [cogcreator](https://github.com/mykolakozyr/cogcreator) -> Simple Cloud Optimized GeoTIFF Creator. Generates COG from GeoTIFF files.
* [cogvalidator](https://github.com/mykolakozyr/cogvalidator) -> Simple Cloud Optimized GeoTIFF validator
* [streamlit-image-comparison](https://github.com/fcakyon/streamlit-image-comparison) -> compare images with a slider. Used in [example-app-image-comparison](https://github.com/streamlit/example-app-image-comparison)
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) -> Streamlit Component for rendering Folium maps
* [streamlit-keplergl](https://github.com/chrieke/streamlit-keplergl) -> Streamlit component for rendering kepler.gl maps
* [streamlit-light-leaflet](https://github.com/andfanilo/streamlit-light-leaflet) -> Streamlit quick & dirty Leaflet component that sends back coordinates on map click
* [leafmap-streamlit](https://github.com/giswqs/leafmap-streamlit) -> various examples showing how to use streamlit to: create a 3D map using Kepler.gl, create a heat map, display a GeoJSON file on a map, and add a colorbar or change the basemap on a map
* [geemap-apps](https://github.com/giswqs/geemap-apps) -> build a multi-page Earth Engine App using streamlit and geemap
* [streamlit-geospatial](https://github.com/giswqs/streamlit-geospatial) -> A multi-page streamlit app for geospatial
* [geospatial-apps](https://github.com/giswqs/geospatial-apps) -> A collection of streamlit web apps for geospatial applications
* [BirdsPyView](https://github.com/rjtavares/BirdsPyView) -> convert images to top-down view and get coordinates of objects
* [Build a useful web application in Python: Geolocating Photos](https://medium.com/spatial-data-science/build-a-useful-web-application-in-python-geolocating-photos-186122de1968) -> Step by Step tutorial using Streamlit, Exif, and Pandas
* [Wild fire detection app](https://github.com/yueureka/WildFireDetection)
* [dvc-streamlit-example](https://github.com/sicara/dvc-streamlit-example) -> how dvc and streamlit can help track model performance during R&D exploration
* [stacdiscovery](https://github.com/mykolakozyr/stacdiscovery) -> Simple STAC Catalogs discovery tool
* [SARveillance](https://github.com/MJCruickshank/SARveillance) -> Sentinel-1 SAR time series analysis for OSINT use
* [streamlit-template](https://github.com/giswqs/streamlit-template) -> A streamlit app template for geospatial applications
* [streamlit-labelstudio](https://github.com/deneland/streamlit-labelstudio) -> A Streamlit component that provides an annotation interface using the LabelStudio Frontend
* [streamlit-img-label](https://github.com/lit26/streamlit-img-label) -> a graphical image annotation tool using streamlit. Annotations are saved as XML files in PASCAL VOC format
* [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) -> A secure authentication module to validate user credentials in a Streamlit application
* [prettymapp](https://github.com/chrieke/prettymapp) -> Create beautiful maps from OpenStreetMap data in a webapp
* [mapa-streamlit](https://github.com/fgebhart/mapa-streamlit) -> creating 3D-printable models of the earth surface based on mapa
* [BoulderAreaDetector](https://github.com/pszemraj/BoulderAreaDetector) -> CNN to classify whether a satellite image shows an area would be a good rock climbing spot or not, deployed to streamlit app
* [streamlit-remotetileserver](https://github.com/banesullivan/streamlit-remotetileserver) -> Easily visualize a remote raster given a URL and check if it is a valid Cloud Optimized GeoTiff (COG)
* [Streamlit_Image_Sorter](https://github.com/2320sharon/Streamlit_Image_Sorter) -> Generic Image Sorter Interface for Streamlit
* [Streamlit-Folium + Snowflake + OpenStreetMap](https://github.com/randyzwitch/streamlit-folium-snowflake-openstreetmap) -> demonstrates the power of Snowflake Geospatial data types and queries combined with Streamlit
* [observing-earth-from-space-with-streamlit](https://blog.streamlit.io/observing-earth-from-space-with-streamlit/) -> blog post on the [SatSchool](https://github.com/Spiruel/SatSchool) app
* [vector-validator](https://github.com/chrieke/vector-validator) -> Webapp that validates and automatically fixes your geospatial vector data

## Julia language
[Julia](https://julialang.org/) looks and feels a lot like Python, but can be much faster. Julia can call Python, C, and Fortran libraries and is capabale of C/Fortran speeds. Julia can be used in the familiar Jupyterlab notebook environment
* [Why you should invest in Julia now, as a Data Scientist](https://medium.com/@logankilpatrick/why-you-should-invest-in-julia-now-as-a-data-scientist-30dc346d62e4)
* [eBook: Introduction to Datascience with Julia](https://datascience-book.gitlab.io/)
* [FastAI.jl](https://github.com/FluxML/FastAI.jl) -> Repository of best practices for deep learning in Julia, inspired by fastai
* [Flux.jl](https://github.com/FluxML/Flux.jl) -> the ML library that doesn't make you tensor. Checkout [The Deep Learning with Julia book](https://github.com/logankilpatrick/DeepLearningWithJulia)
* [GDAL.jl](https://github.com/JuliaGeo/GDAL.jl) -> Thin Julia wrapper for GDAL
* [GeoInterface.jl](https://github.com/JuliaGeo/GeoInterface.jl) -> A Julia Protocol for Geospatial Data
* [GeoJSON.jl](https://github.com/JuliaGeo/GeoJSON.jl) -> Utilities for working with GeoJSON data
* [JuliaImages: image processing and machine vision for Julia](https://juliaimages.org/stable/)
* [Julia_Geospatial](https://github.com/acgeospatial/Julia_Geospatial) -> Examples for a blog series on Geospatial Julia using ArchGDAL
* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) -> A Julia machine learning framework
* [Proj4.jl](https://github.com/JuliaGeo/Proj4.jl) -> Julia wrapper for the PROJ cartographic projections library
* [Rasters.jl](https://github.com/rafaqz/Rasters.jl) -> types and methods for reading, writing and manipulating rasterized spatial data including GeoTIFF and NetCDF
* [RemoteS.jl](https://github.com/GenericMappingTools/RemoteS.jl) -> Remote sensing data processing
* [SatelliteToolbox.jl](https://github.com/JuliaSpace/SatelliteToolbox.jl) -> This package contains several functions to build simulations related with satellites
* [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl) -> a satellite dynamics modeling package
* [Sentinel.jl](https://github.com/mhudecheck/Sentinel.jl) -> library for processing ESA Sentinel 2 satellite data
* [DeepSat-Kaggle](https://github.com/athulsudheesh/DeepSat-Kaggle) -> uses Julia

# Model deployment
This section discusses how to get a trained machine learning & specifically deep learning model into production. For an overview on serving deep learning models checkout [Practical-Deep-Learning-on-the-Cloud](https://github.com/PacktPublishing/-Practical-Deep-Learning-on-the-Cloud). There are many options if you are happy to dedicate a server, although you may want a GPU for batch processing. For serverless use AWS lambda.

## Rest API on dedicated server
A common approach to serving up deep learning model inference code is to wrap it in a rest API. The API can be implemented in python (flask or FastAPI), and hosted on a dedicated server e.g. EC2 instance. Note that making this a scalable solution will require significant experience.
* Basic API: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced API with request queuing: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/
* [How to make a geospatial Rest Api web service with Python, Flask and Shapely - Tutorial](https://hatarilabs.com/ih-en/how-to-make-a-geospatial-rest-api-web-service-with-python-flask-and-shapely-tutorial)
* [BMW-YOLOv4-Training-Automation](https://github.com/BMW-InnovationLab/BMW-YOLOv4-Training-Automation) -> project that demos training ML model via rest API
* [Basic REST API for a keras model using FastAPI](https://github.com/SoySauceNZ/backend)
* [NI4OS-RSSC](https://github.com/risojevicv/NI4OS-RSSC) -> Web Service for Remote Sensing Scene Classification (RS2C) using TensorFlow Serving and Flask
* [Sat2Graph Inference Server](https://github.com/songtaohe/Sat2Graph/tree/master/docker) -> API in Go for road segmentation model inferencing
* [API algorithm to apply object detection model to terabyte size satellite images with 800% better performance and 8 times less resources usage](https://github.com/orhannurkan/API-algorithm-for-terabyte-size-images-)
* [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> django backend
* [airbus-ship-detection](https://github.com/jancervenka/airbus-ship-detection) -> CNN with REST API

## Model serving with GRPC 
GPRC is a framework for implementing Remote Procedure Call (RPC) via HTTP/2. Developed and maintained mainly by Google, it is widely used in the industry. It allows two machines to communicate, similar to HTTP but with better syntax and performance.
* [deploy-models-with-grpc-pytorch-asyncio](https://github.com/FrancescoSaverioZuppichini/deploy-models-with-grpc-pytorch-asyncio)

## Framework specific model serving
If you are happy to live with some lock-in, these are good options:
* [Tensorflow serving](https://www.tensorflow.org/tfx/guide/serving) is limited to Tensorflow models
* [TensorRT_Inference](https://github.com/lzh420202/TensorRT_Inference) -> An oriented object detection framework based on TensorRT
* [Pytorch serve](https://github.com/pytorch/serve) is easy to use, limited to Pytorch models, and can be deployed via AWS Sagemaker, See [pl-lightning-torchserve-neptune-template](https://github.com/i008/pl-lightning-torchserve-neptune-template)
* [sagemaker-inference-toolkit](https://github.com/aws/sagemaker-inference-toolkit) -> Serve machine learning models within a Docker container using AWS SageMaker

## Framework agnostic model serving
* The [Triton Inference Server](https://github.com/triton-inference-server/server) provides an optimized cloud and edge inferencing solution. Read [CAPE Analytics Uses Computer Vision to Put Geospatial Data and Risk Information in Hands of Property Insurance Companies](https://blogs.nvidia.com/blog/2021/05/21/cape-analytics-computer-vision/)
* [RedisAI](https://oss.redis.com/redisai/) is a Redis module for executing Deep Learning/Machine Learning models and managing their data

## Using lambda functions - i.e. serverless
Using lambda functions allows inference without having to configure or manage the underlying infrastructure
* On AWS either use regular lambdas from AWS or [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
* [Object detection inference with AWS Lambda and IceVision (PyTorch)](https://laurenzstrothmann.com/object-detection-inference-aws-lambda-icevision) with [repo](https://github.com/2649/laurenzstrothmann)
* [Deploying PyTorch on AWS Lambda](https://segments.ai/blog/pytorch-on-lambda)
* [Example deployment behind an API Gateway Proxy](https://github.com/philschmid/cdk-samples/tree/master/sagemaker-serverless-huggingface-endpoint)

## Inferencing on large images
Models are typically trained and inferenced on relatively small images, e.g. 640x640 pixels for YOLOv5m. To inference on a large image it is necessary to use a sliding window over the image, inference on each window, then combine the results. However lower confidence predicitons will be made at the edges of the window where objects may be partially cropped. To overcome this a framework called [sahi](https://github.com/obss/sahi) has been developed. An example of how to use sahi with yolo [is here](https://github.com/open-mmlab/mmyolo/blob/dev/demo/large_image_demo.py). For an example of using threading to process a large image see [Fast-Large-Image-Object-Detection-yolov7](https://github.com/shah0nawaz/Fast-Large-Image-Object-Detection-yolov7)

## Models in the browser
The model is run in the browser itself on live images, ensuring processing is always with the latest model available and removing the requirement for dedicated server side inferencing
* [Classifying satellite imagery - Made with TensorFlow.js YoutTube video](https://www.youtube.com/watch?v=9zqjgeqc-ew)

## Model optimisation for deployment
The general approaches are outlined in [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed. There are also toolkits for optimisation, in particular [ONNX](https://github.com/microsoft/onnxruntime) which is framework agnostic.

## MLOps
[MLOps](https://en.wikipedia.org/wiki/MLOps) is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently.
* [How to Build MLOps Pipelines with GitHub Actions](https://neptune.ai/blog/build-mlops-pipelines-with-github-actions-guide/)

## Model monitoring
Once your model is deployed you will want to monitor for data errors, broken pipelines, and model performance degradation/drift [ref](https://towardsdatascience.com/deploy-and-monitor-your-ml-application-with-flask-and-whylabs-4cd1e757c94b)
* [Blog post by Neptune: Doing ML Model Performance Monitoring The Right Way](https://neptune.ai/blog/ml-model-performance-monitoring)
* [whylogs](https://github.com/whylabs/whylogs) -> Profile and monitor your ML data pipeline end-to-end

# Cloud providers
An overview of the most relevant services provided by AWS, Google and Microsoft. Also consider one of the many smaller but more specialised platorms such as [paperspace](https://www.paperspace.com/)

## AWS
* Host your data on [S3](https://aws.amazon.com/s3/) and metadata in a db such as [postgres](https://aws.amazon.com/rds/postgresql/)
* For batch processing use [Batch](https://aws.amazon.com/batch/). GPU instances are available for [batch deep learning](https://aws.amazon.com/blogs/compute/deep-learning-on-aws-batch/) inferencing. See how Rastervision implement this [here](https://docs.rastervision.io/en/0.13/cloudformation.html)
* If processing can be performed in 15 minutes or less, serverless [Lambda](https://aws.amazon.com/lambda/) functions are an attractive option owing to their ability to scale. Note that lambda may not be a particularly quick solution for deep learning applications, since you do not have the option to batch inference on a GPU. Creating a docker container with all the required dependencies can be a challenge. To get started read [Using container images to run PyTorch models in AWS Lambda](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/) and for an image classification example [checkout this repo](https://github.com/aws-samples/aws-lambda-docker-serverless-inference). Also read [Processing satellite imagery with serverless architecture](https://aws.amazon.com/blogs/compute/processing-satellite-imagery-with-serverless-architecture/) which discusses queuing & lambda. Sagemaker also supports server less inference, see  [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html). For managing a serverless infrastructure composed of multiple lambda functions use [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html) and read [How to continuously deploy a FastAPI to AWS Lambda with AWS SAM](https://iwpnd.pw/articles/2020-01/deploy-fastapi-to-aws-lambda)
* [Sagemaker](https://aws.amazon.com/sagemaker/) is an ecosystem of ML tools accessed via a hosted Jupyter environment & API. Read [Build GAN with PyTorch and Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/build-gan-with-pytorch-and-amazon-sagemaker/), [Run computer vision inference on large videos with Amazon SageMaker asynchronous endpoints](https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/), [Use Amazon SageMaker to Build, Train, and Deploy ML Models Using Geospatial Data](https://aws.amazon.com/blogs/aws/preview-use-amazon-sagemaker-to-build-train-and-deploy-ml-models-using-geospatial-data/)
* [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) competes with Google colab being free to use with no credit card or AWS account required
* [Deep learning AMIs](https://aws.amazon.com/machine-learning/amis/) are EC2 instances with deep learning frameworks preinstalled. They do require more setup from the user than Sagemaker but in return allow access to the underlying hardware, which makes debugging issues more straightforward. There is a [good guide to setting up your AMI instance on the Keras blog](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html). Read [Deploying the SpaceNet 6 Baseline on AWS](https://medium.com/the-downlinq/deploying-the-spacenet-6-baseline-on-aws-c811ad82da1)
* Specifically created for deep learning inferencing is [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)
* [Rekognition](https://aws.amazon.com/rekognition/custom-labels-features/) custom labels is a 'no code' annotation, training and inferencing service. Read [Training models using Satellite (Sentinel-2) imagery on Amazon Rekognition Custom Labels](https://ryfeus.medium.com/training-models-using-satellite-imagery-on-amazon-rekognition-custom-labels-dd44ac6a3812). For a comparison with Azure and Google alternatives [read this article](https://blog.roboflow.com/automl-vs-rekognition-vs-custom-vision/)
* Use [Glue](https://aws.amazon.com/glue) for data preprocessing - or use Sagemaker
* To orchestrate basic data pipelines use [Step functions](https://aws.amazon.com/step-functions/). Use the [AWS Step Functions Workflow Studio](https://aws.amazon.com/blogs/aws/new-aws-step-functions-workflow-studio-a-low-code-visual-tool-for-building-state-machines/) to get started. Read [Orchestrating and Monitoring Complex, Long-running Workflows Using AWS Step Functions](https://aws.amazon.com/blogs/architecture/field-notes-orchestrating-and-monitoring-complex-long-running-workflows-using-aws-step-functions/) and checkout the [aws-step-functions-data-science-sdk-python](https://github.com/aws/aws-step-functions-data-science-sdk-python)
* If step functions are too limited or you want to write pipelines in python and use Directed Acyclic Graphs (DAGs) for workflow management, checkout hosted [AWS managed Airflow](https://aws.amazon.com/managed-workflows-for-apache-airflow/). Read [Orchestrate XGBoost ML Pipelines with Amazon Managed Workflows for Apache Airflow](https://aws.amazon.com/blogs/machine-learning/orchestrate-xgboost-ml-pipelines-with-amazon-managed-workflows-for-apache-airflow/) and checkout [amazon-mwaa-examples](https://github.com/aws-samples/amazon-mwaa-examples)
* When developing you will definitely want to use [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and probably [aws-data-wrangler](https://github.com/awslabs/aws-data-wrangler)
* For managing infrastructure use [Terraform](https://www.terraform.io/). Alternatively if you wish to use TypeScript, JavaScript, Python, Java, or C# checkout [AWS CDK](https://aws.amazon.com/cdk/), although I found relatively few examples to get going using python
* [AWS Ground Station now supports data delivery to Amazon S3](https://aws.amazon.com/about-aws/whats-new/2021/04/aws-ground-station-now-supports-data-delivery-to-amazon-s3/)
* [Redshift](https://aws.amazon.com/redshift/) is a fast, scalable data warehouse that can extend queries to S3. Redshift is based on PostgreSQL but [has some differences](https://docs.aws.amazon.com/redshift/latest/dg/c_redshift-and-postgres-sql.html). Redshift supports geospatial data.
* [AWS App Runner](https://aws.amazon.com/blogs/containers/introducing-aws-app-runner/) enables quick deployment of containers as apps
* [AWS Athena](https://aws.amazon.com/athena/) allows running SQL queries against CSV files stored on S3. Serverless so pay only for the queries you run
* If you are using pytorch checkout [the S3 plugin for pytorch](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/) which provides streaming data access
* [Amazon AppStream 2.0](https://aws.amazon.com/appstream2/) is a service to securely share desktop apps over the internet
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch
* [Building a robust data pipeline for processing Satellite Imagery at scale](https://medium.com/fasal-engineering/building-a-robust-data-pipeline-for-processing-satellite-imagery-at-scale-808700b008cd) using AWS services & Airflow
* [Using artificial intelligence to detect product defects with AWS Step Functions](https://aws.amazon.com/blogs/compute/using-artificial-intelligence-to-detect-product-defects-with-aws-step-functions/) -> demonstrates image classification workflow
* [sagemaker-defect-detection](https://github.com/awslabs/sagemaker-defect-detection) -> demonstrates object detection training and deployment
* [How do you process space data and imagery in low earth orbit?](https://www.aboutamazon.com/news/aws/how-do-you-process-space-data-and-imagery-in-low-earth-orbit) -> Snowcone is a standalone computer that can run AWS services at the edge, and has been demonstraed on the ISS (International space station)
* [Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) -> can be used to create a visual search service
* [Automated Earth observation using AWS Ground Station Amazon S3 data delivery](https://aws.amazon.com/blogs/publicsector/automated-earth-observation-aws-ground-station-amazon-s3-data-delivery/)
* [Satellogic makes Earth observation data more accessible and affordable with AWS](https://aws.amazon.com/blogs/publicsector/satellogic-makes-earth-observation-data-more-accessible-affordable-aws/)
* [Analyze terabyte-scale geospatial datasets with Dask and Jupyter on AWS](https://aws.amazon.com/blogs/publicsector/analyze-terabyte-scale-geospatial-datasets-with-dask-and-jupyter-on-aws/)
* [How SkyWatch built its satellite imagery solution using AWS Lambda and Amazon EFS](https://aws.amazon.com/blogs/storage/how-skywatch-built-its-imagery-solution-using-aws-lambda-and-amazon-efs/)
* [Identify mangrove forests using satellite image features using Amazon SageMaker Studio and Amazon SageMaker Autopilot](https://aws.amazon.com/blogs/machine-learning/part-2-identify-mangrove-forests-using-satellite-image-features-using-amazon-sagemaker-studio-and-amazon-sagemaker-autopilot/)
* [Detecting invasive Australian tree ferns in Hawaiian forests](https://aws.amazon.com/blogs/machine-learning/automated-scalable-and-cost-effective-ml-on-aws-detecting-invasive-australian-tree-ferns-in-hawaiian-forests/)
* [Improve ML developer productivity with Weights & Biases: A computer vision example on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/improve-ml-developer-productivity-with-weights-biases-a-computer-vision-example-on-amazon-sagemaker/)
* [terraform-aws-tile-service](https://github.com/addresscloud/terraform-aws-tile-service) -> Terraform module to create a vector tile service using Amazon API Gateway and S3
* [sagemaker-ssh-helper](https://github.com/aws-samples/sagemaker-ssh-helper) -> A helper library to connect into Amazon SageMaker with AWS Systems Manager and SSH

## Google Cloud
* For storage use [Cloud Storage](https://cloud.google.com/storage) (AWS S3 equivalent)
* For data warehousing use [BigQuery](https://cloud.google.com/bigquery) (AWS Redshift equivalent). Visualize massive spatial datasets directly in BigQuery using [CARTO](https://carto.com/bigquery-tiler/)
* For model training use [Vertex](https://cloud.google.com/vertex-ai) (AWS Sagemaker equivalent)
* For containerised apps use [Cloud Run](https://cloud.google.com/run) (AWS App Runner equivalent but can scale to zero)

## Microsoft Azure
* [Azure Orbital](https://azure.microsoft.com/en-us/services/orbital/) -> Satellite ground station and scheduling services for fast downlinking of data
* [ShipDetection](https://github.com/microsoft/ShipDetection) -> use the Azure Custom Vision service to train an object detection model that can detect and locate ships in a satellite image
* [SwimmingPoolDetection](https://github.com/retkowsky/SwimmingPoolDetection) -> Swimming pool detection with Azure Custom Vision
* [Geospatial analysis with Azure Synapse Analytics](https://docs.microsoft.com/en-us/azure/architecture/industries/aerospace/geospatial-processing-analytics) and [repo](https://github.com/Azure/Azure-Orbital-Analytics-Samples)
* [AIforEarthDataSets](https://github.com/microsoft/AIforEarthDataSets) -> Notebooks and documentation for AI-for-Earth managed datasets on Azure

# State of the art engineering
* Compute and data storage are on the cloud. Read how [Planet](https://cloud.google.com/customers/planet) and [Airbus](https://cloud.google.com/customers/airbus) use the cloud
* Traditional data formats aren't designed for processing on the cloud, so new standards are evolving such as [COG](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-optimised-geotiff-cog) and [STAC](https://github.com/robmarkcole/satellite-image-deep-learning#spatiotemporal-asset-catalog-specification-stac)
* Google Earth Engine and Microsoft Planetary Computer are democratising access to 'planetary scale' compute
* Google Colab and others are providing free acces to GPU compute to enable training deep learning models
* No-code platforms and auto-ml are making ML techniques more accessible than ever
* Serverless compute (e.g. AWS Lambda) mean that managing servers may become a thing of the past
* Custom hardware is being developed for rapid training and inferencing with deep learning models, both in the datacenter and at the edge
* Supervised ML methods typically require large annotated datasets, but approaches such as self-supervised and active learning require less or even no annotation
* Computer vision traditionally delivered high performance image processing on a CPU by using compiled languages like C++, as used by OpenCV for example. The advent of GPUs are changing the paradigm, with alternatives optimised for GPU being created, such as [Kornia](https://github.com/kornia/kornia)
* Whilst the combo of python and keras/tensorflow/pytorch are currently preeminent, new python libraries such as [Jax](https://github.com/google/jax) and alternative languages such as [Julia](https://julialang.org/) are showing serious promise
