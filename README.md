<div align="center">
  <p>
    <a href="https://www.satellite-image-deep-learning.com/">
        <img src="logo.png" width="700">
    </a>
</p>
  <h2>Datasets for deep learning applied to satellite and aerial imagery.</h2>

# 👉 [satellite-image-deep-learning.com](https://www.satellite-image-deep-learning.com/) 👈

</div>

**How to use this repository:** if you know exactly what you are looking for (e.g. you have the paper name) you can `Control+F` to search for it in this page (or search in the raw markdown).

# Lists of datasets
<!-- markdown-link-check-disable -->
* [Earth Observation Database](https://eod-grss-ieee.com/)
<!-- markdown-link-check-enable -->
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)
* [Awesome_Satellite_Benchmark_Datasets](https://github.com/Seyed-Ali-Ahmadi/Awesome_Satellite_Benchmark_Datasets)
* [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) -> dedicated to change detection
* [Callisto-Dataset-Collection](https://github.com/Agri-Hub/Callisto-Dataset-Collection) -> datasets that use Copernicus/sentinel data
* [geospatial-data-catalogs](https://github.com/giswqs/geospatial-data-catalogs) -> A list of open geospatial datasets available on AWS, Earth Engine, Planetary Computer, and STAC Index
* [BED4RS](https://captain-whu.github.io/BED4RS/)
* [Satellite-Image-Time-Series-Datasets](https://github.com/corentin-dfg/Satellite-Image-Time-Series-Datasets)

# Remote sensing dataset hubs
* [Radiant MLHub](https://mlhub.earth/) -> both datasets and models
* [Registry of Open Data on AWS](https://registry.opendata.aws)
* [Microsoft Planetary Computer data catalog](https://planetarycomputer.microsoft.com/catalog)
* [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)

## Sentinel
As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions)
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel) -> a curated list of awesome tools, tutorials and APIs related to data from the Copernicus Sentinel Satellites.
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) and [Sentinel-2 L2A 120m Mosaic](https://registry.opendata.aws/sentinel-s2-l2a-mosaic-120/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F)
* Paid access to Sentinel & Landsat data via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py)
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* [Analyzing Sentinel-2 satellite data in Python with Keras](https://github.com/jensleitloff/CNN-Sentinel)
* [Xarray backend to Copernicus Sentinel-1 satellite data products](https://github.com/bopen/xarray-sentinel)
* [SEN2VENµS](https://zenodo.org/record/6514159#.YoRxM5PMK3I) -> a dataset for the training of Sentinel-2 super-resolution algorithms
* [M3LEO](https://huggingface.co/M3LEO) -> [Github](https://github.com/spaceml-org/M3LEO). A very large scale georeferenced dataset of Sentinel 1/2 imagery plus interferometric SAR products and auxiliary datasets such as Land cover, Biomass and Digital Elevation Models.
* [SEN12MS](https://github.com/zhu-xlab/SEN12MS) -> A Curated Dataset of Georeferenced Multi-spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion. Checkout [SEN12MS toolbox](https://github.com/schmitt-muc/SEN12MS) and many referenced uses on [paperswithcode.com](https://paperswithcode.com/dataset/sen12ms)
* [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) -> A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning, with and [models](https://github.com/Orion-AI-Lab/S4A-Models)
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
* [Amazon Rainforest dataset for semantic segmentation](https://zenodo.org/record/3233081#.Y6LPLOzP1hE) -> Sentinel 2 images
* [Mining and clandestine airstrips datasets](https://github.com/earthrise-media/mining-detector)
* [Satellite Burned Area Dataset](https://zenodo.org/record/6597139#.Y9ufiezP1hE) -> segmentation dataset containing several satellite acquisitions related to past forest wildfires. It contains 73 acquisitions from Sentinel-2 and Sentinel-1 (Copernicus).
* [mmflood](https://github.com/edornd/mmflood) -> Flood delineation from Sentinel-1 SAR imagery, with [paper](https://ieeexplore.ieee.org/abstract/document/9882096)
* [MATTER](https://github.com/periakiva/MATTER) -> a Sentinel 2 dataset for Self-Supervised Training
* [Industrial Smoke Plumes](https://zenodo.org/record/4250706)
* [MARIDA: Marine Debris Archive](https://github.com/marine-debris/marine-debris.github.io)
* [S2GLC](https://s2glc.cbk.waw.pl/) -> High resolution Land Cover Map of Europe
* [Generating Imperviousness Maps from Multispectral Sentinel-2 Satellite Imagery](https://zenodo.org/record/7058860#.ZDrAeuzMLdo)
* [Sentinel-2 Water Edges Dataset (SWED)](https://openmldata.ukho.gov.uk/)
* [Sentinel-1 for Science Amazonas](https://sen4ama.gisat.cz/) -> forest lost time series dataset
* [Sentinel2 Munich480](https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480) -> dataset for crop mapping by exploiting the time series of Sentinel-2 satellite
* [Meadows vs Orchards](https://www.kaggle.com/datasets/baptistel/meadows-vs-orchards) -> a pixel time series dataset
* [SEN12_GUM](https://zenodo.org/record/6914898) -> SEN12 Global Urban Mapping Dataset
* [Sentinel-1&2 Image Pairs (SAR & Optical)](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
* [Sentinel-2 Image Time Series for Crop Mapping](https://www.kaggle.com/datasets/ignazio/sentinel2-crop-mapping) -> data for the Lombardy region in Italy
* [Deforestation in Ukraine from Sentinel2 data](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)
* [Multitask Learning for Estimating Power Plant Greenhouse Gas Emissions from Satellite Imagery](https://zenodo.org/record/5644746)
* [METER-ML: A Multi-sensor Earth Observation Benchmark for Automated Methane Source Mapping](https://stanfordmlgroup.github.io/projects/meter-ml/) -> data [on Zenodo](https://zenodo.org/record/6911013)
* [satellite-change-events](https://www.cs.cornell.edu/projects/satellite-change-events/) -> CaiRoad & CalFire change detection Sentinel 2 datasets
* [OMS2CD](https://github.com/Dibz15/OpenMineChangeDetection) -> hand-labelled images for change-detection in open-pit mining areas
* [coal power plants’ emissions](https://transitionzero.medium.com/estimating-coal-power-plant-operation-from-satellite-images-with-computer-vision-b966af56919e) -> a dataset of coal power plants’ emissions, including images, metadata and labels.
* [RapidAI4EO](https://rapidai4eo.radiant.earth/) -> dense time series satellite imagery sampled at 500,000 locations across Europe, comprising S2 & Planet imagery, with CORINE Land Cover multiclass labels for 2018
* [Sentinel 2 super-resolved data cubes - 92 scenes over 2 regions in Switzerland spanning 5 years](https://ieee-dataport.org/documents/sentinel-2-super-resolved-data-cubes-92-scenes-over-2-regions-switzerland-spanning-5-years)
* [MS-HS-BCD-dataset](https://github.com/arcgislearner/MS-HS-BCD-dataset) -> multisource change detection dataset used in paper: Building Change Detection with Deep Learning by Fusing Spectral and Texture Features of Multisource Remote Sensing Images: A GF-1 and Sentinel 2B Data Case
* [MSOSCD](https://github.com/Lihy256/MSCDUnet) -> change detection datasets containing VHR, multispectral (Sentinel-2) and SAR (Sentinel-1)
* [Sentinel-2 dataset for ship detection](https://zenodo.org/records/3923841), also edited and redistributed as [VDS2RAW](https://zenodo.org/records/7982468#.ZIiLxS8QOo4)
* [MineSegSAT](https://github.com/macdonaldezra/MineSegSAT) -> dataset for paper: AN AUTOMATED SYSTEM TO EVALUATE MINING DISTURBED AREA EXTENTS FROM SENTINEL-2 IMAGERY
* [CropNet: An Open Large-Scale Dataset with Multiple Modalities for Climate Change-aware Crop Yield Predictions](https://anonymous.4open.science/r/CropNet/README.md) -> terabyte-sized, publicly available, and multi-modal dataset for climate change-aware crop yield predictions
* [Tiny CropNet dataset](https://github.com/fudong03/MMST-ViT)
* [CaBuAr](https://github.com/DarthReca/CaBuAr) -> California Burned Areas dataset for delineation
* [sen12mscr](https://patricktum.github.io/cloud_removal/sen12mscr/) -> Multimodal Cloud Removal
* [Greenearthnet](https://github.com/vitusbenson/greenearthnet/tree/main) -> dataset specifically designed for high-resolution vegetation forecasting
* [MultiSenGE](https://zenodo.org/records/6375466) -> large-scale multimodal and multitemporal benchmark dataset
* [Floating-Marine-Debris-Data](https://github.com/miguelmendesduarte/Floating-Marine-Debris-Data) -> floating marine debris, with annotations for six debris classes, including plastic, driftwood, seaweed, pumice, sea snot, and sea foam.
* [Sen2Fire](https://zenodo.org/records/10881058) -> A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data
* [L1BSR](https://zenodo.org/records/7826696) -> 3740 pairs of overlapping image crops extracted from two L1B products
* [GloSoFarID](https://github.com/yzyly1992/GloSoFarID) -> Global multispectral dataset for Solar Farm IDentification
* [SICKLE](https://github.com/Depanshu-Sani/SICKLE) -> A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters. Multi-resolution time-series images from Landsat-8, Sentinel-1, and Sentinel-2
* [MARIDA](https://marine-debris.github.io/index.html) -> Marine Debris detection from Sentinel-2
* [MADOS](https://github.com/gkakogeorgiou/mados) -> Marine Debris and Oil Spill from Sentinel-2
* [Sentinel-1 and Sentinel-2 Vessel Detection](https://github.com/allenai/vessel-detection-sentinels)
* [TreeSatAI](https://zenodo.org/records/6780578) -> Sentinel-1, Sentinel-2
* [Sentinel-2 dataset for ship detection and characterization](https://zenodo.org/records/10418786) -> RGB
* [S2-SHIPS](https://github.com/alina2204/contrastive_SSL_ship_detection) -> all 12 channels
* [ChatEarthNet](https://github.com/zhu-xlab/ChatEarthNet) -> A Global-Scale Image-Text Dataset Empowering Vision-Language Geo-Foundation Models, utilizes Sentinel-2 data with captions generated by ChatGPT
* [UKFields](https://github.com/Spiruel/UKFields) -> over 2.3 million automatically delineated field boundaries spanning England, Wales, Scotland, and Northern Ireland
* [ShipWakes](https://zenodo.org/records/7947694) -> Keypoints Method for Recognition of Ship Wake Components in Sentinel-2 Images by Deep Learning
* [TimeSen2Crop](https://zenodo.org/records/4715631) -> a Million Labeled Samples Dataset of Sentinel 2 Image Time Series for Crop Type Classification
* [AgriSen-COG](https://github.com/tselea/agrisen-cog) -> a Multicountry, Multitemporal Large-Scale Sentinel-2 Benchmark Dataset for Crop Mapping: includes an anomaly detection preprocessing step
* [MagicBathyNet](https://www.magicbathy.eu/magicbathynet.html) -> a new multimodal benchmark dataset made up of image patches of Sentinel-2, SPOT-6 and aerial imagery, bathymetry in raster format and seabed classes annotations
* [AI2-S2-NAIP](https://huggingface.co/datasets/allenai/s2-naip) -> aligned NAIP, Sentinel-2, Sentinel-1, and Landsat images spanning the entire continental US
* [MuS2: A Benchmark for Sentinel-2 Multi-Image Super-Resolution](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F1JMRAT)
* [Sen4Map](https://datapub.fz-juelich.de/sen4map/) -> Sentinel-2 time series images, covering over 335,125 geo-tagged locations across the European Union. These geo-tagged locations are associated with detailed landcover and land-use information
* [CloudSEN12Plus](https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus) -> the largest cloud detection dataset to date for Sentinel-2
* [mayrajeo S2 ship detection](https://github.com/mayrajeo/ship-detection) -> labels for Detecting marine vessels from Sentinel-2 imagery with YOLOv8
* [Fields of The World](https://fieldsofthe.world/) -> instance segmentation of agricultural field boundaries
* [ai4boundaries](https://github.com/waldnerf/ai4boundaries) -> field boundaries with Sentinel-2 and aerial photography
* [California Wildfire GeoImaging Dataset - CWGID](https://arxiv.org/abs/2409.16380) -> Development and Application of a Sentinel-2 Satellite Imagery Dataset for Deep-Learning Driven Forest Wildfire Detection
* [POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2](https://popcorn-population.github.io/)
* [substation-seg](https://github.com/Lindsay-Lab/substation-seg) -> segmenting substations dataset
* [PhilEO-downstream](https://huggingface.co/datasets/PhilEO-community/PhilEO-downstream) -> a 400GB Sentinel-2 dataset for building density estimation, road segmentation, and land cover classification.
* [PhilEO-pretrain](https://huggingface.co/datasets/PhilEO-community/PhilEO-pretrain) -> a 500GB global dataset of Sentinel-2 images for model pre-training.
* [KappaSet: Sentinel-2 KappaZeta Cloud and Cloud Shadow Masks](https://zenodo.org/records/7100327)
* [AllClear](https://allclear.cs.cornell.edu/) A Comprehensive Dataset and Benchmark for Cloud Removal in Satellite Imagery
* [Sentinel-2 reference cloud masks generated by an active learning method](https://zenodo.org/records/1460961)
* [Cloud gap-filling with deep learning for improved grassland monitoring](https://zenodo.org/records/11651601)

## Landsat
Long running US program -> see [Wikipedia](https://en.wikipedia.org/wiki/Landsat_program)
* 8 bands, 15 to 60 meters, 185km swath, the temporal resolution is 16 days
* [Landsat 4, 5, 7, and 8 imagery on Google](https://cloud.google.com/storage/docs/public-datasets/landsat), see [the GCP bucket here](https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/), with Landsat 8 imagery in COG format analysed in [this notebook](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb)
* [Landsat 8 imagery on AWS](https://registry.opendata.aws/landsat-8/), with many tutorials and tools listed
* https://github.com/kylebarron/landsat-mosaic-latest -> Auto-updating cloudless Landsat 8 mosaic from AWS SNS notifications
* [Visualise landsat imagery using Datashader](https://examples.pyviz.org/landsat/landsat.html#landsat-gallery-landsat)
* [Landsat-mosaic-tiler](https://github.com/kylebarron/landsat-mosaic-tiler) -> This repo hosts all the code for landsatlive.live website and APIs.
* [LandsatSCD](https://github.com/ggsDing/SCanNet/tree/main) -> a change detection dataset, it consists of 8468 pairs of images, each having the spatial resolution of 416 × 416
* [The Landsat Irish Coastal Segmentation Dataset](https://zenodo.org/records/8414665)

## VENμS
Vegetation and Environment monitoring on a New Micro-Satellite ([VENμS](https://en.wikipedia.org/wiki/VEN%CE%BCS))
* [VENUS L2A Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/venus-l2a-cogs/)
* [VENuS cloud mask training dataset](https://zenodo.org/records/7040177)
* [Sen2Venµs](https://zenodo.org/records/6514159) -> a dataset for the training of Sentinel-2 super-resolution algorithms
* [sen2venus-pytorch-dataset](https://github.com/piclem/sen2venus-pytorch-dataset) -> torch dataloader and other utilities

## Maxar
Satellites owned by Maxar (formerly DigitalGlobe) include [GeoEye-1](https://en.wikipedia.org/wiki/GeoEye-1), [WorldView-2](https://en.wikipedia.org/wiki/WorldView-2), [3](https://en.wikipedia.org/wiki/WorldView-3) & [4](https://en.wikipedia.org/wiki/WorldView-4)
* [Maxar Open Data Program](https://github.com/opengeos/maxar-open-data) provides pre and post-event high-resolution satellite imagery in support of emergency planning, response, damage assessment, and recovery
* [WorldView-2 European Cities](https://earth.esa.int/eogateway/catalog/worldview-2-european-cities) -> dataset covering the most populated areas in Europe at 40 cm resolution

## Planet
* [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/), supported through Norway’s International Climate & Forests Initiative. [BBC coverage](https://www.bbc.co.uk/news/science-environment-54651453)
* Planet have made imagery available via kaggle competitions
* [Alberta Wells Dataset](https://zenodo.org/records/13743323) -> Pinpointing Oil and Gas Wells from Satellite Imagery

## UC Merced
Land use classification dataset with 21 classes and 100 RGB TIFF images for each class. Each image measures 256x256 pixels with a pixel resolution of 1 foot
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* Also [available as a multi-label dataset](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d)
* Read [Vision Transformers for Remote Sensing Image Classification](https://www.mdpi.com/2072-4292/13/3/516/htm) where a Vision Transformer classifier achieves 98.49% classification accuracy on Merced

## EuroSAT
Land use classification dataset of Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples. Available in RGB and 13 band versions
* [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT) -> publication where a CNN achieves a classification accuracy 98.57%
* Repos using fastai [here](https://github.com/shakasom/Deep-Learning-for-Satellite-Imagery) and [here](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html)
* [evolved_channel_selection](http://matpalm.com/blog/evolved_channel_selection/) -> explores the trade off between mixed resolutions and whether to use a channel at all, with [repo](https://github.com/matpalm/evolved_channel_selection)
* RGB version available as [dataset in pytorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT) with the 13 band version [in torchgeo](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat). Checkout the tutorial on [data augmentation with this dataset](https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html)
* [EuroSAT-SAR](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR) -> matched each Sentinel-2 image in EuroSAT with one Sentinel-1 patch according to the geospatial coordinates

## PatternNet
Land use classification dataset with 38 classes and 800 RGB JPG images for each class
* https://sites.google.com/view/zhouwx/dataset?authuser=0
* Publication: [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424)

## Gaofen Image Dataset (GID) for classification
- https://captain-whu.github.io/GID/
- a large-scale classification set and a fine land-cover classification set

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
* [Object_Detection_Satellite_Imagery_Yolov8_DIOR](https://github.com/JohnPPinto/Object_Detection_Satellite_Imagery_Yolov8_DIOR)

## Multiscene
MultiScene dataset aims at two tasks: Developing algorithms for multi-scene recognition & Network learning with noisy labels
* https://multiscene.github.io/ & https://github.com/Hua-YS/Multi-Scene-Recognition

## FAIR1M object detection dataset
A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery
* [arxiv papr](https://arxiv.org/abs/2103.05569)
* Download at gaofen-challenge.com
* [2020Gaofen](https://github.com/AICyberTeam/2020Gaofen) -> 2020 Gaofen Challenge data, baselines, and metrics

## DOTA object detection dataset
A Large-Scale Benchmark and Challenges for Object Detection in Aerial Images. Segmentation annotations available in iSAID dataset
* https://captain-whu.github.io/DOTA/index.html
* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for loading dataset
* [Arxiv paper](https://arxiv.org/abs/1711.10398)
* [Pretrained models in mmrotate](https://github.com/open-mmlab/mmrotate)
* [DOTA2VOCtools](https://github.com/Complicateddd/DOTA2VOCtools) -> dataset split and transform to voc format
* [dotatron](https://github.com/naivelogic/dotatron) -> 2021 Learning to Understand Aerial Images Challenge on DOTA dataset

## iSAID instance segmentation dataset
A Large-scale Dataset for Instance Segmentation in Aerial Images
* https://captain-whu.github.io/iSAID/dataset.html
* Uses images from the DOTA dataset

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
* [Satellite_Imagery_Detection_YOLOV7](https://github.com/Radhika-Keni/Satellite_Imagery_Detection_YOLOV7) -> YOLOV7 applied to xView1

## xView2: xBD building damage assessment
Annotated high-resolution satellite imagery for building damage assessment, precise segmentation masks and damage labels on a four-level spectrum, 0.3m resolution imagery
* [Official website](https://xview2.org/)
* [arXiv paper](https://arxiv.org/abs/1911.09296)
* [paperswithcode](https://paperswithcode.com/paper/xbd-a-dataset-for-assessing-building-damage)
* [xView2_baseline](https://github.com/DIUx-xView/xView2_baseline) -> baseline solution in tensorflow
* [metadamagenet](https://github.com/nimaafshar/metadamagenet) -> pytorch solution
* [U-Net models from michal2409](https://github.com/michal2409/xView2)
* [DAHiTra](https://github.com/nka77/DAHiTra) -> code for 2022 [paper](https://arxiv.org/abs/2208.02205): Large-scale Building Damage Assessment using a Novel Hierarchical Transformer Architecture on Satellite Images. Uses xView2 xBD dataset
* [Damage assessment using Amazon SageMaker geospatial capabilities and custom SageMaker models](https://aws.amazon.com/blogs/machine-learning/damage-assessment-using-amazon-sagemaker-geospatial-capabilities-and-custom-sagemaker-models/)
* [Xview2_Strong_Baseline](https://github.com/PaulBorneP/Xview2_Strong_Baseline) -> a simple implementation of a strong baseline

## xView3: Detecting dark vessels in SAR
Detecting dark vessels engaged in illegal, unreported, and unregulated (IUU) fishing activities on synthetic aperture radar (SAR) imagery. With human and algorithm annotated instances of vessels and fixed infrastructure across 43,200,000 km^2 of Sentinel-1 imagery, this multi-modal dataset enables algorithms to detect and classify dark vessels
* [Official website](https://iuu.xview.us/)
* [arXiv paper](https://arxiv.org/abs/2206.00897)
* [Github](https://github.com/DIUx-xView) -> all reference code, dataset processing utilities, and winning model codes + weights
* [paperswithcode](https://paperswithcode.com/dataset/xview3-sar)
* [xview3_ship_detection](https://github.com/naivelogic/xview3_ship_detection)

## Vehicle Detection in Aerial Imagery (VEDAI)
Vehicle Detection in Aerial Imagery. Bounding box annotations
* https://downloads.greyc.fr/vedai/
* [pytorch-vedai](https://github.com/MichelHalmes/pytorch-vedai)

## Cars Overhead With Context (COWC)
Large set of annotated cars from overhead. Established baseline for object detection and counting tasks. Annotations in bounding box format
* http://gdo152.ucllnl.org/cowc/
* https://github.com/LLNL/cowc
* [Detecting cars from aerial imagery for the NATO Innovation Challenge](https://arthurdouillard.com/post/nato-challenge/)

## AI-TOD & AI-TOD-v2 - tiny object detection
The mean size of objects in AI-TOD is about 12.8 pixels, which is much smaller than other datasets. Annotations in bounding box format. V2 is a meticulous relabelling of the v1 dataset
* https://github.com/jwwangchn/AI-TOD
* https://chasel-tsui.github.io/AI-TOD-v2/
* [NWD](https://github.com/jwwangchn/NWD) -> code for 2021 [paper](https://arxiv.org/abs/2110.13389): A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. Uses AI-TOD dataset
* [ORFENet](https://github.com/dyl96/ORFENet) -> Tiny Object Detection in Remote Sensing Images Based on Object Reconstruction and Multiple Receptive Field Adaptive Feature Enhancement. Uses LEVIR-ship & AI-TOD-v2

## RarePlanes
* [RarePlanes](https://registry.opendata.aws/rareplanes/) -> incorporates both real and synthetically generated satellite imagery including aircraft. Read the [arxiv paper](https://arxiv.org/abs/2006.02963) and checkout [this repo](https://github.com/jdc08161063/RarePlanes). Note the dataset is available through the AWS Open-Data Program for free download
* [Understanding the RarePlanes Dataset and Building an Aircraft Detection Model](https://encord.com/blog/rareplane-dataset-aircraft-detection-model/) -> blog post
* Read [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed
* [yoltv4](https://github.com/avanetten/yoltv4) includes examples on the [RarePlanes dataset](https://registry.opendata.aws/rareplanes/)
* [rareplanes-yolov5](https://github.com/jeffaudi/rareplanes-yolov5) -> using YOLOv5 and the RarePlanes dataset to detect and classify sub-characteristics of aircraft, with [article](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)

## Counting from Sky
A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method
* https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method

## AIRS (Aerial Imagery for Roof Segmentation)
Public dataset for roof segmentation from very-high-resolution aerial imagery (7.5cm). Covers almost the full area of Christchurch, the largest city in the South Island of New Zealand.
* [On Kaggle](https://www.kaggle.com/datasets/atilol/aerialimageryforroofsegmentation)
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

## LEVIR-CD building change detection dataset
* https://justchenhao.github.io/LEVIR/
* [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) -> pytorch implemention of FCCDN for change detection task
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset uses LEVIR-CD imagery

## Onera (OSCD) Sentinel-2 change detection dataset
It comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018. 
* [Onera Satellite Change Detection Dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018
* [Website](https://rcdaudt.github.io/oscd/)
* [change_detection_onera_baselines](https://github.com/previtus/change_detection_onera_baselines) -> Siamese version of U-Net baseline model
* [Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks](https://github.com/rcdaudt/patch_based_change_detection) -> with [paper](https://ieeexplore.ieee.org/abstract/document/8518015)
* [DS_UNet](https://github.com/SebastianHafner/DS_UNet) -> code for 2021 paper: Sentinel-1 and Sentinel-2 Data Fusion for Urban Change Detection using a Dual Stream U-Net, uses Onera Satellite Change Detection dataset
* [ChangeDetection_wOnera](https://github.com/tonydp03/ChangeDetection_wOnera)
* [OSCD + additional Dates](https://github.com/granularai/fabric) -> extended with three different dates
* [MSOSCD](https://github.com/Lihy256/MSCDUnet) -> change detection datasets containing VHR, multispectral (Sentinel-2) and SAR (Sentinel-1)

## SECOND - semantic change detection
* https://captain-whu.github.io/SCD/
* Change detection at the pixel level

## Amazon and Atlantic Forest dataset
For semantic segmentation with Sentinel 2
* [Amazon and Atlantic Forest image datasets for semantic segmentation](https://zenodo.org/record/4498086#.Y6LPLuzP1hE)
* [attention-mechanism-unet](https://github.com/davej23/attention-mechanism-unet) -> An attention-based U-Net for detecting deforestation within satellite sensor imagery
* [TransUNetplus2](https://github.com/aj1365/TransUNetplus2) -> Rethinking attention gated TransU-Net for deforestation mapping

## Functional Map of the World ( fMoW)
* https://github.com/fMoW/dataset
* RGB & multispectral variants
* High resolution, chip classification dataset
* Purpose: predicting the functional purpose of buildings and land use from temporal sequences of satellite images and a rich set of metadata features

## HRSCD change detection
* https://rcdaudt.github.io/hrscd/
* 291 coregistered image pairs of high resolution RGB aerial images
* Pixel-level change and land cover annotations are provided

## MiniFrance-DFC22 - semi-supervised semantic segmentation
* The [MiniFrance-DFC22 (MF-DFC22) dataset](https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022) extends and modifies the [MiniFrance dataset](https://ieee-dataport.org/open-access/minifrance) for training semi-supervised semantic segmentation models for land use/land cover mapping
* [dfc2022-baseline](https://github.com/isaaccorley/dfc2022-baseline) -> baseline solution to the 2022 IEEE GRSS Data Fusion Contest (DFC2022) using TorchGeo, PyTorch Lightning, and Segmentation Models PyTorch to train a U-Net with a ResNet-18 backbone and a loss function of Focal + Dice loss to perform semantic segmentation on the DFC2022 dataset
* https://github.com/mveo/mveo-challenge

## FLAIR
Semantic segmentation and domain adaptation challenge proposed by the French National Institute of Geographical and Forest Information (IGN). Uses a dataset composed of over 70,000 aerial imagery patches with pixel-based annotations and 50,000 Sentinel-2 satellite acquisitions.
* [Challenge on codalab](https://codalab.lisn.upsaclay.fr/competitions/13447)
* [FLAIR-2 github](https://github.com/IGNF/FLAIR-2)
* [flair-2 8th place solution](https://github.com/association-rosia/flair-2)
* [IGNF HuggingFace](https://huggingface.co/IGNF)

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
* [Quick tour of the WorldStrat Dataset](https://medium.com/@robmarkcole/quick-tour-of-the-worldstrat-dataset-b2d1c2d435db)
* Each high-resolution image (1.5 m/pixel) comes with multiple temporally-matched low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites (10 m/pixel)
* Several super-resolution benchmark models trained on it

## Satlas Pretrain
SatlasPretrain is a large-scale pre-training dataset for tasks that involve understanding satellite images. Regularly-updated satellite data is publicly available for much of the Earth through sources such as Sentinel-2 and NAIP, and can inform numerous applications from tackling illegal deforestation to monitoring marine infrastructure. 
* [Website](https://satlas-pretrain.allen.ai/)
* [Code](https://github.com/allenai/satlas)

## FLAIR 1 & 2 Segmentation datasets
* https://ignf.github.io/FLAIR/
* The FLAIR #1 semantic segmentation dataset consists of 77,412 high resolution patches (512x512 at 0.2 m spatial resolution) with 19 semantic classes
* FLAIR #2 includes an expanded dataset of Sentinel-2 time series for multi-modal semantic segmentation

## Five Billion Pixels segmentation dataset
* https://x-ytong.github.io/project/Five-Billion-Pixels.html
* 4m Gaofen-2 imagery over China
* 24 land cover classes
* Paper and code demonstrating domain adaptation to Sentinel-2 and Planetscope imagery
* Extends the [GID15 large scale semantic segmentation dataset](https://captain-whu.github.io/GID15/)
* [GID](https://x-ytong.github.io/project/GID.html) -> the Gaofen Image Dataset is a large-scale land-cover dataset with Gaofen-2 (GF-2) satellite images

## RF100 object detection benchmark
RF100 is compiled from 100 real world datasets that straddle a range of domains. The aim is that performance evaluation on this dataset will enable a more nuanced guide of how a model will perform in different domains. Contains 10k aerial images
* https://www.rf100.org/
* https://github.com/roboflow-ai/roboflow-100-benchmark

## SODA-A rotated bounding boxes
* https://shaunyuan22.github.io/SODA/
* SODA-A comprises 2513 high-resolution images of aerial scenes, which has 872069 instances annotated with oriented rectangle box annotations over 9 classes
* https://github.com/shaunyuan22/CFINet

## EarthView from Satellogic
* https://huggingface.co/datasets/satellogic/EarthView
* Dataset for foundational models, with Sentinel 1 & 2 and 1m RGB

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

## Image captioning datasets
* [RSICD](https://github.com/201528014227051/RSICD_optimal) -> 10921 images with five sentences descriptions per image. Used in  [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd), models at [this repo](https://github.com/arampacha/CLIP-rsicd)
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset contains 10077 pairs of bi-temporal remote sensing images and 50385 sentences describing the differences between images. Uses LEVIR-CD imagery
* [ChatEarthNet](https://github.com/zhu-xlab/ChatEarthNet) -> A Global-Scale Image-Text Dataset Empowering Vision-Language Geo-Foundation Models, utilizes Sentinel-2 data with captions generated by ChatGPT

## Weather Datasets
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/datasets/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data
* [DeepWeather](https://github.com/adamhazimeh/DeepWeather) -> improve weather forecasting accuracy by analyzing satellite images

## Cloud datasets
* [Planet-CR](https://github.com/zhu-xlab/Planet-CR) -> A Multi-Modal and Multi-Resolution Dataset for Cloud Removal in High Resolution Optical Remote Sensing Imagery, 3m resolution, with [paper](https://arxiv.org/abs/2301.03432)
* [The Azavea Cloud Dataset](https://www.azavea.com/blog/2021/08/02/the-azavea-cloud-dataset/) which is used to train this [cloud-model](https://github.com/azavea/cloud-model)
* [Sentinel-2 Cloud Cover Segmentation Dataset](https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1) on Radiant mlhub
* [cloudsen12](https://cloudsen12.github.io/) -> see [video](https://youtu.be/GhQwnVhJ1wo)
* [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) -> High-Resolution Cloud Detection Dataset comprising 150 RGB images and a resolution varying from 0.5 to 15 m in different global regions
* [AIR-CD](https://github.com/AICyberTeam/AIR-CD) -> a challenging cloud detection data set called AIR-CD, with higher spatial resolution and more representative landcover types
* [Landsat 8 Cloud Cover Assessment Validation Data](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)

## Forest datasets
* [OpenForest](https://github.com/RolnickLab/OpenForest) -> A catalogue of open access forest datasets
* [awesome-forests](https://github.com/blutjens/awesome-forests) -> A curated list of ground-truth forest datasets for the machine learning and forestry community
* [ReforesTree](https://github.com/gyrrei/ReforesTree) -> A dataset for estimating tropical forest biomass based on drone and field data
* [yosemite-tree-dataset](https://github.com/nightonion/yosemite-tree-dataset) -> a benchmark dataset for tree counting from aerial images
* [Amazon Rainforest dataset for semantic segmentation](https://zenodo.org/record/3233081#.Y6LPLOzP1hE) -> Sentinel 2 images. Used in the paper 'An attention-based U-Net for detecting deforestation within satellite sensor imagery'
* [Amazon and Atlantic Forest image datasets for semantic segmentation](https://zenodo.org/record/4498086#.Y6LPLuzP1hE) -> Sentinel 2 images. Used in paper 'An attention-based U-Net for detecting deforestation within satellite sensor imagery'
* [TreeSatAI](https://zenodo.org/records/6780578) -> Sentinel-1, Sentinel-2
* [PureForest](https://huggingface.co/datasets/IGNF/PureForest) -> VHR RGB + Near-Infrared & lidar, each patch represents a monospecific forest

## Geospatial datasets
* [Resource Watch](https://resourcewatch.org/data/explore) provides a wide range of geospatial datasets and a UI to visualise them

## Time series & change detection datasets
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) -> A Time Series Dataset for Crop Type Mapping
* The SeCo dataset contains image patches from Sentinel-2 tiles captured at different timestamps at each geographical location. [Download SeCo here](https://github.com/ElementAI/seasonal-contrast)
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
* [BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos](https://github.com/exb7900/BIRDSAI) -> Thermal IR videos of humans and animals
* [ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos](https://lcmou.github.io/ERA_Dataset/)
* [DroneVehicle](https://github.com/VisDrone/DroneVehicle) -> Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning. Annotations are rotated bounding boxes. With [Github repo](https://github.com/SunYM2020/UA-CMDet)
* [UAVOD10](https://github.com/weihancug/10-category-UAV-small-weak-object-detection-dataset-UAVOD10) -> 10 class of objects at 15 cm resolution. Classes are; building, ship, vehicle, prefabricated house, well, cable tower, pool, landslide, cultivation mesh cage, and quarry. Bounding boxes
* [Busy-parking-lot-dataset---vehicle-detection-in-UAV-video](https://github.com/zhu-xlab/Busy-parking-lot-dataset---vehicle-detection-in-UAV-video) -> Vehicle instance segmentation. Unsure format of annotations, possible Matlab specific
* [dd-ml-segmentation-benchmark](https://github.com/dronedeploy/dd-ml-segmentation-benchmark) -> DroneDeploy Machine Learning Segmentation Benchmark
* [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) -> Vision Benchmark for Maritime Search and Rescue. Bounding box object detection, single-object tracking and multi-object tracking annotations
* [aeroscapes](https://github.com/ishann/aeroscapes) -> semantic segmentation benchmark comprises of images captured using a commercial drone from an altitude range of 5 to 50 metres.
* [ALTO](https://github.com/MetaSLAM/ALTO) -> Aerial-view Large-scale Terrain-Oriented. For deep learning based UAV visual place recognition and localization tasks.
* [HIT-UAV-Infrared-Thermal-Dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) -> A High-altitude Infrared Thermal Object Detection Dataset for Unmanned Aerial Vehicles
* [caltech-aerial-rgbt-dataset](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset) -> synchronized RGB, thermal, GPS, and IMU data
* [Leafy Spurge Dataset](https://leafy-spurge-dataset.github.io/) -> Real-world Weed Classification Within Aerial Drone Imagery
* [UAV-HSI-Crop-Dataset](https://github.com/MrSuperNiu/UAV-HSI-Crop-Dataset) -> dataset for "HSI-TransUNet: A Transformer based semantic segmentation model for crop mapping from UAV hyperspectral imagery"
* [UAVVaste](https://github.com/PUTvision/UAVVaste) -> COCO-like dataset and effective waste detection in aerial images

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
* [dynnet](https://github.com/aysim/dynnet) -> DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
* [open_earth_map](https://github.com/bao18/open_earth_map) -> a benchmark dataset for global high-resolution land cover mapping
* [Satellite imagery datasets containing ships](https://github.com/NaLiu613/Satellite-Imagery-Datasets-Containing-Ships) -> A list of radar and optical satellite datasets for ship detection, classification, semantic segmentation and instance segmentation tasks
* [SolarDK](https://arxiv.org/abs/2212.01260) -> A high-resolution urban solar panel image classification and localization dataset
* [Roofline-Extraction](https://github.com/loosgagnet/Roofline-Extraction) -> dataset for paper 'Knowledge-Based 3D Building Reconstruction (3DBR) Using Single Aerial Images and Convolutional Neural Networks (CNNs)'
* [Building-detection-and-roof-type-recognition](https://github.com/loosgagnet/Building-detection-and-roof-type-recognition) -> datasets for the paper 'A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image'
* [PanCollection](https://github.com/liangjiandeng/PanCollection) -> Pansharpening Datasets from WorldView 2, WorldView 3, QuickBird, Gaofen 2 sensors
* [OnlyPlanes](https://github.com/naivelogic/OnlyPlanes) -> Synthetic dataset and pretrained models for Detectron2
* [Remote Sensing Satellite Video Dataset for Super-resolution](https://zenodo.org/record/6969604#.ZCBd-OzMJhE)
* [WHU-Stereo](https://github.com/Sheng029/WHU-Stereo) -> A Challenging Benchmark for Stereo Matching of High-Resolution Satellite Images
* [FireRisk](https://github.com/CharmonyShen/FireRisk) -> A Remote Sensing Dataset for Fire Risk Assessment with Benchmarks Using Supervised and Self-supervised Learning
* [Road-Change-Detection-Dataset](https://github.com/fightingMinty/Road-Change-Detection-Dataset)
* [3DCD](https://sites.google.com/uniroma1.it/3dchangedetection/home-page) -> infer 3D CD maps using only remote sensing optical bitemporal images as input without the need of Digital Elevation Models (DEMs)
* [Hyperspectral Change Detection Dataset Irrigated Agricultural Area](https://github.com/SicongLiuRS/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area)
* [CNN-RNN-Yield-Prediction](https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction) -> soybean dataset
* [HySpecNet-11k](https://hyspecnet.rsim.berlin/) -> a large-scale hyperspectral benchmark dataset
* [Mumbai-Semantic-Segmentation-Dataset](https://github.com/GeoAI-Research-Lab/Mumbai-Semantic-Segmentation-Dataset)
* [SZTAKI](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html) -> A Ground truth collection for change detection in optical aerial images taken with several years time differences
* [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) -> change detection dataset, it consists of six large bi-temporal high resolution images covering six cities in China
* [SV248S](https://github.com/xdai-dlgvv/SV248S) -> Single Object Tracking Dataset, tracking Vehicle, Large-Vehicle, Ship and Airplane
* [GAMUS](https://github.com/EarthNets/RSI-MMSegmentation) ->  A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data
* [Oil and Gas Infrastructure Mapping (OGIM) database](https://zenodo.org/record/7922117) -> includes locations and facility attributes of oil and gas infrastructure types that are important sources of methane emissions
* [openWUSU](https://github.com/AngieNikki/openWUSU) -> WUSU is a semantic understanding dataset focusing on urban structure and the urbanization process in Wuhan
* [Digital Typhoon Dataset](https://github.com/kitamoto-lab/digital-typhoon/) -> aimed at benchmarking machine learning models for long-term spatio-temporal data
* [RSE_Cross-city](https://github.com/danfenghong/RSE_Cross-city) -> Cross-City Matters: A Multimodal Remote Sensing Benchmark Dataset for Cross-City Semantic Segmentation using High-Resolution Domain Adaptation Networks
* [AErial Lane](https://github.com/Jiawei-Yao0812/AerialLaneNet) -> AErial Lane (AEL) Dataset is a first large-scale aerial image dataset built for lane detection, with high-quality polyline lane annotations on high-resolution images of around 80 kilometers of road
* [GeoPile pretraining dataset](https://github.com/mmendiet/GFM) -> compiles imagery from other datasets including RSD46-WHU, MLRSNet and RESISC45 for pretraining of Foundational models
* [NWPU-MOC](https://github.com/lyongo/NWPU-MOC) -> A Benchmark for Fine-grained Multi-category Object Counting in Aerial Images
* [Chesapeake Roads Spatial Context (RSC)](https://github.com/isaaccorley/ChesapeakeRSC)
* [STARCOP dataset: Semantic Segmentation of Methane Plumes with Hyperspectral Machine Learning Models](https://zenodo.org/records/7863343)
* [Toulouse Hyperspectral Data Set](https://www.toulouse-hyperspectral-data-set.com/)
* [CloudTracks: A Dataset for Localizing Ship Tracks in Satellite Images of Clouds](https://zenodo.org/records/10042922) -> the dataset consists of 1,780 MODIS satellite images hand-labeled for the presence of more than 12,000 ship tracks.
* [Vehicle Perception from Satellite](https://github.com/Chenxi1510/Vehicle-Perception-from-Satellite-Videos) -> a large-scale benchmark for traffic monitoring from satellite
* [SARDet-100K](https://github.com/zcablii/SARDet_100K) -> Large-Scale Synthetic Aperture Radar (SAR) Object Detection
* [So2Sat-POP-DL](https://github.com/zhu-xlab/So2Sat-POP-DL) -> Dataset discovery: So2Sat Population dataset covering 98 EU cities
* [Urban Vehicle Segmentation Dataset (UV6K)](https://zenodo.org/records/8404754)
* [TimeMatch](https://zenodo.org/records/5636422) -> dataset for cross-region adaptation for crop identification from SITS in four different regions in Europe
* [BirdSAT](https://github.com/mvrl/BirdSAT) -> Cross-View iNAT Birds 2021: This cross-view birds species dataset consists of paired ground-level bird images and satellite images, along with meta-information associated with the iNaturalist-2021 dataset.
* [OpenSARWake](https://github.com/libzzluo/OpenSARWake) -> A SAR ship wake rotation detection benchmark dataset.
* [TUE-CD](https://github.com/RSMagneto/MSI-Net) -> A change detection detection for building damage estimation after earthquake
* [Overhead Wind Turbine Dataset - NAIP](https://zenodo.org/records/7385227#.Y419qezMLdr)
* [Toulouse Hyperspectral Data Set](https://github.com/Romain3Ch216/TlseHypDataSet)
* [Hi-UCD](https://github.com/Daisy-7/Hi-UCD-S) -> ultra-High Urban Change Detection for urban semantic change detection
* [LEVIR-CC-Dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) -> A Large Dataset for Remote Sensing Image Change Captioning
* [ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet) -> A Large-scale Fine-Grained Dataset for Ship Detection in High-Resolution Optical Remote Sensing Images
* [pangaea-bench](https://github.com/yurujaja/pangaea-bench) -> A Global and Inclusive Benchmark for Geospatial Foundation Models
* [VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding](https://vrsbench.github.io/)
* [SeeFar](https://coastalcarbon.ai/seefar) -> Satellite Agnostic Multi-Resolution Dataset for Geospatial Foundation Models
* [RSHaze+](https://zenodo.org/records/13837162) -> remote sensing dehazing datasets in PhDnet: A novel physic-aware dehazing network for remote sensing images
* [GDCLD](https://zenodo.org/records/13612636) -> A globally distributed dataset of coseismic landslide mapping via multi-source high-resolution remote sensing images
* [10,000 Crop Field Boundaries across India](https://zenodo.org/records/7315090) -> using Airbus SPOT

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
* [satellite-deforestation](https://github.com/drewhibbard/satellite-deforestation) -> Using Satellite Imagery to Identify the Leading Indicators of Deforestation, applied to the Kaggle Challenge Understanding the Amazon from Space

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
* [Detecting ships in satellite imagery: five years later…](https://medium.com/artificialis/detecting-ships-in-satellite-imagery-five-years-later-28df2e83f987)
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined
* [Lessons Learned from Kaggle’s Airbus Challenge](https://towardsdatascience.com/lessons-learned-from-kaggles-airbus-challenge-252e25c5efac)
* [Airbus-Ship-Detection](https://github.com/kheyer/Airbus-Ship-Detection) -> This solution scored 139 out of 884 for the competition, combines ResNeXt50 based classifier and a U-net segmentation model
* [Ship-Detection-Project](https://github.com/ZTong1201/Ship-Detection-Project) -> uses Mask R-CNN and UNet model
* [Airbus_SDC](https://github.com/WillieMaddox/Airbus_SDC)
* [Airbus_SDC_dup](https://github.com/WillieMaddox/Airbus_SDC_dup) -> Project focused on detecting duplicate regions of overlapping satellite imagery. Applied to Airbus ship detection dataset
* [airbus-ship-detection](https://github.com/jancervenka/airbus-ship-detection) -> CNN with REST API
* [Ship-Detection-from-Satellite-Images-using-YOLOV4](https://github.com/debasis-dotcom/Ship-Detection-from-Satellite-Images-using-YOLOV4) -> uses Kaggle Airbus Ship Detection dataset
* [Image Segmentation: Kaggle experience](https://towardsdatascience.com/image-segmentation-kaggle-experience-9a41cb8924f0) -> Medium article by gold medal winner Vlad Shmyhlo

### Kaggle - Shipsnet classification dataset
* https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery -> Classify ships in San Franciso Bay using Planet satellite imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, 3 meter pixel size
* [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning

### Kaggle - Ships in Google Earth
* https://www.kaggle.com/datasets/tomluther/ships-in-google-earth
* 794 jpegs showing various sized ships in satellite imagery, annotations in Pascal VOC format for object detection models
* [/kaggle-ships-in-satellite-imagery-with-YOLOv8](https://github.com/robmarkcole/kaggle-ships-in-satellite-imagery-with-YOLOv8)

### Kaggle - Ships in San Franciso Bay
* https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, provided by Planet
* [DeepLearningShipDetection](https://github.com/PenguinDan/DeepLearningShipDetection)
* [Ship-Detection-Using-Satellite-Imagery](https://github.com/Dhruvisha29/Ship-Detection-Using-Satellite-Imagery)

### Kaggle - Swimming pool and car detection using satellite imagery
* https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection
* 3750 satellite images of residential areas with annotation data for swimming pools and cars
* [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5)

### Kaggle - Planesnet classification dataset
* https://www.kaggle.com/datasets/rhammell/planesnet -> Detect aircraft in Planet satellite image chips
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
* https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery
* 72 satellite images of Dubai, the UAE, and is segmented into 6 classes
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset
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
* [Sat4](https://www.kaggle.com/datasets/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three**
* [Sat6](https://www.kaggle.com/datasets/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**

### Kaggle - High resolution ship collections 2016 (HRSC2016)
* https://www.kaggle.com/datasets/guofeng/hrsc2016
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
* https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage
* https://github.com/dbuscombe-usgs/HurricaneHarvey_buildingdamage

### Kaggle - Austin Zoning Satellite Images
* https://www.kaggle.com/datasets/franchenstein/austin-zoning-satellite-images
* classify a images of Austin into one of its zones, such as residential, industrial, etc. 3667 satellite images

### Kaggle - Statoil/C-CORE Iceberg Classifier Challenge
Classify the target in a SAR image chip as either a ship or an iceberg. The dataset for the competition included 5000 images extracted from multichannel SAR data collected by the Sentinel-1 satellite. Top entries used ensembles to boost prediction accuracy from about 92% to 97%.
* https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
* [An interview with David Austin: 1st place winner](https://pyimagesearch.com/2018/03/26/interview-david-austin-1st-place-25000-kaggles-popular-competition/)
* [radar-image-recognition](https://github.com/siarez/radar-image-recognition)
* [Iceberg-Classification-Using-Deep-Learning](https://github.com/mankadronit/Iceberg-Classification-Using-Deep-Learning) -> uses keras
* [Deep-Learning-Project](https://github.com/singh-shakti94/Deep-Learning-Project) -> uses keras
* [iceberg-classifier-challenge solution by ShehabSunny](https://github.com/ShehabSunny/iceberg-classifier-challenge) -> uses keras
* [Analyzing Satellite Radar Imagery with Deep Learning](https://uk.mathworks.com/company/newsletters/articles/analyzing-satellite-radar-imagery-with-deep-learning.html) -> by Matlab, uses ensemble with greedy search
* [16th place solution](https://github.com/sergeyshilin/kaggle-statoil-iceberg-classifier-challenge)
* [fastai solution](https://github.com/smarkochev/ds_notebooks/blob/master/Statoil_Kaggle_competition_google_colab_notebook.ipynb)

### Kaggle - Land Cover Classification Dataset from DeepGlobe Challenge - segmentation
* https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)
* [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net, uses DeepGlobe Land Cover Segmentation dataset, with [code](https://github.com/TarunKumar1995-glitch/land_cover_classification_unet)
* [DeepGlobe Land Cover Classification Challenge solution](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)

### Kaggle - Next Day Wildfire Spread
A Data Set to Predict Wildfire Spreading from Remote-Sensing Data
* https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
* https://arxiv.org/abs/2112.02447

### Kaggle - Satellite Next Day Wildfire Spread
Inspired by the above dataset, using different data sources
* https://www.kaggle.com/datasets/satellitevu/satellite-next-day-wildfire-spread
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
* https://www.kaggle.com/datasets/datamunge/overheadmnist -> kaggle
* https://arxiv.org/abs/2102.04266 -> paper
* https://github.com/reveondivad/ov-mnist -> github

## Kaggle - Satellite Image Classification
* https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
* [satellite-image-classification-pytorch](https://github.com/dilaraozdemir/satellite-image-classification-pytorch)

## Kaggle - EuroSAT - Sentinel-2 Dataset
* https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset
* RGB Land Cover and Land Use Classification using Sentinel-2 Satellite
* Used in paper [Image Augmentation for Satellite Images](https://arxiv.org/abs/2207.14580)

## Kaggle - Satellite Images of Water Bodies
* https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies
* [pytorch-waterbody-segmentation](https://github.com/gauthamk02/pytorch-waterbody-segmentation) -> UNET model trained on the Satellite Images of Water Bodies dataset from Kaggle. The model is deployed on Hugging Face Spaces

## Kaggle - NOAA sea lion count
* https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count
* [noaa](https://github.com/darraghdog/noaa) -> UNET, object detection and image level regression approaches

### Kaggle - miscellaneous
* https://www.kaggle.com/datasets/reubencpereira/spatial-data-repo -> Satellite + loan data
* https://www.kaggle.com/datasets/towardsentropy/oil-storage-tanks -> Image data of industrial oil tanks with bounding box annotations, estimate tank fill % from shadows
* https://www.kaggle.com/datasets/airbusgeo/airbus-wind-turbines-patches -> Airbus SPOT satellites images over wind turbines for classification
* https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes -> CGI planes object detection dataset
* https://www.kaggle.com/datasets/atilol/aerialimageryforroofsegmentation -> Aerial Imagery for Roof Segmentation
* https://www.kaggle.com/datasets/andrewmvd/ship-detection -> 621 images of boats and ships
* https://www.kaggle.com/datasets/alpereniek/vehicle-detection-from-satellite-images-data-set
* https://www.kaggle.com/datasets/sergiishchus/maxar-satellite-data -> Example Maxar data at 15 cm resolution
* https://www.kaggle.com/datasets/cici118/swimming-pool-detection-algarves-landscape
* https://www.kaggle.com/datasets/donkroco/solar-panel-module -> object detection for solar panels
* https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset -> segment roads
* https://www.kaggle.com/datasets/towardsentropy/oil-storage-tanks -> Image data of industrial Oil Storage Tanks with bounding box annotations
* https://www.kaggle.com/competitions/widsdatathon2019/ -> Palm oil plantations
* https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images -> Ships/Vessels in Aerial Images
* https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects -> Aerial dataset for maritime Search and Rescue applications
* https://www.kaggle.com/datasets/yaroslavnaychuk/satelliteimagesegmentation -> Segmentation on Gaofen Satellite Image, extracted from GID-15 dataset

# Competitions
Competitions are an excellent source for accessing clean, ready-to-use satellite datasets and model benchmarks.  

* https://codalab.lisn.upsaclay.fr/competitions/9603 -> object detection from diversified satellite imagery
* https://www.drivendata.org/competitions/143/tick-tick-bloom/ -> detect and classify algal bloom
* https://www.drivendata.org/competitions/81/detect-flood-water/ -> map floodwater from radar imagery
* https://platform.ai4eo.eu/enhanced-sentinel2-agriculture -> map cultivated land using Sentinel imagery
* https://www.diu.mil/ai-xview-challenge -> multiple challenges ranging from detecting fishing vessals to estimating building damages
* https://competitions.codalab.org/competitions/30440 -> flood detection
* https://www.drivendata.org/competitions/83/cloud-cover/ -> cloud cover detection
* https://www.drivendata.org/competitions/78/overhead-geopose-challenge/page/372/ -> predicts geocentric pose from single-view oblique satellite images
* https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/ -> building segmentation
* https://captain-whu.github.io/DOTA/ -> large dataset for object detection in aerial imagery
* https://spacenet.ai/ -> set of 8 challenges such as road network detection
* https://huggingface.co/spaces/competitions/ChaBuD-ECML-PKDD2023 -> binary image segmentation task on forest fires monitored over California
<!-- markdown-link-check-disable -->
* https://spaceml.org/repo/project/6269285b14d764000d798fde -> ML for floods
* https://spaceml.org/repo/project/60002402f5647f00129f7287 -> lightning and extreme weather
* https://spaceml.org/repo/project/6025107d79c197001219c481/true -> ~1TB dataset for precipitation forecasting
* https://spaceml.org/repo/project/61c0a1b9ff8868000dfb79e1/true -> Sentinel-2 image super-resolution
<!-- markdown-link-check-enable --