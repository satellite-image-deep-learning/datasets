# Software for working with remote sensing data
[A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code) -> lists many useful tools and resources
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery
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

# Image dataset creation
Many datasets on kaggle & elsewhere have been created by screen-clipping Google Maps or browsing web portals. The tools below are to create datasets programatically
* [MapTilesDownloader](https://github.com/AliFlux/MapTilesDownloader) -> A super easy to use map tiles downloader built using Python
* [jimutmap](https://github.com/Jimut123/jimutmap) -> get enormous amount of high resolution satellite images from apple / google maps quickly through multi-threading
* [google-maps-downloader](https://github.com/yildirimcagatay34/google-maps-downloader) -> A short python script that downloads satellite imagery from Google Maps
* [ExtractSatelliteImagesFromCSV](https://github.com/thewati/ExtractSatelliteImagesFromCSV) -> extract satellite images using a CSV file that contains latitude and longitude, uses mapbox
* [sentinelsat](https://github.com/sentinelsat/sentinelsat) -> Search and download Copernicus Sentinel satellite images
* [SentinelDownloader](https://github.com/cordmaur/SentinelDownloader) -> a high level wrapper to the SentinelSat that provides an object oriented interface, asynchronous downloading, quickview & simpler searching methods
* [GEES2Downloader](https://github.com/cordmaur/GEES2Downloader) -> Downloader for GEE S2 bands
* [Sentinel-2 satellite tiles images downloader from Copernicus](https://github.com/flaviostutz/sentinelloader) -> Minimizes data download and combines multiple tiles to return a single area of interest
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data
* [Easy Landsat Download](https://github.com/dgketchum/Landsat578)
* [A simple python scrapper to get satellite images of Africa, Europe and Oceania's weather using the Sat24 website](https://github.com/luistripa/sat24-image-scrapper)
* [RGISTools](https://github.com/spatialstatisticsupna/RGISTools) -> Tools for Downloading, Customizing, and Processing Time Series of Satellite Images from Landsat, MODIS, and Sentinel
* [DeepSatData](https://github.com/michaeltrs/DeepSatData) -> Automatically create machine learning datasets from satellite images
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [satpy](https://github.com/pytroll/satpy) -> a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [GIBS-Downloader](https://github.com/spaceml-org/GIBS-Downloader) -> a command-line tool which facilitates the downloading of NASA satellite imagery and offers different functionalities in order to prepare the images for training in a machine learning pipeline
* [eodag](https://github.com/CS-SI/eodag) -> Earth Observation Data Access Gateway
* [pylandsat](https://github.com/yannforget/pylandsat) -> Search, download, and preprocess Landsat imagery
* [landsatxplore](https://github.com/yannforget/landsatxplore) -> Search and download Landsat scenes from EarthExplorer
* [OpenSarToolkit](https://github.com/ESA-PhiLab/OpenSarToolkit) -> High-level functionality for the inventory, download and pre-processing of Sentinel-1 data in the python language
* [lsru](https://github.com/loicdtx/lsru) -> Query and Order Landsat Surface Reflectance data via ESPA
* [eoreader](https://github.com/sertit/eoreader) -> Remote-sensing opensource python library reading optical and SAR sensors, loading and stacking bands, clouds, DEM and index in a sensor-agnostic way
* [Export thumbnails from Earth Engine](https://gorelick.medium.com/fast-er-downloads-a2abd512aa26)
* [deepsentinel-osm](https://github.com/Lkruitwagen/deepsentinel-osm) -> A repository to generate land cover labels from OpenStreetMap
* [img2dataset](https://github.com/rom1504/img2dataset) -> Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap (OSM) Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data
* [sentinel2tools](https://github.com/QuantuMobileSoftware/sentinel2tools) -> downloading & basic processing of Sentinel 2 imagesry. Read [Sentinel2tools: simple lib for downloading Sentinel-2 satellite images](https://medium.com/geekculture/sentinel2tools-simple-lib-for-downloading-sentinel-2-satellite-images-f8a6be3ee894)
* [Aerial-Satellite-Imagery-Retrieval](https://github.com/chiragkhandhar/Aerial-Satellite-Imagery-Retrieval) -> A program using Bing maps tile system to automatically download Aerial / Satellite Imagery given a lat/lon bounding box and level of detail
* [google-maps-at-88-mph](https://github.com/doersino/google-maps-at-88-mph) -> Google Maps keeps old satellite imagery around for a while – this tool collects what's available for a user-specified region in the form of a GIF
* [srtmDownloader](https://github.com/Abdi-Ghasem/srtmDownloader) -> Python library (multi-threaded) for retrieving SRTM elevation map of CGIAR-CSI
* [ImageDatasetViz](https://github.com/vfdev-5/ImageDatasetViz) -> create a mosaic of images in a dataset for previewing purposes
* [landsatlinks](https://github.com/ernstste/landsatlinks) -> A simple CLI interface to generate download urls for Landsat Collection 2 Level 1 product bundles
* [pyeo](https://github.com/clcr/pyeo) -> a set of portable, extensible and modular Python scripts for machine learning in earth observation and GIS, including downloading, preprocessing, creation of base layers, classification and validation.
* [metaearth](https://github.com/bair-climate-initiative/metaearth) -> Download and access remote sensing data from any platform
* [geoget](https://github.com/mnpinto/geoget) -> Download geodata for anywhere in Earth via ladsweb.modaps.eosdis.nasa.gov
* [geeml](https://github.com/Geethen/geeml) -> A python package to extract Google Earth Engine data for machine learning

# Image chipping/tiling & merging
Since raw images can be very large, it is usually necessary to chip/tile them into smaller images before annotation & training
* [image_slicer](https://github.com/samdobson/image_slicer) -> Split images into tiles. Join the tiles back together
* [tiler by nuno-faria](https://github.com/nuno-faria/tiler) -> split images into tiles and merge tiles into a large image
* [tiler by the-lay](https://github.com/the-lay/tiler) -> N-dimensional NumPy array tiling and merging with overlapping, padding and tapering
* [xbatcher](https://github.com/pangeo-data/xbatcher) -> Xbatcher is a small library for iterating xarray DataArrays in batches. The goal is to make it easy to feed xarray datasets to machine learning libraries such as Keras
* [GeoTagged_ImageChip](https://github.com/Hejarshahabi/GeoTagged_ImageChip) -> A simple script to create geo tagged image chips from high resolution RS iamges for training deep learning models such as Unet
* [geotiff-crop-dataset](https://github.com/tayden/geotiff-crop-dataset) -> A Pytorch Dataloader for tif image files that dynamically crops the image
* [Train-Test-Validation-Dataset-Generation](https://github.com/salarghaffarian/Train-Test-Validation-Dataset-Generation) ->  app to crop images and create small patches of a large image e.g. Satellite/Aerial Images, which will then be used for training and testing Deep Learning models specifically semantic segmentation models
* [satproc](https://github.com/dymaxionlabs/satproc) -> Python library and CLI tools for processing geospatial imagery for ML
* [Sliding Window](https://github.com/adamrehn/slidingwindow) ->  break large images into a series of smaller chunks
* [patchify](https://github.com/dovahcrow/patchify.py) -> A library that helps you split image into small, overlappable patches, and merge patches into original image
* [split-rs-data](https://github.com/Youssef-Harby/split-rs-data) -> Divide remote sensing images and their labels into data sets of specified size
* [image-reconstructor-patches](https://github.com/marijavella/image-reconstructor-patches) -> Reconstruct Image from Patches with a Variable Stride
* [rpc_cropper](https://github.com/carlodef/rpc_cropper) -> A small standalone tool to crop satellite images and their RPC
* [geotile](https://github.com/iamtekson/geotile) -> python library for tiling the geographic raster data
* [GeoPatch](https://github.com/Hejarshahabi/GeoPatch) -> generating patches from remote sensing data
* [ImageTilingUtils](https://github.com/vfdev-5/ImageTilingUtils) -> Minimalistic set of image reader agnostic tools to easily iterate over large images
* [split_raster](https://github.com/cuicaihao/split_raster) -> Creates a tiled output from an input raster dataset. pip installable
* [SAHI](https://github.com/obss/sahi) -> Utilties for slicing COCO formatted annotations and image files, performing sliced inference using MMDetection, Detectron2, YOLOv5, HuggingFace detectors and calculating AP over image slices.

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
