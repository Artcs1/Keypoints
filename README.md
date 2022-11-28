# Keypoints

This is the official repository of the OMNICV-CVPR2022 workshop named "Pose Estimation for Two-View Panoramas: a Comparative Analysis"

Visit our paper:

[IEEE link](https://ieeexplore.ieee.org/document/9857392) 

[CVF access](https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/papers/Murrugarra-Llerena_Pose_Estimation_for_Two-View_Panoramas_Based_on_Keypoint_Matching_A_CVPRW_2022_paper.pdf)

## Installation

1. Create a conda enviorenment

```
conda create --name py37-keypoints python=3.7
conda activate py37-keypoints
```

2. Install the Spherical Package from [Meder github](https://github.com/meder411/Spherical-Package)

3. Git clone the project and enter the folder


```
git clone https://github.com/Artcs1/Keypoints.git
cd Keypoints
```

4. Install the requirements

```
pip install -r requirements.txt
```
5. Install [liegroups](https://github.com/utiasSTARS/liegroups) package in utils directory

6. Compile the SPHORB package

```
cd SPHORB-master
conda create --name SPHORB python=3.7
conda activate SPHORB
conda install -c conda-forge opencv==3.4.2
mkdir build1
cd build1
cmake ..
make
conda deactivate
```

7. Compile the five-point algorithm package

```
cd fivepoint
conda create --name FP python=3.7
conda activate FP
conda install -c conda-forge opencv==4.5.5
mkdir build
cd build
cmake ..
make
conda deactivate
```

## Usage

1. Activate the conda enviorenment

```
conda activate py37-keypoints
```

2. Run extrack_keypoints.py file

```
python3 extract_keypoints.py --datas [Urban1|Urban2|Urban3|Urban4|Room|Classroom|Realistic|Interior1|Interior2] --descriptors [sift|tsift|orb|torb|spoint|tspoint|sphorb]
```

3. Example for Outdoor dataset

```
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors  sift tsift orb torb spoint tspoint sphorb
```

4. Example for Indoor dataset

```
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2--descriptors  sift tsift orb torb spoint tspoint sphorb
```


## Benchmark and results

### 100 images per dataset

[Downloand link](https://www.inf.ufrgs.br/panorama/data_100.zip)

| Descriptors  | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
| git status   | git status     | git status    |



### 1000 images per dataset

[Downloand link](https://www.inf.ufrgs.br/panorama/data_1000.zip)

## Cite our work

```
@InProceedings{Murrugarra-Llerena_2022_CVPR,
    author    = {Murrugarra-Llerena, Jeffri and da Silveira, Thiago L. T. and Jung, Claudio R.},
    title     = {Pose Estimation for Two-View Panoramas Based on Keypoint Matching: A Comparative Study and Critical Analysis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {5202-5211}
}
```

