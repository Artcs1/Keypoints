#!/bin/bash

source activate spherical_functions

AXIS=('X' 'Y' 'Z')
DEGREES=(0 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345)
for f in {0..45};
do
    cp ./data/test_rgb/$f.jpg ./Image/
    mv ./Image/$f.jpg ./Image/O.jpg 
    for r in "${DEGREES[@]}"
    do
        for i in "${AXIS[@]}"
        do
            python3 gerandoR.py --axis $i --degree_rot $r --img $f
            mkdir -p $2_$1/$i/$r/$f
            mogrify -format png ./Image/*.jpg
            python3 matched_superpoint_keypoints.py --opt $1  --mode $2 ./Image/O.png ./Image/R.png
            #rm ./Image/R.png
            cp puntos1.dat ./$2_$1/$i/$r/$f/
            cp puntos2.dat ./$2_$1/$i/$r/$f/
            cp p1.dat ./$2_$1/$i/$r/$f/
            cp p2.dat ./$2_$1/$i/$r/$f/
            cp mismatch.dat  ./$2_$1/$i/$r/$f/
        done
    done
    rm ./Image/O.jpg
done
