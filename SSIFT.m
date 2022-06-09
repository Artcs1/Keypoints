% SIFTSdemo2
%
% DESCRIPTION:
%  Example of computation of Local Spherical Descriptors of a spherical
%  omnidirectional image
%
% VERSION:
%  1.1
%
% AUTHOR:
%  Javier Cruz-Mota, email: javier.cruz@epfl.ch
%  Copyright (C) 2011 EPFL (Ecole Polytechnique Fédérale de Lausanne)
%  Transport and Mobility Laboratoy (TRANSP-OR)
% 
% LICENSE:
%  SIFTS is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%  Please refer to the file COPYING for more information.
%
% REFERENCE:
%  @TECHREPORT{CruzBogdPaquBierThir09,
%     Author = {Javier Cruz-Mota and Iva Bogdanova and Beno{\^\i}t Paquier and Michel Bierlaire and Jean-Philippe Thiran},
%     Institution = {Transport and Mobility Laboratory, Ecole Polytechnique F\'ed\'erale de Lausanne (EPFL)},
%	  Title = {Scale Invariant Feature Transform on the Sphere: Theory and Applications},
%     Year = {2009}
%  }
function [] = SSIFT(img1, img2, destiny_path)

    global SIFTSLocalPath;
    % LadyBug2 image:
    
    
    nPoints = 1000
    
    % Spherical image size
    sphSize = 720;
    
    imageStr = [img1];
    
    % Compute the spherical image
    ladybugImage = imread(imageStr);
    
    disp(size(ladybugImage))
         origImg = LADYBUGToSphere_colour(ladybugImage, sphSize);
          sphImg = double(rgb2gray(origImg));
          sphImg = sphImg - min(min(sphImg));
          sphImg = sphImg/max(max(sphImg));
    
    % Double-size image
    % octaveOffset = -1;
    % Original-size image
     octaveOffset = 0;
    
    % Number of intervals per octave
    S = 3;
    
    % Nominal and Base sigmas (in relative distance between pixels)
    sigmaN = 0.5;
    sigma0 = 3.0;
    
    % Number of stages in the pyramid
    numOctaves = floor(log2(sphSize))-5;
    
    % Spherical Scale-Space and DoG computation
     SSS = sphericalScaleSpace(sphImg, S, numOctaves, octaveOffset, sigma0, sigmaN);
    SDoG = differenceOfGaussians(SSS);
    
    % Find local maxima/minima in the spherical DoG
    [extremePoints, numPoints] = findLocalExtrema(SDoG);
    
    nPoints = min(nPoints, numPoints)
    %nPoints = numPoints
    % Compute Local Spherical Descriptors (LSD)
    numBinsPrincipalAngle = 36;
                  numBins = 8;
      numSpatialDivisions = 4;
     [LSD1, extremePoints1] = computeLSD(extremePoints, SSS, numBinsPrincipalAngle, numBins, numSpatialDivisions);
    
    
    [values, indices] = sort(extremePoints1.gradientAngle,'descend');
    indices = indices(1:nPoints);
    LSD1 = LSD1(indices);
    
    imageStr = [img2];
    
    % Compute the spherical image
    ladybugImage = imread(imageStr);
         origImg = LADYBUGToSphere_colour(ladybugImage, sphSize);
          sphImg = double(rgb2gray(origImg));
          sphImg = sphImg - min(min(sphImg));
          sphImg = sphImg/max(max(sphImg));
    
    % Double-size image
    % octaveOffset = -1;
    % Original-size image
     octaveOffset = 0;
    
    % Number of intervals per octave
    S = 3;
    
    % Nominal and Base sigmas (in relative distance between pixels)
    sigmaN = 0.5;
    sigma0 = 3.0;
    
    % Number of stages in the pyramid
    numOctaves = floor(log2(sphSize))-5;
    
    % Spherical Scale-Space and DoG computation
     SSS = sphericalScaleSpace(sphImg, S, numOctaves, octaveOffset, sigma0, sigmaN);
    SDoG = differenceOfGaussians(SSS);
    
    % Find local maxima/minima in the spherical DoG
    [extremePoints, numPoints] = findLocalExtrema(SDoG);
    
    nPoints = min(nPoints, numPoints)
    
    %nPoints = numPoints
    
    % Compute Local Spherical Descriptors (LSD)
    numBinsPrincipalAngle = 36;
                  numBins = 8;
      numSpatialDivisions = 4;
     [LSD2, extremePoints2] = computeLSD(extremePoints, SSS, numBinsPrincipalAngle, numBins, numSpatialDivisions);
    
    
    
    [values, indices] = sort(extremePoints2.gradientAngle,'descend');
    indices = indices(1:nPoints);
    LSD2 = LSD2(indices);
   
    disp(destiny_path)
    
    fid = fopen(strcat(destiny_path,'./puntos1.dat'),'w');
    for c = 1:length(LSD1);
        
        LSD1(c,1).imgOptPoints(1) = LSD1(c,1).imgOptPoints(1)-1;
        LSD1(c,1).imgOptPoints(2) = LSD1(c,1).imgOptPoints(2)-1;
    
        LSD1(c,1).imgOptPoints(2) = LSD1(c,1).imgOptPoints(2)*2;
        LSD1(c,1).imgOptPoints = fliplr(LSD1(c,1).imgOptPoints);
        fprintf(fid,'%f %f 1\n', LSD1(c,1).imgOptPoints);
    end
    
    
    fid = fopen(strcat(destiny_path,'./puntos2.dat'),'w');
    for c = 1:length(LSD2);
        LSD2(c,1).imgOptPoints(1) = LSD2(c,1).imgOptPoints(1)-1;
        LSD2(c,1).imgOptPoints(2) = LSD2(c,1).imgOptPoints(2)-1;
        LSD2(c,1).imgOptPoints(2) = LSD2(c,1).imgOptPoints(2)*2;
        LSD2(c,1).imgOptPoints = fliplr(LSD2(c,1).imgOptPoints);
        fprintf(fid,'%f %f 1\n', LSD2(c,1).imgOptPoints);
    end
     
     
    distRatio = 0.75;
    integerMatching = true;
    samePoints = computeMatching(LSD1, LSD2, distRatio, integerMatching);
    
    fid = fopen(strcat(destiny_path,'./p1.dat'),'w');
    for c = 1:length(samePoints);
        %samePoints(c).point1(2) = samePoints(c).point1(2)*2;
        fprintf(fid,'%f %f 1\n', LSD1(samePoints(c).index1,1).imgOptPoints);
    end
    
    fid = fopen(strcat(destiny_path,'./p2.dat'),'w');
    for c = 1:length(samePoints);
        %samePoints(c).points2(2) = samePoints(c).point2(2)*2;
        fprintf(fid,'%f %f 1\n', LSD2(samePoints(c).index2,1).imgOptPoints);
    end
    
    M = [ length(LSD1), length(LSD2), length(samePoints.index1)]
    fid = fopen(strcat(destiny_path,'./mismatch.dat'),'w');
    fprintf(fid,'%f %f %f\n', M);
    
end
