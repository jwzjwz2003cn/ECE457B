function [page] = OCRvision(inputImage)
%parameters to fine tune for function to work
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = 50;    %for paragraph
L = [10,30];    %for line
W = [10,20];   %for word
C = [20,10];    %for char
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AST objects
WORD = [];
LINE = [];
PARAGRAPH = [];
PAGE = [];
%matlab blob analysis object
hblob = vision.BlobAnalysis( ...
    'AreaOutputPort', false, ...
    'CentroidOutputPort', false, ...
    'BoundingBoxOutputPort', true, ...
    'OutputDataType', 'single', ...
    'MaximumCount', intmax, ...
    'MinimumBlobArea', 0, ...
    'MaximumBlobArea', intmax ...
    );
%convert color image to grey to binary
greyImage = rgb2gray(inputImage);
level = graythresh(greyImage);
BINARYIMAGE = im2bw(greyImage,level);
%close binary image to create paragraph blobs, and return the coordinates
%of their bounding boxes
SE = strel('square', P);
paragraphBlob = imclose(BINARYIMAGE,SE);
paragraphBbox = step(hblob, paragraphBlob);
paragraphBbox = sortrows(paragraphBbox,2);
%for each paragraph bounding box
for i=1:size(paragraphBbox,1)
    x1=paragraphBbox(i,1);
    y1=paragraphBbox(i,2);
    x2=x1+paragraphBbox(i,3);
    y2=y1+paragraphBbox(i,4);
    PARAGRAPHIMAGE = BINARYIMAGE(y1:y2,x1:x2);
    %close binary image to create line blobs, and return the coordinates of
    %their bouding boxes
    SE = strel('rectangle', L);
    lineBlob = imclose(PARAGRAPHIMAGE,SE);
    lineBbox = step(hblob, lineBlob);
    lineBbox = sortrows(lineBbox,2);
    %for each line bounding box
    for l=1:size(lineBbox,1);
        x1=lineBbox(l,1);
        y1=lineBbox(l,2);
        x2=x1+lineBbox(l,3);
        y2=y1+lineBbox(l,4);
        LINEIMAGE = PARAGRAPHIMAGE(y1:y2,x1:x2);
        %close binary image to create word blobs, and return the coordinates of
        %their bouding boxes
        SE = strel('rectangle', W);
        wordBlob = imclose(LINEIMAGE,SE);
        wordBbox = step(hblob, wordBlob);
        wordBbox = sortrows(wordBbox,1);
        %for each word bounding box
        for j=1:size(wordBbox,1)
            x1=wordBbox(j,1);
            y1=wordBbox(j,2);
            x2=x1+wordBbox(j,3);
            y2=y1+wordBbox(j,4);
            WORDIMAGE = LINEIMAGE(y1:y2,x1:x2);
            %close binary image to create char blobs, and return the
            %coordinates of their bounding boxes
            SE = strel('rectangle', C);
            charBlob = imclose(WORDIMAGE,SE);
            charBbox = step(hblob, charBlob);
            charBbox = sortrows(charBbox,1);
            %for each char bounding box
            for k=1:size(charBbox,1)
                x1=charBbox(k,1);
                y1=charBbox(k,2);
                x2=x1+charBbox(k,3);
                y2=y1+charBbox(k,4);
                CHARIMAGE = WORDIMAGE(y1:y2,x1:x2);
                %construct word object using char image matrices
                WORD{k} = CHARIMAGE;
            end
            %use word objects to create line object
            LINE{j} = WORD;
            WORD = [];
        end
        %use line objects to create paragraph object
        PARAGRAPH{l} = LINE;
        LINE = [];
    end
    %use paragraph objects to create page object
    PAGE{i} = PARAGRAPH;
    PARAGRAPH = [];
end
%return page object for neural processing
page = PAGE;
end
