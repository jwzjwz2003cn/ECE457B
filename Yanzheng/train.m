% ECE 457B - Winter 2013
% OCR using Neural Network and Fuzzy Logic
%
% Yanzheng Li
% Weizhong Ji
% Michael Lin
% Zhihua Song

function train(op)

global H I stats count net net1 P T

global TRAINING_PATTERNS
global TRAINING_RATE
global TRAINING_ERR_TOLERANCE
global TRAINING_HIDDEN_NODES
global TRAINING_NUM_LAYERS

TRAINING_PATTERNS = 50;
TRAINING_RATE = 0.05;
TRAINING_ERR_TOLERANCE = 1e-3;
TRAINING_HIDDEN_NODES = 10;
TRAINING_NUM_LAYERS = 3;

if nargin == 0 % if no input argument, draw the GUI
	op = 0;
end

width = 800;
height = 600;

switch op

%----------------------------------------------------------------------------------- 0.0
case 0 % Application Entry Point

    count = 0;
	InitializeControls(width, height);
    InitializeValuesToControls();
    InitializeNeuralNetwork();

%----------------------------------------------------------------------------------- 1.0
case 1 % Read and display an image

	[filename,pathname] = uigetfile({'*.png;*.tif;*.jpg','Image files'});

	if filename ~= 0

		% Clear the old data
		ClearGlobals()
		global I stats count
		set(H.edit(1),'String','');

		% Read the image and convert to intensity
        I = ReadImageAndConvertIntensity(pathname, filename);

		% Resize the first axes accordingly
		w_max = 425;
		h_max = 540;
		[h,w] = size(I.Image);
		[im_width,im_height] = fit_im(w_max,h_max,w,h);
		left = 20 + (w_max - im_width)/2;
		bott = 20 + (h_max - im_height)/2;

		% Display the image in the first axes
		colormap(gray(256));
		axes(H.ax(1));
		set(H.ax(1),'position',[left/width bott/height im_width/width im_height/height]);
		image(I.Image);
		set(H.ax(1),'XTick',[],'YTick',[]);

	end

%----------------------------------------------------------------------------------- 2.0
case 2 % Create binary image; label and segment it; find features

    global count

	count = 1;
	thresh = 125;

	I.bw = (I.Image < thresh);

	% Store pre-existing Class labels, if they exist
	try
		len = length(stats);
		[temp(1:len,1).Class] = deal(stats.Class);
		stored_class = 1;
	catch
		stored_class = 0;
    end

	% Label the regions (characters)
    L = [];
	[L,num] = bwlabel(I.bw);
    I.label = L;
	I.NumChar = num;

	colormap(gray(2))
	axes(H.ax(1))
	image(I.bw)
	set(H.ax(1),'XTick',[],'YTick',[])

	% Segment the labeled image into character sub-images
	stats = SegmentImageIntoChars(I);
	P = FormInputMatrixForNetwork(stats);

	% Fix the Orientation property
	ang = [stats.Orientation];
	ang = (180+ang).*(ang < 0) + ang.*(ang > 0);
	ang = num2cell(ang');
	len = length(stats);
	[stats.Orientation] = deal(ang{:});

	% Display character and initialize the character class labels	
	disp_char(stats,H,count,width,height);
	if stored_class
		[stats(1:len,1).Class] = deal(temp.Class);
		set(H.edit(1),'String',stats(count).Class);
	else
		stats(1).Class = [];
    end

%----------------------------------------------------------------------------------- 3.0
case 3	% Go to next character image

	count = count + 1;	% sub-image (character) to display
	if count > I.NumChar
		count = I.NumChar;
	else
		disp_char(stats,H,count,width,height)		
	end

	set(H.edit(1),'String',stats(count).Class)

%----------------------------------------------------------------------------------- 4.0
case 4	% Go to previous character image

	count = count - 1;	% sub-image (character) to display
	if count < 1
		count = 1;
	else
		disp_char(stats,H,count,width,height)
	end

	set(H.edit(1),'String',stats(count).Class)

%----------------------------------------------------------------------------------- 5.0
case 5 % Store class data from edit box

	c = get(H.edit(1),'String');
	stats(count).Class = c;

%----------------------------------------------------------------------------------- 6.0	
case 6 % Save Image and feature data to a file

	[filename,pathname] = uiputfile({'*.mat','Data files'});

	if filename ~= 0
		
		% Save the stats to a mat file
		cd(pathname);
		save(filename,'I','stats');

    end

%----------------------------------------------------------------------------------- 7.0
case 7 % Load image and feature data from a file
	
	[filename,pathname] = uigetfile({'*.mat','Data files'});
	
	if filename ~= 0
		
		% Clear the old data
% 		clear I stats count
        ClearGlobals();
		global H I stats count

		cd(pathname);
		load(filename,'I','stats')

		count = 1;	% sub-image (character) to display

		% Resize the first axes accordingly
		w_max = 425;
		h_max = 540;
		[h,w] = size(I.bw);
		[im_width,im_height] = fit_im(w_max,h_max,w,h);
		left = 20 + (w_max - im_width)/2;
		bott = 20 + (h_max - im_height)/2;

		% Display the image in the first axes
		colormap(gray(2))
		axes(H.ax(1))
		set(H.ax(1),'position',[left/width bott/height im_width/width im_height/height])
		image(I.bw)
		set(H.ax(1),'XTick',[],'YTick',[])
		
		disp_char(stats,H,count,width,height)
		set(H.edit(1),'String',stats(1).Class)

    end

%----------------------------------------------------------------------------------- 8.0
case 8 % Train network

    global H I stats count net net1 P T

    T = GetTargetForNetwork(stats);

    disp(size(T));
    disp(size(P));

    net1 = train(net, P, T); % Train the network.

%     a = sim(net1, P);
%     p = plot(P, a-T, P, T);
%     set(p,'Color','red','LineWidth',52,'LineHeight', 50);
%     grid;

%----------------------------------------------------------------------------------- 9.0
case 9 % Simulate network

    global net1
    [filename,pathname] = uigetfile({'*.png;*.tif;*.jpg','Image files'});

    I_ = ReadImageAndConvertIntensity(pathname, filename);

    stats_ = SegmentImageIntoChars(I_);
    input = FormInputMatrixForNetwork(stats_);

    output = net1(input);
    errors = gsubtract(T, output);
    performance = perform(net1, T, output);
    
    disp(output);

    len = length(output);
    for n = 1:len
        disp(char(output(n) + '0'));
    end
    
case 11 % Accept parameters for network training
    global H

    global TRAINING_PATTERNS
    global TRAINING_RATE
    global TRAINING_ERR_TOLERANCE
    global TRAINING_HIDDEN_NODES
    global TRAINING_NUM_LAYERS

    TRAINING_PATTERNS = str2num(get(H.edit(2),'String'));
    TRAINING_RATE = str2num(get(H.edit(3),'String'));
    TRAINING_ERR_TOLERANCE = str2num(get(H.edit(4),'String'));
    TRAINING_HIDDEN_NODES = str2num(get(H.edit(5),'String'));
    TRAINING_NUM_LAYERS = str2num(get(H.edit(6),'String'));

    InitializeNeuralNetwork();

end % end of train()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sub-functions

%-----------------------------------------------------------------------------------
function disp_char(stats,H,count,width,height)
% Display a subimage (character)

% Resize the second axes accordingly
w_max = 100;
h_max = 100;
[h,w] = size(stats(count).Image);
[im_width,im_height] = fit_im(w_max,h_max,w,h);
left = 500 + (w_max - im_width)/2;
bott = 190 + (h_max - im_height)/2;

% Display the character
if gca ~= H.ax(2)
	axes(H.ax(2))
end
image(stats(count).Image);
set(H.ax(2),'position',[left/width bott/height im_width/width im_height/height])
set(H.ax(2),'XTick',[],'YTick',[]);


%-----------------------------------------------------------------------------------
function [im_width,im_height] = fit_im(w_max,h_max,w,h)
% Resize the image accordingly

w_ratio = w/w_max;
h_ratio = h/h_max;

if (w_ratio > 1) | (h_ratio > 1)
	if w_ratio > h_ratio
		im_width = w_max;
		im_height = h/w_ratio;
	else
		im_height = h_max;
		im_width = w/h_ratio;			
	end
else
	im_width = w;
	im_height = h;
end


%-----------------------------------------------------------------------------------
function c = GetLabel(edit_box)
% Get the class label from the edit box

c = get(edit_box,'String');
set(edit_box,'String',[]);

%-----------------------------------------------------------------------------------
function I = ReadImageAndConvertIntensity(pathname, filename)
% Read image and convert intensity

[I.Image,map] = imread([pathname filename]);
cd(pathname);

if ~isempty(map)
    I.Image = ind2gray(I.Image,map);
    I.Image = gray2ind(I.Image,256)
else
    if size(I.Image,3)==3
        I.Image = rgb2gray(I.Image);
    end
end

thresh = 125;
I.bw = (I.Image < thresh);

% Store pre-existing Class labels, if they exist
try
    len = length(stats);
    [temp(1:len,1).Class] = deal(stats.Class);
    stored_class = 1;
catch
    stored_class = 0;
end

% Label the regions (characters)
L = [];
[L,num] = bwlabel(I.bw);
I.label = L;
I.NumChar = num;

function ClearGlobals()
% Clear global variables

clear global I stats count P T

function stats = SegmentImageIntoChars(I)
% Segment the labeled image into character sub-images

stats = regionprops(I.label,'Image','Eccentricity','Orientation');

num = length(stats);
for n = 1:num
    stats(n).Image = imresize(stats(n).Image, [15, 15]);
end

function InitializeControls(width, height)
% Initialize controls on the window.

global H

s = get(0,'ScreenSize');
% ------------------------------------------------------------------------------- 0.1
% --------------------------------  FIGURE & MENUS  ---------------------------------
H.fig = figure(...
    'Position',[(s(3)-width)/2 (s(4)-height)/2 width height],...
    'NumberTitle','off',...
    'MenuBar','none',...
    'Color',[.8 .8 .8],...
    'Name','OCR Neural Network Trainer');

H.menu(1) = uimenu(H.fig,'Label','&Image');
H.menu(2) = uimenu(H.menu(1),'Label','&Open','Callback','train(1)');
H.menu(3) = uimenu(H.fig,'Label','&Data');	
H.menu(4) = uimenu(H.menu(3),'Label','&Load','Callback','train(7)');
H.menu(5) = uimenu(H.menu(3),'Label','&Save','Callback','train(6)');

% ------------------------------------------------------------------------------- 0.2
% ---------------------------------  IMAGE FRAME  -----------------------------------

H.ax(1) = axes(...
    'position',[20/width 210/height 0.5+25/width 360/height],...
    'XTick',[],'YTick',[],'Box','on');

uicontrol(...
    'Style','text',... %
    'BackgroundColor',[.8 .8 .8],...
    'Units','normalized',...
    'Position',[20/width (height-30)/height 0.5+25/width 20/height],...
    'String','Training Image',...
    'HorizontalAlignment','center',...
    'FontSize',10);

% ------------------------------------------------------------------------------- 0.3
% -------------------------  CLASSIFIER TRAINING FRAME  -----------------------------	
H.ax(2) = axes(...
    'position',[0.5+80/width 220/height 0.5-300/width 1-520/height],...
    'XTick',[],'YTick',[],'Box','on');	

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[20/width 170/height 0.5+25/width 30/height],...
    'String','Simulation Result',...
    'HorizontalAlignment','center',...
    'FontSize',10);

H.ax(3) = axes(...
    'position',[20/width 20/height 0.5+25/width 150/height],...
    'XTick',[],'YTick',[],'Box','on');	

uicontrol(...
    'Style','frame',...
    'Units','normalized',...
    'Position',[480/width 320/height 300/width 1-360/height]);

H.button(1) = uicontrol(...
    'Style','pushbutton',...
    'Units','normalized',...
    'Position',[540/width (height-120)/height 180/width 50/height],...
    'String','Binarize and Segment Image',...
    'HorizontalAlignment','left',...
    'FontSize',10,...
    'CallBack', 'train(2)');

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[485/width (height-180)/height 290/width 30/height],...
    'String','- - - - - - - Character navigation - - - - - - -',...
    'HorizontalAlignment','center',...
    'FontSize',10);

H.button(2) = uicontrol(...
    'Style','pushbutton',... % (Next) Character navigation
    'Units','normalized', ...
    'Position',[640/width (height-210)/height 30/width 30/height],...
    'FontSize',10,...
    'String','>>',...
    'CallBack','train(3)');

H.button(3) = uicontrol(...
    'Style','pushbutton',... % (Previous) Character navigation
    'Units','normalized', ...
    'Position',[590/width (height-210)/height 30/width 30/height],...
    'FontSize',10,...
    'String','<<',...
    'CallBack','train(4)');

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[510/width (height-260)/height 90/width 30/height],...
    'String','Label Character:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(1) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[600/width (height-260)/height 90/width 30/height],...
    'HorizontalAlignment','left',...
    'FontSize',10,...
    'CallBack','train(5)');

% -----------------------------------------------------------------------------------
% -----------------------------  CHARACTER FEATURES  --------------------------------
H.button(4) = uicontrol(...
    'Style','pushbutton',...
    'Units','normalized', ...
    'Position',[640/width 260/height 90/width 30/height],...
    'FontSize',10,...
    'String','Train Network',...
    'CallBack','train(8)');

H.button(5) = uicontrol(...
    'Style','pushbutton',...
    'Units','normalized', ...
    'Position',[640/width 220/height 90/width 30/height],...
    'FontSize',10,...
    'String','Simulate',...
    'CallBack','train(9)');

% -----------------------------------------------------------------------------------
% -----------------------------  CHARACTER FEATURES  --------------------------------

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[480/width 160/height 100/width 25/height],...
    'String','Traing Patterns:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(2) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[640/width 160/height 120/width 25/height],...
    'HorizontalAlignment','right',...
    'FontSize',10);

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[480/width 130/height 90/width 25/height],...
    'String','Traing Rate:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(3) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[640/width 130/height 120/width 25/height],...
    'HorizontalAlignment','right',...
    'FontSize',10);

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[480/width 100/height 150/width 25/height],...
    'String','Traing Error Tolerance:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(4) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[640/width 100/height 120/width 25/height],...
    'HorizontalAlignment','right',...
    'FontSize',10);

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[480/width 70/height 150/width 25/height],...
    'String','Traing # Hidden Nodes:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(5) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[640/width 70/height 120/width 25/height],...
    'HorizontalAlignment','right',...
    'FontSize',10);

uicontrol(...
    'Style','text',...
    'Units','normalized',...
    'Position',[480/width 40/height 150/width 25/height],...
    'String','Traing # Layers:',...
    'HorizontalAlignment','left',...
    'FontSize',10);

H.edit(6) = uicontrol(...
    'Style','edit',...
    'Units','normalized',...
    'Position',[640/width 40/height 120/width 25/height],...
    'HorizontalAlignment','right',...
    'FontSize',10);

H.button(6) = uicontrol(... % Accept button
    'Style','pushbutton',...
    'Units','normalized', ...
    'Position',[660/width 5/height 100/width 25/height],...
    'FontSize',10,...
    'String','Accept',...
    'CallBack','train(11)');

function InitializeValuesToControls()

    global H
    global TRAINING_PATTERNS
    global TRAINING_RATE
    global TRAINING_ERR_TOLERANCE
    global TRAINING_HIDDEN_NODES
    global TRAINING_NUM_LAYERS

    set(H.edit(2),'String', TRAINING_PATTERNS);
    set(H.edit(3),'String', TRAINING_RATE);
    set(H.edit(4),'String', TRAINING_ERR_TOLERANCE);
    set(H.edit(5),'String', TRAINING_HIDDEN_NODES);
    set(H.edit(6),'String', TRAINING_NUM_LAYERS);

function InitializeNeuralNetwork()
    global net

    global TRAINING_PATTERNS
    global TRAINING_RATE
    global TRAINING_ERR_TOLERANCE
    global TRAINING_HIDDEN_NODES
    global TRAINING_NUM_LAYERS

    net = feedforwardnet(TRAINING_HIDDEN_NODES, 'traingd');
    net.numLayers = TRAINING_NUM_LAYERS;
    net.numInputs = 1;
    net.inputConnect(1) = 1;
    net.layerConnect(2,1) = 1;
    net.layerConnect(3,2) = 1;
    net.outputConnect(1:2) = 0;
    net.outputConnect(2) = 1;
    net.performFcn = 'sae';
    net.divideFcn = 'dividetrain';
    net = init(net);
    net.trainParam.show = 100; % show result every 100 iterations.
    net.trainParam.lr = TRAINING_RATE; % Learning rate
    net.trainParam.epochs = TRAINING_PATTERNS; % Max number of iterations.TR
    net.trainParam.goal = TRAINING_ERR_TOLERANCE; % Error tolerance.

function P = FormInputMatrixForNetwork(stats)

num = length(stats);

P = [];

for n = 1:num
    row = stats(n).Image(:)';
    if size(P) == 0
        P = [row];
    else
        P = [P;row];
    end
end

P = P';

function T = GetTargetForNetwork(stats)

T = [];
num = length(stats);
for n = 1:num
    T = [T, stats(n).Class - '0'];
end









