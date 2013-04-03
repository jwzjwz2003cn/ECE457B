% ECE 457B - Winter 2013
% Yanzheng Li

function train(op)

global H I stats count net P T

if nargin == 0 % if no input argument, draw the GUI
	op = 0;
end

width = 800;
height = 600;

switch op

%----------------------------------------------------------------------------------- 0.0
case 0 % Draw figure
	
	count = 0;
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
    'position',[20/width 20/height 0.5+25/width 1-60/height],...
		'XTick',[],'YTick',[],'Box','on');
	
	uicontrol('Style','text',... %
		'BackgroundColor',[.8 .8 .8],...
		'Units','normalized',...
		'Position',[20/width (height-30)/height 0.5+25/width 20/height],...
		'String','Training Image',...
		'HorizontalAlignment','center',...
		'FontSize',10);
	
	% ------------------------------------------------------------------------------- 0.3
	% -------------------------  CLASSIFIER TRAINING FRAME  -----------------------------	
	H.ax(2) = axes(...
    'position',[0.5+180/width 190/height 0.5-300/width 1-500/height],...
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

	% ------------------------------------------------------------------------------- 0.4
	% -----------------------------  CHARACTER FEATURES  --------------------------------
    H.button(4) = uicontrol(...
        'Style','pushbutton',... % Determine Character Features
        'Units','normalized', ...
        'Position',[510/width 150/height 90/width 30/height],...
        'FontSize',10,...
        'String','Train Network',...
        'CallBack','train(8)');	
    
%     H.button(5) = uicontrol(...
%         'Style','pushbutton',... % Determine Character Features
%         'Units','normalized', ...
%         'Position',[480/width 130/height 30/width 30/height],...
%         'FontSize',10,...
%         'String','Character Features',...
%         'CallBack','train(9)');	

% 	H.checkbox(1) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[530/width 150/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','Eccentricity',...
% 		'FontSize',10);
% 	
% 	H.checkbox(2) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[530/width 100/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','Orientation',...
% 		'FontSize',10);
% 		
% 	H.checkbox(3) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[530/width 70/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','HPSkewness',...
% 		'FontSize',10);
% 	
% 	H.checkbox(4) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[530/width 40/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','VPSkewness',...
% 		'FontSize',10);
% 
% 	H.checkbox(5) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[650/width 130/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','HPKurtosis',...
% 		'FontSize',10);
% 	
% 	H.checkbox(6) = uicontrol(...
%         'Style','checkbox',...
% 		'Units','normalized', ...
% 		'Position',[650/width 100/height 120/width 30/height],...
% 		'HorizontalAlignment','left',...
% 		'String','VPKurtosis',...
% 		'FontSize',10);

%----------------------------------------------------------------------------------- 1.0
case 1 % Read and display an image

	[filename,pathname] = uigetfile({'*.png;*.tif;*.jpg','Image files'});

	if filename ~= 0

		% Clear the old data
		clear global I stats count
		global I stats count
		set(H.edit(1),'String','');
		
		% Read the image and convert to intensity
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
	
	count = 1;	% sub-image (character) to display
	% Threshhold the image and smooth the edges
	thresh = 125;
	I.bw = (I.Image < thresh);
	%I.bw = bwmorph(I.bw,'majority',1);

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
    P = [];
	[L,num] = bwlabel(I.bw);
    I.label = L;
	I.NumChar = num;
	fprintf('%d objects have been detected\n\n',I.NumChar);

	colormap(gray(2))
	axes(H.ax(1))
	image(I.bw)
	set(H.ax(1),'XTick',[],'YTick',[])

	% Segment the labeled image into character sub-images
	stats = regionprops(I.label,'Image','Eccentricity','Orientation');
	
	% Fix the Orientation property
	ang = [stats.Orientation];
	ang = (180+ang).*(ang < 0) + ang.*(ang > 0);
	ang = num2cell(ang');
	len = length(stats);
	[stats.Orientation] = deal(ang{:});

	% Find additional features
	stats = FindFeatures(stats);

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
		clear I stats count
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

    TRAINING_PATTERNS = 50; % Number of training patterns
    TRAINING_RATE = 0.05;   % Training rate
    TRAINING_ERR_TOLERANCE = 1e-3; % Training error tolerance

    global H I stats count net P T

    num = length(stats);

    T = [];
    for n = 1:num
        T = [T, stats(n).Class - '0'];
    end
    P = P';

    disp(P);

    disp(size(T));
    disp(size(P));

    %net = newff([0 1], [num 1], {'tansig', 'tansig', 'purelin'}, 'traingd');
    hidden_nodes = 10;
    net = feedforwardnet(hidden_nodes,'traingd');
    net.numLayers = 3;
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
    net1 = train(net, P, T); % Train the network.

    a = sim(net1, P);
    plot(P, a-T, P, T);
    grid;

%----------------------------------------------------------------------------------- 9.0
case 9 % Plot 2 of the features 
	
	% Find all the features
	%stats = regionprops(I.label,'Image','Eccentricity','Orientation');
	%stats = FindFeatures(stats);
	
	num_checked = 0;
	
	for k = 1:length(H.checkbox)
		if get(H.checkbox(k),'Value')
			num_checked = num_checked + 1;
			if num_checked == 1
				feature_name1 = get(H.checkbox(k),'String');
				s = ['[stats.' feature_name1 ']'];
				f1 = eval(s);
			elseif num_checked == 2
				feature_name2 = get(H.checkbox(k),'String');
				s = ['[stats.' feature_name2 ']'];
				f2 = eval(s);
			elseif num_checked > 2, break
			end
		end
	end
	
	if num_checked >= 2
		c = [stats.Class]';
		
		figure(3)
		plot(f1,f2)
		
		YLim = get(gca,'YLim');
		XLim = get(gca,'XLim');
		cla
		text(f1,f2,c)
		set(gca,'XLim',XLim,'YLim',YLim)
		xlabel(feature_name1),ylabel(feature_name2)
		grid
	end
	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sub-functions

%----------------------------------------------------------------------------------- A.0
function disp_char(stats,H,count,width,height)
% Display a subimage (character)

% Resize the second axes accordingly
w_max = 100;
h_max = 100;
[h,w] = size(stats(count).Image);
[im_width,im_height] = fit_im(w_max,h_max,w,h);
left = 580 + (w_max - im_width)/2;
bott = 190 + (h_max - im_height)/2;

% Display the character
if gca ~= H.ax(2)
	axes(H.ax(2))
end
image(stats(count).Image);
set(H.ax(2),'position',[left/width bott/height im_width/width im_height/height])
set(H.ax(2),'XTick',[],'YTick',[]);


%----------------------------------------------------------------------------------- B.0
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


%----------------------------------------------------------------------------------- C.0
function c = GetLabel(edit_box)
% Get the class label from the edit box

c = get(edit_box,'String');
set(edit_box,'String',[])


%----------------------------------------------------------------------------------- D.0
function stats = FindFeatures(stats)
% Extract features from the sub-images

num = length(stats);

for n = 1:num

	% Compute the horizontal and vertical projections (column vectors)
	stats(n).HorizontalProjection = sum(stats(n).Image,2);
	stats(n).VerticalProjection = sum(stats(n).Image,1)';	

	% Find the central moments of the horizontal and vertical projections

	[r,c] = size(stats(n).Image);
	% Find the 0th and 1st moments of H and V projections
	stats(n).HPMoment = [ones(1,r) ; 1:r]*stats(n).HorizontalProjection;
	stats(n).VPMoment = [ones(1,c) ; 1:c]*stats(n).VerticalProjection;

	stats(n).HCenter = stats(n).HPMoment(2)/stats(n).HPMoment(1);
	stats(n).VCenter = stats(n).VPMoment(2)/stats(n).VPMoment(1);

	r_norm = [1:r]-stats(n).HCenter;
	ind_r = [r_norm.^2 ; r_norm.^3 ; r_norm.^4];
	c_norm = [1:c]-stats(n).VCenter;
	ind_c = [c_norm.^2 ; c_norm.^3 ; c_norm.^4];
	
	stats(n).HPMoment(3:5) = ind_r*stats(n).HorizontalProjection;
	stats(n).VPMoment(3:5) = ind_c*stats(n).VerticalProjection;

	stats(n).HPSkewness = stats(n).HPMoment(4)/stats(n).HPMoment(3)^(3/2);
	stats(n).VPSkewness = stats(n).VPMoment(4)/stats(n).VPMoment(3)^(3/2);

	stats(n).HPKurtosis = stats(n).HPMoment(5)/stats(n).HPMoment(3)^(2);
	stats(n).VPKurtosis = stats(n).VPMoment(5)/stats(n).VPMoment(3)^(2);

    stats(n).Image = imresize(stats(n).Image, [15, 15]);
    
    disp(size(stats(n).Image));
    
    % set up input matrix
    global P

    disp(stats(n).Image);

    row = stats(n).Image(:)';
    if size(P) == 0
        P = [row];
    else
        P = [P;row];
    end

end

    



