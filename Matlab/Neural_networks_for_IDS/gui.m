function varargout = gui(varargin)
% GUI MATLAB code for gui.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gui

% Last Modified by GUIDE v2.5 05-Jun-2013 20:26:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gui is made visible.
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui (see VARARGIN)

% Choose default command line output for gui

% Update handles structure
% UIWAIT makes gui wait for user response (see UIRESUME)
% uiwait(handles.form);
guidata(hObject, handles);



% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
%varargout{1} = handles.output;


% --- Executes on button press in btnTrain.
function btnTrain_Callback(hObject, eventdata, handles)
% hObject    handle to btnTrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('datanumtrain','C'); %k�dolt training adatsor bet�lt�se
load ('datanumtest','Ctest'); %k�dolt teszt adatsor bet�lt�se
[nv,lv] = size(C); %nv: sorok sz�ma, lv: attrib�tumok/oszlopok sz�ma

guidata(hObject, handles); %popupmenu-t�l kapott �rt�k friss�t�se
trainpercentage = handles.T; % a trainingpercentage �rt�k�l kapja a popupmenu kiv�lasztott elem�hez tartoz� �rt�ket
tr = ceil(nv*trainpercentage); % traininghez felhaszn�lt sorok sz�ma
indices = randperm(nv);           % indexek randomiz�l�sa
traindices = indices(1:tr); %a random indexekb�l tr db kiv�laszt�sa �s t�rol�sa
% tan�t�s
P = C(traindices,1:lv-1)';      % training input vektorok t�rol�sa
T = C(traindices,lv)' > 1;      % kimenet: 0 -> norm�l,   1 -> t�mad�s
net = feedforwardnet(10, 'trainlm');  %training f�ggv�ny megad�sa
[net] = train(net, P,T); %tan�t�s
          
save('net','net'); %betan�tott network ment�se

% teszt
[nv,lv] = size(Ctest); %nv: sorok sz�ma, lv: attrib�tumok/oszlopok sz�ma a teszt adatsorban
indices2 = randperm(nv);  %indexek randomiz�l�sa
startindex = 100000;
endindex = 500000;
testindices = indices2(startindex:endindex); %tesztel�shez haszn�lt indexek megad�sa, nekem csak 200 sort b�r a g�pem:( de 1:end lenne a j� �s m�g csak az is a teljes 4 milli�s adatsor hatoda kb...
Ptest = Ctest(testindices,1:lv-1)';    % test input vectorok
Ttest = Ctest(testindices,lv)' > 1;      % kimenet: 0 -> norm�l,   1 -> t�mad�s

Q = sim(net,Ptest); %tesztel�s

% g�rb�k kirajzol�sa

thresspan = [0.025:0.025:0.975]; %hat�r �rt�kek t�rol�sa egy vektorban
lthsp = length(thresspan); %t�rolt hat�r �rt�kek sz�ma

n = length(find(Ttest==0));     % norm�l adatok sz�ma
p = length(find(Ttest==1));     % t�mad�sok sz�ma

tnr = zeros(lthsp,1); tpr = tnr; fnr=tnr; fpr=tnr; acc=tnr; %inicializ�l�s

%a megadott hat�r�rt�kek eset�ben tnr, tpr, fnr, fpr, pontoss�g meghat�roz�sa
for ts=1:lthsp,
    
    Qthres = Q>thresspan(ts); %hat�r�rt�k szolg�l a kimenet bin�risan (0: norm�l, 1: t�mad�s) t�rt�n� meghat�roz�s�ra

    tnr(ts) = length(find(Qthres==0 & Ttest==0))/n;  % true negative rate (val�di t�mad�s) 
    tpr(ts) = length(find(Qthres==1 & Ttest==1))/p;  % true positive rate (val�di norm�l)
    fnr(ts) = length(find(Qthres==0 & Ttest==1))/p;  % false negative rate (nem �szlelt t�mad�s)
    fpr(ts) = length(find(Qthres==1 & Ttest==0))/n;  % false positive rate (t�vesen �szlelt t�mad�s)  

    acc(ts) = (tnr(ts)*n+tpr(ts)*p)/(p+n);  % pontoss�g
    
if ts==20 %0.05 hat�r�rt�k eset�n az eredm�ny kiirat�sa
s=sprintf('K�sz�bszint: %0d \nTPR, \tTNR, \tFPR, \tFNR:\n%8.6f, \t%8.6f, \t%8.6f, \t%8.6f \nPontoss�g: \n%8.6f\n',thresspan(ts), tpr(ts),tnr(ts),fpr(ts),fnr(ts), acc(ts));
set(handles.txtResult,'String',s);
end 
%eredm�nyek t�rol�sa a g�rb�k kirajzol�s�hoz
[tnr tpr fnr fpr acc]
%eredm�nyek ment�se
save(sprintf('test-%.5f-results',round(trainpercentage*100)),'traindices','testindices','trainpercentage','thresspan','tnr','tpr','fnr','fpr','acc');

%els� �bra: false positive �s true positive kapcsolata
plot(handles.axes1,fpr,tpr); 
guidata(hObject, handles); 

%m�sodik �bra: hat�r�rt�k �s pontoss�g kapcsolata
plot(handles.axes2,thresspan,acc); 
guidata(hObject, handles);  
    
end


% --- Executes on selection change in chooseTrainPercent.
function chooseTrainPercent_Callback(hObject, eventdata, handles)
% hObject    handle to chooseTrainPercent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)%
%handles = guidata(hObject);

%a popupmenu kiv�lasztott elem�hez tartoz� �rt�kek megad�sa
 val = get(hObject,'Value');  
 switch val
  case 1
    T=0.001;
  case 2
    T=0.01;
  case 3
    T=0.1;
  case 4
    T=0.2;   
  case 5
    T=0.3;
  case 6
    T=0.4;
  case 7
    T=0.5;
  case 8
    T=0.6;
  case 9
    T=0.7;
  case 10
    T=0.8;
  case 11
    T=0.9;
  case 12
    T=1.0;
  otherwise
 end 
  handles.T=T;
guidata(hObject,handles); %kiv�lasztott �rt�k friss�t�se
% Hints: contents = cellstr(get(hObject,'String')) returns chooseTrainPercent contents as cell array
%        contents{get(hObject,'Value')} returns selected item from chooseTrainPercent

        
% --- Executes during object creation, after setting all properties.
function chooseTrainPercent_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chooseTrainPercent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function form_CreateFcn(hObject, eventdata, handles)
% hObject    handle to form (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

handles.T=0.001; %popupmenu default �rt�k�nek be�ll�t�sa

guidata(hObject,handles); 
