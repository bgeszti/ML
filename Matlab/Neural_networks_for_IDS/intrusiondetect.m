function intrusiondetect

%1: adatsor �jra bet�lt�se �s k�dol�sa, 0: kor�bbi adatsor bet�lt�se. (�gy
%oldottam meg, hogy egyes�vel be tudjam t�lteni a training �s a teszt
%adatsort is)
reload = 0;

%sz�veges jellemz�k �rt�keinek t�rol�sa vektorokban
protocol_type ={'tcp','udp', 'icmp'};
service ={'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'};
flag = { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' };
attack_type = {'normal', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'};

%binary = [7 12 21 22]; bin�ris �rt�keket tartalmaz� oszlopok sz�ma az
%adatsorban
%sz�veget tartalmaz� oszlopok t�rol�sa 
stringdata = [2 3 4 42];

%bet�lt�s �s k�dol�s
if reload,
    fn='kddcup.data.corrected';
    fid = fopen(fn);
    data = textscan(fid,'%f%s%s%s%f%f%d%f%f%f%f%d%f%f%f%f%f%f%f%f%d%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s','delimiter',',');
    fclose(fid);
    lv = length(data);
    nv = length(data{1});
    C = -1*ones(nv, lv);      % mem�ria lefoglal�sa a l�trehozand� numerikus �rt�keket tartalmaz� vektornak
    %sz�veges �rt�kek k�dol�sa sz�mokkal
    for l = 1:lv,
        switch l
            case 2
                for n=1:nv,
                    aux = data{l}(n);
                    for i=1:length(protocol_type),
                        if strcmp(aux,protocol_type{i}),
                            C(n,l) = i;
                            % Ctest(n,l) = i;  %k�l�nb�z� adatsorok
                            % elk�l�n�t�s�hez
                            break;
                        end
                    end
                end
            case 3
                for n=1:nv,
                    aux = data{l}(n);
                    for i=1:length(service),
                        if strcmp(aux,service{i}),
                            C(n,l) = i;
                            break;
                        end
                    end
                end
            case 4
                for n=1:nv,
                    aux = data{l}(n);
                    for i=1:length(flag),
                        if strcmp(aux,flag{i}),
                            C(n,l) = i;
                            break;
                        end
                    end
                end
            case 42
                for n=1:nv,
                    aux = data{l}{n};
                    for i=1:length(attack_type),
                        if strfind(aux,attack_type{i}),
                            C(n,l) = i;
                            break;
                        end
                    end
                end
            otherwise
                C(:,l) = data{l};
        end
    end
    save('datanumtrain','C'); %numerikus �rt�keket tartalmaz� m�trix ment�se
else
    load('datanumtest','Ctest'); %vagy bet�lt�se
end

%kommentezett sorok a training-n�l alkalmazand�ak

%[nv,lv] = size(C); %nv: sorok sz�ma, lv: attrib�tumok/oszlopok sz�ma a
%tan�t� adatsorban
[nv,lv] = size(Ctest); %nv: sorok sz�ma, lv: attrib�tumok/oszlopok sz�ma a
%teszt adatsorban
%trainpercentage = 0.01; %training %
%tr = ceil(nv*trainpercentage); % traininghez felhaszn�lt sorok sz�ma

indices = randperm(nv);           % indexek randomiz�l�sa
%traindices = indices(1:tr);        % a random indexekb�l tr db kiv�laszt�sa �s t�rol�sa
startindex = 300000;
endindex = 500000;
testindices = indices(startindex:endindex);  %tesztel�shez haszn�lt indexek megad�sa


% start training
%P = C(traindices,1:lv-1)';      % training input vektorok t�rol�sa
%T = C(traindices,lv)' > 1;      % kimenet: 0 -> norm�l,   1 -> t�mad�s

%1: �jra tan�t�s, 0: kor�bbi network bet�lt�se
retrain = 0;  
if retrain,
    net = feedforwardnet(10, 'trainlm'); %h�l�zat l�rehoz�sa           
    [net] = train(net, P,T); %tan�t�s
    save('net','net'); %betan�tott network ment�se
else
    load('net-trainedon10percent'); %vagy kor�bbi network bet�lt�se
end


% test start
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

    tnr(ts) = length(find(Qthres==0 & Ttest==0))/n;  % true negative rate (val�di norm�l) 
    tpr(ts) = length(find(Qthres==1 & Ttest==1))/p;  % true positive rate  (val�di t�mad�s)
    fnr(ts) = length(find(Qthres==0 & Ttest==1))/p;  % false negative rate  (nem �szlelt t�mad�s)
    fpr(ts) = length(find(Qthres==1 & Ttest==0))/n;  % false positive rate  (t�vesen �szlelt t�mad�s) 

    acc(ts) = (tnr(ts)*n+tpr(ts)*p)/(p+n);  % pontoss�g

end

%�rt�kek kiirat�sa, ment�se, g�rb�k kirajzol�sa
fprintf('TNR TPR FNR FPR Pontoss�g:\n');
[tnr tpr fnr fpr acc]
%eredm�nyek ment�se
%save(sprintf('test-%0d-results',round(trainpercentage*100)),'traindices','testindices','trainpercentage','thresspan','tnr','tpr','fnr','fpr','acc');

%els� �bra: false positive �s true positive kapcsolata
figure(1);
plot(fpr,tpr); xlabel('Hamis pozit�v ar�ny'); ylabel('Igaz pozit�v ar�ny'); 
        
%m�sodik �bra: false negative �s true negative kapcsolata       
figure(2);
plot(fnr,tnr); xlabel('Hamis negat�v ar�ny'); ylabel('Igaz negat�v ar�ny');

%harmadik �bra: hat�r�rt�k �s pontoss�g kapcsolata
figure(3);
plot(thresspan,acc); xlabel('K�sz�bszint'); ylabel('Pontoss�g'); 

