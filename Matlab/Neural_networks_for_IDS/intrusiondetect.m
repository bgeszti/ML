function intrusiondetect

%1: adatsor újra betöltése és kódolása, 0: korábbi adatsor betöltése. (így
%oldottam meg, hogy egyesével be tudjam tölteni a training és a teszt
%adatsort is)
reload = 0;

%szöveges jellemzõk értékeinek tárolása vektorokban
protocol_type ={'tcp','udp', 'icmp'};
service ={'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'};
flag = { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' };
attack_type = {'normal', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'};

%binary = [7 12 21 22]; bináris értékeket tartalmazó oszlopok száma az
%adatsorban
%szöveget tartalmazó oszlopok tárolása 
stringdata = [2 3 4 42];

%betöltés és kódolás
if reload,
    fn='kddcup.data.corrected';
    fid = fopen(fn);
    data = textscan(fid,'%f%s%s%s%f%f%d%f%f%f%f%d%f%f%f%f%f%f%f%f%d%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s','delimiter',',');
    fclose(fid);
    lv = length(data);
    nv = length(data{1});
    C = -1*ones(nv, lv);      % memória lefoglalása a létrehozandó numerikus értékeket tartalmazó vektornak
    %szöveges értékek kódolása számokkal
    for l = 1:lv,
        switch l
            case 2
                for n=1:nv,
                    aux = data{l}(n);
                    for i=1:length(protocol_type),
                        if strcmp(aux,protocol_type{i}),
                            C(n,l) = i;
                            % Ctest(n,l) = i;  %különbözõ adatsorok
                            % elkülönítéséhez
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
    save('datanumtrain','C'); %numerikus értékeket tartalmazó mátrix mentése
else
    load('datanumtest','Ctest'); %vagy betöltése
end

%kommentezett sorok a training-nél alkalmazandóak

%[nv,lv] = size(C); %nv: sorok száma, lv: attribútumok/oszlopok száma a
%tanító adatsorban
[nv,lv] = size(Ctest); %nv: sorok száma, lv: attribútumok/oszlopok száma a
%teszt adatsorban
%trainpercentage = 0.01; %training %
%tr = ceil(nv*trainpercentage); % traininghez felhasznált sorok száma

indices = randperm(nv);           % indexek randomizálása
%traindices = indices(1:tr);        % a random indexekbõl tr db kiválasztása és tárolása
startindex = 300000;
endindex = 500000;
testindices = indices(startindex:endindex);  %teszteléshez használt indexek megadása


% start training
%P = C(traindices,1:lv-1)';      % training input vektorok tárolása
%T = C(traindices,lv)' > 1;      % kimenet: 0 -> normál,   1 -> támadás

%1: újra tanítás, 0: korábbi network betöltése
retrain = 0;  
if retrain,
    net = feedforwardnet(10, 'trainlm'); %hálózat lérehozása           
    [net] = train(net, P,T); %tanítás
    save('net','net'); %betanított network mentése
else
    load('net-trainedon10percent'); %vagy korábbi network betöltése
end


% test start
Ptest = Ctest(testindices,1:lv-1)';    % test input vectorok
Ttest = Ctest(testindices,lv)' > 1;      % kimenet: 0 -> normál,   1 -> támadás

Q = sim(net,Ptest); %tesztelés

% görbék kirajzolása

thresspan = [0.025:0.025:0.975]; %határ értékek tárolása egy vektorban
lthsp = length(thresspan); %tárolt határ értékek száma

n = length(find(Ttest==0));     % normál adatok száma
p = length(find(Ttest==1));     % támadások száma

tnr = zeros(lthsp,1); tpr = tnr; fnr=tnr; fpr=tnr; acc=tnr; %inicializálás

%a megadott határértékek esetében tnr, tpr, fnr, fpr, pontosság meghatározása
for ts=1:lthsp,
    
    Qthres = Q>thresspan(ts); %határérték szolgál a kimenet binárisan (0: normál, 1: támadás) történõ meghatározására   

    tnr(ts) = length(find(Qthres==0 & Ttest==0))/n;  % true negative rate (valódi normál) 
    tpr(ts) = length(find(Qthres==1 & Ttest==1))/p;  % true positive rate  (valódi támadás)
    fnr(ts) = length(find(Qthres==0 & Ttest==1))/p;  % false negative rate  (nem észlelt támadás)
    fpr(ts) = length(find(Qthres==1 & Ttest==0))/n;  % false positive rate  (tévesen észlelt támadás) 

    acc(ts) = (tnr(ts)*n+tpr(ts)*p)/(p+n);  % pontosság

end

%értékek kiiratása, mentése, görbék kirajzolása
fprintf('TNR TPR FNR FPR Pontosság:\n');
[tnr tpr fnr fpr acc]
%eredmények mentése
%save(sprintf('test-%0d-results',round(trainpercentage*100)),'traindices','testindices','trainpercentage','thresspan','tnr','tpr','fnr','fpr','acc');

%elsõ ábra: false positive és true positive kapcsolata
figure(1);
plot(fpr,tpr); xlabel('Hamis pozitív arány'); ylabel('Igaz pozitív arány'); 
        
%második ábra: false negative és true negative kapcsolata       
figure(2);
plot(fnr,tnr); xlabel('Hamis negatív arány'); ylabel('Igaz negatív arány');

%harmadik ábra: határérték és pontosság kapcsolata
figure(3);
plot(thresspan,acc); xlabel('Küszöbszint'); ylabel('Pontosság'); 

