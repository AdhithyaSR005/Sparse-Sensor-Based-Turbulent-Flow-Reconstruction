data_unresolved = load('Xntr_1.mat');
data_resolved = load('u_final_reduced.mat');
%timi=50;%time interval
% t1=0:0.4:75.96;%downsample20
%t1=0:(0.032/400):0.032; % untimeresolved
%t2=0:(0.032/9600):0.032; % timeresolved
%t1 = t1(2:end);
%t2 = t2(2:end);
%[mm,nn]=size(t1);
%aa=1:50:3000;
trunk=40;%truncation
ps=21*21;% number of timeresloved snapshots.
deltr=49;
nn = 240;
nt=12000;
PP=zeros(deltr*ps+ps,nt-deltr);
Ycol=zeros(deltr*ps+ps,nn);
% Pr=zeros(3799,timi+1);
%filename= 'reco';
%mask_test=true([21,21]);

%% load data
%data = load('timeresolved2D.mat');
%matrix = data.finalmatrix;
%for i=1:48000
% duu(:,:,i)=cell2mat(u_original(i,:)) ;
   % duu(:,:,i)= matrix(:,i);
% dw(:,:,i)=curl(du(:,:,i),dv(:,:,i));
%end

%% subtract cylinder and mean
% du=[duu;duv];
matrix_unresolved = data_unresolved.Xntr ;
matrix_resolved = data_resolved.matrix ;
%matrix_unresolved = matrix_unresolved(1:760,1:1024,:);

%X=du;
 %temp1=duu(:,:,1000);
%     figure(10)
%    imagesc(twmp1)
%    jkjk
X_untimeresolved=reshape(matrix_unresolved,[21*21 ,240]);
mean_untimeresolved = mean(X_untimeresolved, 1, 'omitnan');
    X_untimeresolved=X_untimeresolved-mean_untimeresolved;
[U_untimeresolved,S_untimeresolved,V_untimeresolved]=svd(X_untimeresolved,'econ');
 
 

X_timeresolved=reshape(matrix_resolved,[21*21,12000]);
mean_timeresolved = mean(X_timeresolved, 1, 'omitnan');
    X_timeresolved=X_timeresolved-mean_timeresolved;
[U_timeresolved,S_timeresolved,V_timeresolved]=svd(X_timeresolved,'econ');

%% generate non-time-resolved data
%j=1;
    %for i=1:timi:nt
       % Xntr(:,j)=X(:,i);
      % j=j+1;  
   % end
%% perform SVD
%[Uorg,Sorg,Vorg]=svd(Xntr,'econ');
%% truncate data
% uu= mean(du, 3, 'omitnan');
% [Ut,St,Vt]=svd(duu,'econ');
r=trunk;
 U_untimeresolved_1=U_untimeresolved(:,1:r);S_untimeresolved_1=S_untimeresolved(1:r,1:r);V_untimeresolved_1=V_untimeresolved(:,1:r);
 U_timeresolved_1=U_timeresolved(:,1:r);S_timeresolved_1=S_timeresolved(1:r,1:r);V_timeresolved_1=V_timeresolved(:,1:r);
%   Ut=Ut(:,1:r);St=St(1:r,1:r);Vt=Vt(:,1:r);
 X_untimeresolved_truncated =U_untimeresolved_1*S_untimeresolved_1*V_untimeresolved_1';
%   XXX=Ut*St*Vt';
X_timeresolved_truncated =U_timeresolved_1*S_timeresolved_1*V_timeresolved_1';
dod=max(max(X_untimeresolved_truncated)) ;
%filename= 'truncated';
%  Xtruee=mean(Xfull,2);
%  Xtruee=reshape(Xtruee,[53 51]);
%  figure(1)
%  imagesc(Xtruee)
%  klkl
%   F_make_fig_sensortr(Xfull, mean_X, filename, mask_test);
%   hjhjhj
%%%%
 %% sensor data
% X=reshape(X,[53 51 nt]);
% s=3;
%  Pr1(1,:)=X(20,20,:);
%  Pr2(1,:)=X(30,30,:);
%  klkl
%  Pr3(1,:)=X(40,40,:);
%  Pr=[Pr1];
%  Pr1(1,:)=X(20,1,:);
%  Pr2(1,:)=X(30,1,:);
%  Pr3(1,:)=X(40,1,:);
% I=[20;30;40];
% J=[1;1;1]
%  ssm BDG
% Pr1(1,:)=X(23,19,:);
% Pr2(1,:)=X(33,32,:);
% Pr3(1,:)=X(53,38,:);
% I=[23;33;53];
%J=[19;32;38];
%  %%ssm DG
%   Pr1(1,:)=X(34,32,:);
%  Pr2(1,:)=X(22,18,:);
%  Pr3(1,:)=X(39,23,:);
%  Pr4(1,:)=X(40,1,:);
%  Pr5(1,:)=X(50,1,:);
%  Pr=[Pr1;Pr2;Pr3;Pr4;Pr5];
%   Pr=Pr';

 %% create data matrix of sensor data with time delay
%  deltr=0;
  j =1 ;
 for i=1:50:12000
   % i2tar=find(abs(t2-t1(i))==min(abs(t2-t1(i))));
   %com_index = find(abs(t2-t1(i)) == min(abs(t2-t1(i))));
   st = i;
   ed = i + deltr;
   Ycol(:,j) = reshape(X_timeresolved_truncated(:,st:ed),[ps*(deltr+1),1]);
   j = j+1;
  end
   
  
 
%     tw=t2(10+(1:5))
%     jjj
   % i2st  =i2tar;
  %  i2ed  = i2tar+deltr;
   % ip=1;
    %Ycol(i,:) = [Pr1(ip,i2st:i2ed) Pr2(ip,i2st:i2ed) Pr3(ip,i2st:i2ed)];
%        Ycol(i,:) = [Pr1(ip,i2st:i2ed) Pr2(ip,i2st:i2ed)];
  %end
  
  for i=1:nt-deltr
   
      PP(:,i) = reshape(X_timeresolved_truncated(:,i:i+deltr),[ps*(deltr+1),1]);
  end

      %PP(i,:)=[Pr1(1,i:i+deltr) Pr2(1,i:i+deltr) Pr3(1,i:i+deltr)];
%       PP(i,:)=[Pr1(1,i) Pr2(1,i) Pr3(1,i)];
%       PP(i,:)=[Pr1(1,i:i+deltr) Pr2(1,i:i+deltr)];
 % end
 
  %% SVD of pressure data
  
 % Ycol=Ycol';
 
  [phipr,sumpr,psipr]=svd(Ycol,'econ');
 %% truncate pressure data
%  rr=trunk
%  Utpr=Upr(:,1:rr);Stpr=Spr(1:rr,1:rr);Vtpr=Vpr(:,1:rr);
%  Ptpr=Utpr*Stpr*Vtpr';%%truncate
 % Ptpr=Ycol;%without truncation
 %% u instantaneous data estimation
 % EE=Vpr'*Vorg;%% non truncated
  EEtr=psipr'*V_untimeresolved_1;%% truncated
 
 % ntt=76;
 % term=3/sqrt(ntt);
  %% filtering of epod
 % for i=1:deltr*s+s
%       for i=1:76
 % for j=1:76
 % if(-term<=EE(i,j)&&(EE(i,j)<term))
 % EE(i,j)=0;
 % end
 % end
% end
 % for i=1:1:deltr*s+s
% for i=1:76
  %for j=1:trunk
 % if(-term<=EEtr(i,j)&&(EEtr(i,j)<term))
% EEtr(i,j)=0;
 % end

 % end
 % end
  
% %   uins=Ycol'*Upr*inv(Spr)*EE*Sorg*Uorg';
 % uins=PP*Upr*inv(Spr)*EE*Sorg*Uorg';
  %uins=uins';
  %utrins=PP'*phipr*inv(sumpr)*EEtr*S_untimeresolved_1*U_untimeresolved_1';
  utrins = U_untimeresolved_1*S_untimeresolved_1'*EEtr'*inv(sumpr)'*phipr'*PP ;
  %utrins=utrins';
  %[Utrins,Strins,Vtrins]=svd(utrins,'econ');

%% truncate data
%r=trunk;
 %U=Utrins(:,1:r);S=Sins(1:r,1:r);V=Vins(:,1:r);
%Xins=Utrins*Sins*Vins';
%% plot data
  %filename= 'reco';
  %mask_test=true([53, 51]);
    
    
% k=1;
% for j=1:188
%         R=j*20+1;
%          for i=k:R-1
% xx(:,i)=utrins(:,i);
% xtrue(:,i)=Xfull(:,i);
%     k=R+1;
%          end
% end
% for i=1:50:3779
%     utrins(:,i)=Xfull(:,i);%%replace the images which are known
% end
 %F_make_fig_sensortr(utrins, mean_X, filename, mask_test,J,I);
 %%Error
  Error_DG=norm(utrins(:,11951)-X_timeresolved_truncated(:,11951), 'fro')*100/norm(X_timeresolved_truncated(:,11951), 'fro');
%  ER=rms(xtrue-xx)*100/(max(max(xtrue)));%line error
%  for i=1:3779
%      X=
%errline=rms(((Xfull(:,1:20:3779)/(max(max(0.03))))-(utrins(:,1:20:3779)/(max(max(0.03))))),"all")*100;%%point
%X=reshape(du,[2703 nt]);
Error_imse=(sqrt((immse(utrins(:,1:50:12000),( X_untimeresolved_truncated(:,1:nn)))))/dod);
%Error_imsewholemodes=(sqrt((immse(X(:,1:nt-deltr),(uins))))/dod);
%  errline=rms(((xtrue/(max(max(0.03))))-(xx/(max(max(0.03))))),"all")*100;
%  errline1=rms(((Xfull(:,1:nt-deltr))-(utrins(:,1:nt-deltr))),"all")/(max(max(Xfull)));

 err_ptrunc = rms((( X_untimeresolved_truncated(:,1:nn)- utrins(:,1:50:12000))),"all")/dod;
 %d=max(max())
 ER=rms((X_untimeresolved_truncated(1:nn)-utrins(:,1:50:12000))*100/(max(max(X_untimeresolved_truncated(:,1:nn)))));
%  Xrec=mean(utrins,2);
%  Xrec=reshape(Xrec,[53 51]);
%  Xtruee=mean(Xfull(:,1:nt-deltr),2);
%  Xtruee=reshape(Xtruee,[53 51]);
%  figure(1)
%  imagesc(Xrec)
%  figure(2)
%  imagesc(Xtruee)
%  etrrline2=rms(((Xtruee)-(Xrec)),"all")*100;
%  ER=rms(((Xfull(:,1:3799-deltr)/(max(max(Xfull))))-(utrins/(max(max(Xfull))))),"all")*100;
%    errline2=rms((utrins(:,1:3779)/((mean(Xfull,"all"))))-(Xfull(:,1:3779)/((mean(Xfull,"all")))))*100;
% a=max(max(Xtr))
% a=mean(a(10:2703),1)