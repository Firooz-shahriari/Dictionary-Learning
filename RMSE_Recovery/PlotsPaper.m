%%%%%%%%%%%%     Recover of Kronecker Dictionary and RMSE     %%%%%%%%%%%%%

clc
clear
close all
addpath 'line_fewer_markers'
%==========================================================================
%  n = 10; 
%==========================================================================
% Select your choice of plot Number and Number of training
% select Plot number:
%    1 -->  s=5 
%    2 -->  s=7
%    3 -->  s=10
%    4 -->  s=13
%    5 -->  s=15
%==========================================================================
Pnum = 5;
T    = 10000;      % number of trainging signals
% % T    = 5000;

if     T == 5000
    load OutputMain10T5000RDandRMSE.mat
    Xlim    = [1 100; 1 150; 1 200; 1 300; 1 400];
    RDlim   = [73 85; 79 85; 80 90; 81 88; 65 95 ];
    RMSElim = [0.035 0.045; 0.037 0.049;  0.04 0.050; 0.045 0.056; 0.053 0.055  ];
    Iter    = [150 200; 200 250; 250 300; 300 350; 350 400];
elseif T == 10000
    load OutputMain10T10000RDandRMSE.mat
    Xlim    = [1 100; 1 150; 1 200;1 300; 1 400];
    RDlim   = [73 87; 79 90; 80 93; 80 95; 85 90 ];
    RMSElim = [0.035 0.047 ; 0.035 0.047; 0.035 0.053; 0.043 0.054; 0.054 0.058];
    Iter    = [150 200; 200 250; 250 300; 300 350; 350 400];
end


avrat1DMOD   = avRD{1,Pnum};
avrat1DKSVD  = avRD{2,Pnum};
avrat2DMOD   = avRD{3,Pnum};
avrat2DCMOD1 = avRD{4,Pnum};
avrat2DCMOD2 = avRD{5,Pnum};

avRMSE1DMOD    = avRMSE{1,Pnum};
avRMSE1DKSVD   = avRMSE{2,Pnum};
avRMSE2DMOD    = avRMSE{3,Pnum};
avRMSE2DCMOD1  = avRMSE{4,Pnum};
avRMSE2DCMOD2  = avRMSE{5,Pnum};

figure
iter = length(avrat1DMOD);
itr  = 1:iter; 
line_fewer_markers(itr,avrat1DMOD(1:iter)   ,16,'-*','color',[0.4660, 0.6740, 0.1880]   ,'MFC',[0.4660, 0.6740, 0.1880],'Mks',18,'linewidth',4.5);
line_fewer_markers(itr,avrat1DKSVD(1:iter)  ,15,'-^','color',[0.3010, 0.7450, 0.9330]   ,'MFC',[0.3010, 0.7450, 0.9330],'Mks',15,'linewidth',4.5);
line_fewer_markers(itr,avrat2DMOD(1:iter)   ,18,'-o','color',[0.6350, 0.0780, 0.1840]   ,'MFC',[0.6350, 0.0780, 0.1840],'Mks',15,'linewidth',4.5);
line_fewer_markers(itr,avrat2DCMOD1(1:iter) ,14,'-s','color',[0.75, 0, 0.75]            ,'MFC',[0.75, 0, 0.75]         ,'Mks',15,'linewidth',4.5);

set(gca,'FontSize',36, 'linewidth',4.2);
leg = legend('1D-MOD','1D-KSVD','2D-MOD','2D-CMOD','Location','southeast');
set(leg,'FontName','Times New Roman','Interpreter','latex','FontSize',42);
xlabel('Iteration','FontSize',44,'Interpreter','latex');
ylabel('Recovery Percentage','FontSize',48, 'Interpreter','latex');
grid on;
xlim (Xlim(Pnum,:))


%%% MAGNIFIER
axes ('position', [0.493 0.32 0.15 0.22 ])
box on

tt = title('Magnified' , 'color',[0.15, 0.2470, 0.4410]);
set(tt,'FontName','Times New Roman','Interpreter','latex','FontSize',42);

magnify_ind = Iter(Pnum,1) <itr & itr<Iter(Pnum,2);
line_fewer_markers(itr(magnify_ind),avrat1DMOD(magnify_ind)   ,4,'-*','color',[0.4660, 0.6740, 0.1880]    ,'MFC',[0.4660, 0.6740, 0.1880],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avrat1DKSVD(magnify_ind)  ,3,'-^','color',[0.3010, 0.7450, 0.9330]    ,'MFC',[0.3010, 0.7450, 0.9330],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avrat2DMOD(magnify_ind)   ,5,'-o','color',[0.6350, 0.0780, 0.1840]   ,'MFC',[0.6350, 0.0780, 0.1840],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avrat2DCMOD1(magnify_ind) ,3,'-s','color',[0.75, 0, 0.75]            ,'MFC',[0.75, 0, 0.75]         ,'Mks',7,'linewidth',4);
grid on;
set(gca,'FontSize',30, 'linewidth',4);
ylim (RDlim(Pnum,:))






%==========================================================================
%==========================================================================

figure
iter = length(avrat1DMOD);
itr  = 1:iter; 
line_fewer_markers(itr,avRMSE1DMOD(1:iter)   ,16,'-*'  ,'color',[0.4660, 0.6740, 0.1880]   ,'MFC',[0.4660, 0.6740, 0.1880],'Mks',18,'linewidth',4.5);
line_fewer_markers(itr,avRMSE1DKSVD(1:iter)  ,15,'-^'  ,'color',[0.3010, 0.7450, 0.9330]   ,'MFC',[0.3010, 0.7450, 0.9330],'Mks',15,'linewidth',4.5);
line_fewer_markers(itr,avRMSE2DMOD(1:iter)   ,18,'-o'  ,'color',[0.6350, 0.0780, 0.1840]   ,'MFC',[0.6350, 0.0780, 0.1840],'Mks',15,'linewidth',4.5);
line_fewer_markers(itr,avRMSE2DCMOD1(1:iter) ,14,'-s'  ,'color',[0.75, 0, 0.75]            ,'MFC',[0.75, 0, 0.75]         ,'Mks',15,'linewidth',4.5);

set(gca,'FontSize',36, 'linewidth',4.2);
leg = legend('1D-MOD','1D-KSVD','2D-MOD','2D-CMOD','Location','northeast');
set(leg,'FontName','Times New Roman','Interpreter','latex','FontSize',42);
xlabel('Iteration','FontSize',44,'Interpreter','latex');
ylabel('RMSE','FontSize',48, 'Interpreter','latex');
grid on;
xlim (Xlim(Pnum,:))

%%% MAGNIFIER
axes ('position', [0.493 0.6 0.15 0.22 ])
box on

tt = title('Magnified' , 'color',[0.15, 0.2470, 0.4410]);
set(tt,'FontName','Times New Roman','Interpreter','latex','FontSize',42);

magnify_ind = Iter(Pnum,1) <itr & itr<Iter(Pnum,2);
line_fewer_markers(itr(magnify_ind),avRMSE1DMOD(magnify_ind)   ,4,'-*','color',[0.4660, 0.6740, 0.1880]    ,'MFC',[0.4660, 0.6740, 0.1880],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avRMSE1DKSVD(magnify_ind)  ,3,'-^','color',[0.3010, 0.7450, 0.9330]    ,'MFC',[0.3010, 0.7450, 0.9330],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avRMSE2DMOD(magnify_ind)   ,5,'-o','color',[0.6350, 0.0780, 0.1840]    ,'MFC',[0.6350, 0.0780, 0.1840],'Mks',7,'linewidth',4);
line_fewer_markers(itr(magnify_ind),avRMSE2DCMOD1(magnify_ind) ,3,'-s','color',[0.75, 0, 0.75]             ,'MFC',[0.75, 0, 0.75]         ,'Mks',7,'linewidth',4);
grid on;
set(gca,'FontSize',28, 'linewidth',4);
ylim (RMSElim(Pnum,:))







