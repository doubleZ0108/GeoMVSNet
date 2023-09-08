format compact


MaxDist=20; %outlier thresshold of 20 mm

time=clock;

% method_string='mvsnet';
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end


UsedSets=str2num(set)

nStat=length(UsedSets);

BaseStat.nStl=zeros(1,nStat);
BaseStat.nData=zeros(1,nStat);
BaseStat.MeanStl=zeros(1,nStat);
BaseStat.MeanData=zeros(1,nStat);
BaseStat.VarStl=zeros(1,nStat);
BaseStat.VarData=zeros(1,nStat);
BaseStat.MedStl=zeros(1,nStat);
BaseStat.MedData=zeros(1,nStat);

for cStat=1:length(UsedSets) %Data set number
    
    currentSet=UsedSets(cStat);

    EvalName=[resultsPath method_string eval_string num2str(currentSet) '.mat'];
    
    disp(EvalName);
    load(EvalName);
    
    Dstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
    Dstl=Dstl(Dstl<MaxDist); % discard outliers
    
    Ddata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
    Ddata=Ddata(Ddata<MaxDist); % discard outliers
    
    BaseStat.nStl(cStat)=length(Dstl);
    BaseStat.nData(cStat)=length(Ddata);
    
    BaseStat.MeanStl(cStat)=mean(Dstl);
    BaseStat.MeanData(cStat)=mean(Ddata);
    
    BaseStat.VarStl(cStat)=var(Dstl);
    BaseStat.VarData(cStat)=var(Ddata);
    
    BaseStat.MedStl(cStat)=median(Dstl);
    BaseStat.MedData(cStat)=median(Ddata);
    
    disp("acc");
    disp(mean(Ddata));
    disp("comp");
    disp(mean(Dstl));
    time=clock;
end

disp(BaseStat);
disp("mean acc")
disp(mean(BaseStat.MeanData));
disp("mean comp")
disp(mean(BaseStat.MeanStl));

totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat']
save(totalStatName,'BaseStat','time','MaxDist');


exit