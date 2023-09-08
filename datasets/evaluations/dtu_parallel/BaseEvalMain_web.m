format compact

representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end


dst=0.2;    %Min dist between points when reducing

% start this evaluation
cSet = str2num(thisset)

%input data name
DataInName = [plyPath sprintf('%s%03d.ply', lower(method_string), cSet)]



%results name
EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']

%check if file is already computed
if(~exist(EvalName,'file'))
    disp(DataInName);
    
    time=clock;time(4:5), drawnow
    
    tic
    Mesh = plyread(DataInName);
    Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
    toc
    
    BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);
    
    disp('Saving results'), drawnow
    toc
    save(EvalName,'BaseEval');
    toc
    
    % write obj-file of evaluation
    % BaseEval2Obj_web(BaseEval,method_string, resultsPath)
    % toc
    time=clock;time(4:5), drawnow

    BaseEval.MaxDist=20; %outlier threshold of 20 mm
    
    BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
    BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers

    BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
    BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers
    
    fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
    fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
end

fprintf("=== %d done! ===\n", cSet)

exit