function export_matlab_reference(repo_root, output_folder)
% Synthetic reference cases only: safe to publish with the Python tests.
root=char(repo_root); run(fullfile(root,'startup.m'));
if nargin<2, output_folder=fullfile(root,'python','tests','fixtures'); end
maxNumCompThreads(2); set(groot,'defaultFigureVisible','off');

duration=3; sr=10000; n=5;
times=cell(1,n); waveforms=zeros(60,n); units(1,n)=UnitData();
for k=1:n
 times{k}=cumsum(.025+.015*(1+sin((1:50)'*.7+k)));
 x=(0:59)'/10;
 waveforms(:,k)=-exp(-((x-(1.6+.05*k))/(.12+.025*k)).^2)+.32*exp(-((x-(2+.09*k))/(.22+.03*k)).^2);
 units(k)=UnitData(k,k,times{k},waveforms(:,k),"synthetic_recording",sr,duration);
 units(k).UnitID="synthetic_unit_"+k;
end
fs=FeatureStore();
fs.UnitTable=table(string({units.UnitID})',repmat("synthetic_recording",n,1),'VariableNames',{'UnitID','RecordingID'});
fs.MetadataTable=table("synthetic_recording",'VariableNames',{'RecordingID'});
params=struct('CultureKeys',"RecordingID");
ctc=CellTypeClassifier(fs,units,params);
acgs=zeros(401,n); activity=cell(1,n);
for k=1:n
 ensureCCGHeart();
 acgs(:,k)=CCG(times{k},ones(size(times{k})), 'binSize',.0005,'duration',.2,'mode','ccg');
 activity{k}=table2struct(computeActivityFeatures(times{k},duration));
end
[X,names,aligned,norm_acgs]=buildFeatureMatrix(ctc,waveforms,acgs,sr);
shape=computeWaveformFeatures(waveforms,sr);
graph_X=[sin((1:9)'*.7),cos((1:9)'*.9),((1:9)'/9).^2];
G=CellTypeClassifier.knnGraph(graph_X,3);
seed_idx=[1,4,7]; seed_y=[1,2,1];
[F,unreached,converged,iterations]=CellTypeClassifier.propagateLabels(G,seed_idx,seed_y,1000,1e-9);
ctc.UnitLabels=[1,1,2,2,0];
eia=EIAnalyzer(ctc,struct('Activity',struct('SecCutout',[0 duration])));
eia.computeActivity(); eia.detectBursts(); eia.computeCorrelations();
a=eia.Activity(1); corrs=eia.Correlations(1).pop_corr;
ei=struct('counts',a.binned_mat,'labels',ctc.UnitLabels,'bin_size',eia.Parameters.Activity.BinSize, ...
 'smooth_window',eia.Parameters.Activity.SmoothWindow,'E',a.E,'I',a.I,'total',a.total,'ratio',a.ratio, ...
 'E_raw',a.E_raw,'I_raw',a.I_raw,'burst_state',eia.BurstState{1},'population_correlations',[corrs{1}(:),corrs{2}(:)]);
eia.Parameters.Activity.RatioMethod="perbin"; eia.computeActivity(); ei.perbin_ratio=eia.Activity(1).ratio;
sttc=zeros(n,n);
for i=1:n
 for j=i+1:n
  sttc(i,j)=sttc(numel(times{i}),numel(times{j}),.005,[0 duration],times{i}',times{j}'); sttc(j,i)=sttc(i,j);
 end
end
gold=struct('synthetic',true,'matlab_version',version,'upstream_commit','5efdcb82fae3700a179bde10105d9d5dc7743d52', ...
 'duration',duration,'sampling_rate',sr,'times',{times},'waveforms',waveforms,'acgs',acgs, ...
 'activity',{activity},'shape',{shape},'harmonized_features',X,'feature_names',names,'aligned_waveforms',aligned,'normalized_acgs',norm_acgs, ...
 'graph_X',graph_X,'graph',full(G),'seed_indices_zero_based',seed_idx-1,'seed_labels',seed_y, ...
 'propagation',struct('scores',F,'unreached',unreached,'converged',converged,'iterations',iterations), ...
 'ei',ei,'sttc',sttc);
gold.smoothed_acgs=CellTypeClassifier.smoothACGs(acgs,3.5,.0005);
gold.normalization_input=[sin((1:10)'*(1:4)*.37),ones(10,1),NaN(10,1)];
[z,mu,sd]=normalize(gold.normalization_input);
keep=~any(isnan(z),1); z=z(:,keep); scale=max(abs(z),[],1); scale(scale==0)=1;
gold.normalization_expected=z./scale; gold.normalization_keep=keep;
target=char(output_folder); if ~isfolder(target), mkdir(target); end
fid=fopen(fullfile(target,'matlab_synthetic_reference.json'),'w'); c=onCleanup(@()fclose(fid));
fwrite(fid,jsonencode(gold,PrettyPrint=false),'char');
disp('Synthetic reference export complete; no research data included.');
end
