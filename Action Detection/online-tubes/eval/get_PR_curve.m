%%##################################################################################################################################################

%% Author: Gurkirt Singh 
%% Release date: 26th January 2017
% STEP-1: loop over the videos present in the predicited Tubes
% STEP-2: for each video get the GT Tubes
% STEP-3: Compute the spatio-temporal overlap bwtween GT tube and predicited
% tubes
% STEP-4: then label tp 1 or fp 0 to each predicted tube
% STEP-5: Compute PR and AP for each class using scores, tp and fp in allscore
%##################################################################################################################################################

function [mAP,mAIoU,acc,AP] = get_PR_curve(annot, xmldata, testlist, actions, iou_th)
% load(xmlfile)
num_vid = length(testlist);
num_actions = length(actions);
AP = zeros(num_actions,1);
averageIoU = zeros(num_actions,1);

cc = zeros(num_actions,1);
for a=1:num_actions
    allscore{a} = zeros(10000,7,'single');
end

total_num_gt_tubes = zeros(num_actions,1); 
% count all the gt tubes from all the vidoes for label a
% total_num_detection = zeros(num_actions,1);

preds = zeros(num_vid,1) - 1;
gts = zeros(num_vid,1);
annotNames = {annot.name};
dtNames = {xmldata.videoName};
abc = cell(num_vid,1);
for vid=1:num_vid
    abc{vid} = struct();
    abc{vid}.flag = 0;
    maxscore = -10000;
    [action,~] = getActionName(testlist{vid}); %%get action name to which this video belongs to
    [~,action_id] =  find(strcmp(action, actions)); %% process only the videos from current  action a
    [~,gtVidInd] = find(strcmp(annotNames,testlist{vid}));
    [~,dtVidInd] = find(strcmp(dtNames,testlist{vid}));
    abc{vid}.name = testlist{vid};
    dt_tubes = sort_detection(xmldata(dtVidInd));
    gt_tubes = annot(gtVidInd).tubes;
%     pick = dt_tubes.class == gt_tubes(1).class;
%     dt_tubes.class = dt_tubes.class(pick);
%     dt_tubes.boxes = dt_tubes.boxes(pick);
%     dt_tubes.score = dt_tubes.score(pick);
%     dt_tubes.framenr = dt_tubes.framenr(pick);
    
    
    num_detection = length(dt_tubes.class);
    num_gt_tubes = length(gt_tubes);
    abc{vid}.gt = cell(num_gt_tubes,1);
    %     total_num_detection = total_num_detection + num_detection;
    for gtind = 1:num_gt_tubes
        
        abc{vid}.gt{gtind}=[gt_tubes(gtind).sf,gt_tubes(gtind).ef,0,0,0];
        action_id = gt_tubes(gtind).class;
        total_num_gt_tubes(action_id) = total_num_gt_tubes(action_id) + 1;
    end
    gts(vid) = action_id;
    dt_labels = dt_tubes.class;
    covered_gt_tubes = zeros(num_gt_tubes,1);
    for dtind = 1:num_detection
        dt_fnr = dt_tubes.framenr(dtind).fnr;
        dt_bb = dt_tubes.boxes(dtind).bxs;
        dt_label = dt_labels(dtind);
        if dt_tubes.score(dtind)>maxscore
            preds(vid) = dt_label;
            maxscore = dt_tubes.score(dtind);
        end
        cc(dt_label) = cc(dt_label) + 1;
        
        ioumax=-inf;maxgtind=0;
        for gtind = 1:num_gt_tubes
            action_id = gt_tubes(gtind).class;
            if ~covered_gt_tubes(gtind) && dt_label == action_id
%             if ~covered_gt_tubes(gtind)
                gt_fnr = gt_tubes(gtind).sf:gt_tubes(gtind).ef;
%                 if isempty(gt_fnr)
%                     continue
%                 end
                gt_bb = gt_tubes(gtind).boxes;
                [iou, S_iou, T_iou] = compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb);
                if iou>ioumax
                    ioumax=iou;
                    maxgtind=gtind;
                end
            end
        end
        
        if ioumax>iou_th
            covered_gt_tubes(maxgtind) = 1;
            allscore{dt_label}(cc(dt_label),:) = [dt_tubes.score(dtind),1,3,action_id,ioumax,S_iou,T_iou];
            averageIoU(dt_label) = averageIoU(dt_label) + ioumax;
        else
            if iou_th == 0.5
                ioumax_fp = -inf;
                for g = 1:num_gt_tubes
                    gt_fnr = gt_tubes(gtind).sf:gt_tubes(gtind).ef;
                    gt_bb = gt_tubes(gtind).boxes;
                    [iou_fp, S_iou_fp, T_iou_fp] = compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb);
                    if iou_fp > ioumax_fp
                        ioumax_fp = iou_fp;
                        maxgtind_fp=gtind;
                    end
                end
                if ioumax_fp>=iou_th
                    error_type=0;
                elseif ioumax_fp<0.2
                    error_type=1;
                else
                    error_type=2;
                end
                
                if S_iou_fp > 0.7
                    abc{vid}.flag = 1;
                    tdb = dt_fnr(1);        
                    tde = dt_fnr(end);
                    asdf = [tdb,tde,covered_gt_tubes(maxgtind_fp),S_iou_fp, dt_label == action_id];
                    abc{vid}.gt{maxgtind_fp} = [abc{vid}.gt{maxgtind_fp};asdf];
                end

                allscore{dt_label}(cc(dt_label),:) = [dt_tubes.score(dtind),0,error_type,action_id,ioumax_fp,S_iou_fp,T_iou_fp];
            else
                allscore{dt_label}(cc(dt_label),:) = [dt_tubes.score(dtind),0,0,0,0,0,0];
            end
        end
        
    end
end

if iou_th == 0.5
    save('all_scores_tubes.mat','allscore')
    save('abc.mat','abc')
end

for a=1:num_actions
    allscore{a} = allscore{a}(1:cc(a),:);
    scores = allscore{a}(:,1);
    labels = allscore{a}(:,2);
    [~, si] = sort(scores,'descend');
    %     scores = scores(si);
    labels = labels(si);
    fp=cumsum(labels==0);
    tp=cumsum(labels==1);
    cdet =0;
    if ~isempty(tp)>0
        cdet = tp(end);
        averageIoU(a) = (averageIoU(a)+0.000001)/(tp(end)+0.00001);
    end
    
    recall=tp/total_num_gt_tubes(a);
    precision=tp./(fp+tp);
    AP(a) = xVOCap(recall,precision);
    draw = 0;
    if draw
        % plot precision/recall
        plot(recall,precision,'-');
        grid;
        xlabel 'recall'
        ylabel 'precision'
        title(sprintf('class: %s, AP = %.3f',actions{a},AP(a)));
    end
    %     fprintf('Action %02d AP = %0.5f and AIOU %0.5f GT %03d total det %02d correct det %02d %s\n', a, AP(a),averageIoU(a),total_num_gt_tubes(a),length(tp),cdet,actions{a});
    
end
acc = mean(preds==gts);
AP(isnan(AP)) = 0;
mAP  = mean(AP);
averageIoU(isnan(averageIoU)) = 0;
mAIoU = mean(averageIoU);


%% ------------------------------------------------------------------------------------------------------------------------------------------------
function [action,vidID] = getActionName(str)
%------------------------------------------------------------------------------------------------------------------------------------------------
indx = strsplit(str, '/');
action = indx{1};
vidID = indx{2};
%%
function sorted_tubes = sort_detection(dt_tubes)

sorted_tubes = dt_tubes;

if ~isempty(dt_tubes.class)
    
    num_detection = length(dt_tubes.class);
    scores = dt_tubes.score;
    [~,indexs] = sort(scores,'descend');
    for dt = 1 : num_detection
        dtind = indexs(dt);
        sorted_tubes.framenr(dt).fnr = dt_tubes.framenr(dtind).fnr;
        sorted_tubes.boxes(dt).bxs = dt_tubes.boxes(dtind).bxs;
        sorted_tubes.class(dt) = dt_tubes.class(dtind);
        sorted_tubes.score(dt) = dt_tubes.score(dtind);
        sorted_tubes.nr(dt) = dt;
    end
end
%% 
