% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function opts = initDatasetOpts(data_root,baseDir,dataset,imgType,model_type,listid,iteration_num,iouthresh,costtype,gap, nms_th, score_th)

opts = struct();
opts.imgType = imgType;
opts.costtype = costtype;
opts.gap = gap;
opts.baseDir = baseDir;
opts.imgType = imgType;
opts.dataset = dataset;
opts.iouThresh = iouthresh;
opts.weight = iteration_num;
opts.listid = listid;
opts.nms_th = nms_th;
opts.score_th = score_th;

testlist = ['testlist',listid];
%%testlist = 'testlist01';
opts.vidList = sprintf('%s/%s/splitfiles/%s.txt',data_root,dataset,testlist);

if strcmp(dataset,'ucf24')
    opts.actions = {'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',...
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',...
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',...
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',...
        'VolleyballSpiking','WalkingWithDog'};
    opts.annotFile = sprintf('%s/%s/splitfiles/finalAnnots.mat',data_root,dataset);
elseif strcmp(dataset,'jhmdb')
    opts.actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
        'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
        'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};
    opts.annotFile = sprintf('%s/%s/splitfiles/finalAnnots.mat',data_root,dataset);
elseif strcmp(dataset,'LIRIS')
    opts.actions = {'discussion', 'give_object_to_person','put_take_obj_into_from_box_desk',...
        'enter_leave_room_no_unlocking','try_enter_room_unsuccessfully','unlock_enter_leave_room',...
        'leave_baggage_unattended','handshaking','typing_on_keyboard','telephone_conversation'};
end

opts.imgDir = sprintf('%s/%s/%s-images/',data_root,dataset,imgType);

% opts.detDir = sprintf('%s/%s/detections/%s-%s-%s-%06d/',baseDir,dataset,model_type,imgType,listid,iteration_num);
opts.detDir = '/home/rusu5516/postprocessing/ucf24/';

opts.actPathDir = sprintf('/home/rusu5516/postprocessing/ucf24/actionPaths/%06d-%06d-%s-%06d-%s-%d-%04d/',nms_th*100,score_th*1000,listid,iteration_num,costtype,gap,iouthresh*100);
opts.tubeDir = sprintf('/home/rusu5516/postprocessing/ucf24/actionTubes/%06d-%06d-%s-%06d-%s-%d-%04d/',nms_th*100,score_th*1000,listid,iteration_num,costtype,gap,iouthresh*100);

if exist(opts.detDir,'dir')
    if ~isdir(opts.actPathDir)
        fprintf('Creating %s\n',opts.actPathDir);
        mkdir(opts.actPathDir)
    end
    if ~isdir(opts.tubeDir)
        mkdir(opts.tubeDir)
    end
    if strcmp(dataset,'ucf24') || strcmp(dataset,'jhmdb')
        createdires({opts.actPathDir},opts.actions)
    end
end
