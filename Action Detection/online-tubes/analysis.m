load('all_scores_tubes.mat')
for a=1:24
    score = allscore{a}(:,1);
    label = allscore{a}(:,2);
    [~,si] = sort(score,'descend');
    score = score(si);
    li = label(si) == 1;
    score = score(li);
    min_score = min(score);
    allscore{a} = allscore{a}(si,:);
%     abc = allscore{a}(li,:);
    pick = allscore{a}(:,1)>=min_score;
    allscore{a} = allscore{a}(pick,:);
end

classes={'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
        'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',...
        'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',...
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};
% cc = zeros(24,10);
% cc = zeros(24,9,100);
for a=1:24
    cc = zeros(100,9);
%     no_gt_other = zeros(24,100);
%     false_class_ra = zeros(24,100);
%     false_class_other = zeros(24,100);
%     bg_ra = zeros(24,100);
%     bg_other = zeros(24,100);
%     low_iou_ra = zeros(24,100);
%     low_iou_other = zeros(24,100);
%     tp = zeros(24,100);
    tp_pick = allscore{a}(:,2) == 1;
    tp_allscore = allscore{a}(tp_pick,1);
    total_num = length(tp_allscore);
    nums = zeros(100,9);
    pick_ra=allscore{a}(:,4)==a;
    pick_other=allscore{a}(:,4)~=a;
    cc = zeros(total_num,9);
    for i=1:total_num
%         th = round(total_num*i/100);
%         nums(i,:)=th;
        th_scores = allscore{a}(tp_pick,1);
        th_score = th_scores(i);
        th_pick = allscore{a}(:,1) >= th_score;
        th = length(allscore{a}(th_pick,1));
        all_score_ra = allscore{a}(pick_ra,:);
        all_score_other = allscore{a}(pick_other,:);
        pick = all_score_ra(:,1)>=th_score;
        all_score_ra = all_score_ra(pick,:);
        pick = all_score_other(:,1)>=th_score;
        all_score_other = all_score_other(pick,:);
        for n=0:4
            pick=all_score_ra(:,3)==n;
            cc(i, n+1) = length(all_score_ra(pick,3))/th;
            if n<4
                pick=all_score_other(:,3)==n;
                cc(i,n+6) = length(all_score_other(pick,3))/th;
            end
        end
    end
    abcd = cc;
    h = figure;
    plot_data = abcd(:,1:5);
    plot(plot_data,'LineWidth',3);
    hold on;
    plot_data = abcd(:,6:9);
    plot(plot_data,'--');
    title(classes(a));
    legend('no-gt-ra','false-class-ra','bg-ra','low-iou-ra','tp','no-gt-other','false-class-other','bg-other','low-iou-other')
    hold off;
    filename=char(strcat('figs/',classes(a), '-1'));
    saveas(h, filename,'png')
end


% for a=1:24
%     for b=1:24
%         pick=allscore{a}(:,4)==b;
%         temp=allscore{a}(pick,3);
%         for c=1:5
%             pick=temp==(c-1);
%             num=length(temp(pick));
%             if b==a
%                 cc(b,2*c-1) = cc(b,2*c-1) + num;
%             else
%                 cc(b,2*c) = cc(b,2*c)+num;
%             end
%         end
%     end
% end
% bar(cc(:,1:9),'group')
% legend('non-gt-ra','non-gt-other','high-ra','high-other','bg-ra','bg-other','low-ra','low-other','tp')
% set(gca,'xtick',1:24)
