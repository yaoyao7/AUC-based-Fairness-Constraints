avg_level = mean(level);
avg_proxy = mean(proxy);
avg_level_fair = mean(level_fair);
avg_proxy_fair = mean(proxy_fair);

std_level = std(level);
std_proxy = std(proxy);
std_level_fair = std(level_fair);
std_proxy_fair = std(proxy_fair);

%% plot error bar
errorbar(avg_level_fair,avg_level,std_level,std_level,std_level_fair,std_level_fair,'LineWidth',3)
hold on
errorbar(avg_proxy_fair,avg_proxy,std_proxy,std_proxy,std_proxy_fair,std_proxy_fair,'--','LineWidth',3)
hold off

title('a9a','FontSize',50)
% xlim([0 100])
% ylim([0.87 0.91])
xlabel('Group Fairness','fontweight','bold','FontSize',20)
ylabel('AUC for Classification Performance','fontweight','bold','FontSize',20)
legend({'Level-set','Proxy Lagrangian'},'FontSize',20)
set(gca,'FontSize',20)
% set(gca,'yscale','log')
grid on