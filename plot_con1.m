avg_level = mean(level);
avg_proxy = mean(proxy);
avg_corr = mean(corr);
avg_post = mean(post);
avg_level_fair = mean(level_fair);
avg_proxy_fair = mean(proxy_fair);
avg_corr_fair = mean(corr_fair);
avg_post_fair = mean(post_fair);

std_level = std(level);
std_proxy = std(proxy);
std_corr = std(corr);
std_post = std(post);
std_level_fair = std(level_fair);
std_proxy_fair = std(proxy_fair);
std_corr_fair = std(corr_fair);
std_post_fair = std(post_fair);

%% plot error bar
errorbar(avg_level_fair,avg_level,std_level,std_level,std_level_fair,std_level_fair,'LineWidth',3)
hold on
errorbar(avg_proxy_fair,avg_proxy,std_proxy,std_proxy,std_proxy_fair,std_proxy_fair,'--','LineWidth',3)
hold on
errorbar(avg_corr_fair,avg_corr,std_corr,std_corr,std_corr_fair,std_corr_fair,'-.','LineWidth',3)
hold on
errorbar(avg_post_fair,avg_post,std_post,std_post,std_post_fair,std_post_fair,':','LineWidth',3)
hold off

title('a9a','FontSize',50)
xlabel('Group Discrimination AUC Fairness','fontweight','bold','FontSize',20)
ylabel('AUC for Classification Performance','fontweight','bold','FontSize',20)
legend({'Level-set','Proxy Lagrangian','Correlation','Post-Processing'},'FontSize',20)
set(gca,'FontSize',20)
grid on
