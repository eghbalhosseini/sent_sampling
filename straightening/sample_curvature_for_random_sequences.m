% define 8*1000 points in 300 dimensional space 
D=randn(21*8408,1600);
D_cell={};
for i=1:8408
    D_cell{i}=D((21*(i-1)+1):21*(i),:);
end 

% do the same procedure as curvature calculation 
%D_diff=cellfun(@(x) normalize(diff(x,1),2,'norm',2),D_cell,'uni',false);
D_diff=cellfun(@(x) diff(x,1),D_cell,'uni',false);
figure; 
scatter3(D_cell{1}(:,1),D_cell{1}(:,2),D_cell{1}(:,3),40,'filled');
hold on 
for k=1:7
    plot3([D_cell{1}(k,1),D_cell{1}(k,1)+D_diff{1}(k,1)]',...
       [D_cell{1}(k,2),D_cell{1}(k,2)+D_diff{1}(k,2)]',...
       [D_cell{1}(k,3),D_cell{1}(k,3)+D_diff{1}(k,3)]','color',k/10*[1,1,1],'linewidth',3)
end 
    
arrayfun(@(x) text(D_cell{1}(x,1),D_cell{1}(x,2),D_cell{1}(x,3),num2str(x),...
    'HorizontalAlignment','left','VerticalAlignment','top','fontsize',20),1:8)
print('-painters','-dpdf',gcf,'example_curve.pdf')

all_curve={}
for p=1:length(D_diff)
    x=D_diff{p};
    % randomly picka number from 4 to 20 
    ending=randi(14,1)+5;
    curve=[];
    x=x(1:ending,:);
    for q=1:size(x,1)-1
        curve=[curve;acosd(dot(x(q,:),x(q+1,:))/(norm(x(q,:)*norm(x(q+1,:)))))];
    end 
    all_curve{p}=curve;
end 
figure;
histogram(cellfun(@mean,all_curve))
box off
xlabel('average curvature (deg)')
xlim([117,123]);
a=mean(cellfun(@mean,all_curve))
hold on;
plot([a,a],get(gca,'ylim'),'linewidth',2,'color','k')

print('-painters','-dpdf',gcf,'hist_curve.pdf')

%% get perceptual estimates 
d_ij_cell=cellfun(@(x) squareform(pdist(x)),D_cell,'uniformoutput',false);
func= @(x) normcdf(x/sqrt(2)).*normcdf(x/(2))+normcdf(-x/sqrt(2)).*normcdf(-x/(2));
cellfun(@(x) func(x), d_ij_cell,'uni',false)
