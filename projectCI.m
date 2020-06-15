numusers=10;

N=200;
crossoverprob=0.6;
mutationrate=0.01;

totaluserMAE=zeros(1, numusers);
totaluserRMSE=zeros(1, numusers);

for userindex=10:numusers
maxsot=10;
bestsolutionfitness=cell([1,maxsot]);
avgfitness=0;
user_rmse = zeros(1, maxsot);
user_mae = zeros(1, maxsot);
for sot=1:maxsot
data = dlmread('ua.base');
datatest = dlmread('ua.test');
[input, desired_output] = preprocess_data(data);
[inputtest, desired_outputtest] = preprocess_data(datatest);
[init_pop,user] = calculate_init_pop (desired_output, userindex, N);
numberofmovies=size(desired_output,1);
A = [];
b = [];
Aeq = [];
beq = [];
lb = ones(1,numberofmovies);
ub = 5*ones(1,numberofmovies);
nonlcon = [];
IntCon = 1:size(desired_output,1);
options.FunctionTolerance=10^-6;
options.MaxGenerations=100*numberofmovies;
options = optimoptions('ga','MutationFcn', {@mutationadaptfeasible, mutationrate});
options.PopulationSize=N;
options.CrossoverFraction=crossoverprob;
options.InitialPopulationMatrix=init_pop;
options.OutputFcn=@gaoutfun;
options.UseParallel=true;
options.SelectionFcn=selectionroulette;
options.CrossoverFcn=crossoverscattered;
[x,value,exitflag,output] = ga(@(x) pearsonutility(user',x),numberofmovies,A,b,Aeq,beq,lb,ub,nonlcon,...
    IntCon,options);
user_rmse(sot)= calculateRMSE(x,desired_outputtest(:,userindex));
user_mae(sot)= calculateMAE(x,desired_outputtest(:,userindex));
avgfitness=avgfitness+value;
bestsolutionfitness{sot}=min(gapopulationfitness);
comparison=[x;user'];
%plot(bestsolutionfitness{sot})

end
avgRMSE=mean(user_rmse);
avgMAE=mean(user_mae);
totaluserRMSE(userindex)=avgRMSE;
totaluserMAE(userindex)=avgMAE;
plot(averagefitness(bestsolutionfitness));
xlabel('generations');
ylabel('fitness');
plotname=sprintf('Fitness N%d Pc%.2f Pm%.2f', N, crossoverprob, mutationrate);
title(plotname); 
pause(0.001);
print([plotname '.png'], '-dpng', '-r300');
%y fitness, x generations
avggenerations=mean(cellfun('length', bestsolutionfitness))
avgfitness=avgfitness/maxsot

end
totalRMSE=mean(totaluserRMSE)
totalMAE=mean(totaluserMAE)
function a=averagefitness (bestsolutionfitness)
l=min(cellfun('length', bestsolutionfitness));
a=zeros(1,l);
for i=1:length(bestsolutionfitness)
    a=a+bestsolutionfitness{i}(1:l);
end
a=a./length(bestsolutionfitness);

end

function [input, desired_output] = preprocess_data(data)
data(:,4) = [];
old_user_id = 0;
user_data = {[]};
% Loop over all rows.
for i = 1:size(data,1)
    row = data(i,:);
    % Extract the user ID from the row.
    user_id = row(1);

    if (user_id == old_user_id) || (i == 1)
        % Still on the same user, accumulate his rows in user_data.
        user_data{end} = [user_data{end} ; row];
    else
        % New user, add a new cell in user_data and start accumulating the
        % user's data.
        user_data{end + 1} = row;
    end
    old_user_id = user_id;
end
% user_data now contains one user's data in each cell.

% Loop over all user data cells to convert them to the appropriate format.
for i = 1:length(user_data)
    A = user_data{i};
    user_data{i} = [[A(1,1) ; A(1,1)] A(:,2:3)'];
end




filled_user_data=nan(length(user_data), 1682+1); % mallon oxi zeros, isws random 8a deixei
for k = 1:size(user_data,2)
%avg_rating= mean(user_data{k}(2,2:end));
%user_data{k}(2,2:end) =  user_data{k}(2,2:end) - mean(user_data{k}(2,2:end));
 %to neo diasthma timwn afou to max mean mporei na einai 5 kai to min mean mporei na einai 1 8a einai -4 ews 4 
    %edw na dhmiourgw kai na insertarw ta absent movies prepei na to xeiristw me kapoio for/if
    % sto absent movie na dinw rating to mean pou eswsa panw (giati alliws 8a
    % epairne to mean tou centered rating) mallon prepei na einai exw apo auth
    % thn for sthn for tou i
    
   
    filled_user_data(k,:) = nan(1,1682+1);
    filled_user_data(k,1)= user_data{k}(1,1);
    for i = 2:size(user_data{k},2)
    movie_id = user_data{k}(1,i);
    movie_rating = user_data{k}(2,i);
    filled_user_data(k, 1+movie_id)=movie_rating;

end

end



% scaling se [-1,1] logw sigmoid
%filled_user_data(:,2:end)=(filled_user_data(:,2:end))/4;


user_ids=filled_user_data(:,1);
input=zeros(943, length(user_ids));
for i=1:length(user_ids)
    input(user_ids(i),i)=1;
end


desired_output = filled_user_data(:,2:end)';
end

function [state,options,optchanged] = gaoutfun(options,state,flag)
persistent fitness
optchanged = false;
switch flag
    case 'init'
        fitness(:,1) = state.Score;
        assignin('base','gapopulationfitness',fitness);
    case 'iter'
        % Update the history every generation.
        ss = size(fitness,2);
        
        fitness(:,ss+1) = state.Score;
        assignin('base','gapopulationfitness',fitness);
    case 'done'
        % Include the final population in the history.
        ss = size(fitness,2);
        fitness(:,ss+1) = state.Score;
        assignin('base','gapopulationfitness',fitness);
end
end

function RMSE = calculateRMSE(outputtest, desired_outputtest)
d = (outputtest-desired_outputtest).^2;
d = d(~isnan(d));
N=mean(d,1);
RMSE=mean(sqrt(N));

end
function norm_nan = calculatenorm_nan(v)
validindices=~isnan(v);
norm_nan=norm(v(validindices));

end

function p = sample_pearson(x, y)
    % https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
    xmean = mean(x);
    ymean = mean(y);
    numerator = sum((x - xmean) .* (y - ymean));
    denominator_x = sqrt(sum((x - xmean) .^ 2));
    denominator_y = sqrt(sum((y - ymean) .^ 2));
    p = numerator / (denominator_x * denominator_y);
end

function p = sample_pearsonnan(x, y)
    validindices=~isnan(x+y); %where both are not NaN
    
    
    p= sample_pearson(x(validindices),y(validindices));

end

function f=pearsonutility(x,y) %isws anti gi auto 8eloume apolyth timh (unlikely)
   p = sample_pearsonnan(x, y);
if p>=0
        f=-p; %- epeidh h matlab kanei minimum anti gia maximum ths synarthshs
    else 
        f=0;
    end

end

function [init_pop,user] = calculate_init_pop (user_ratings, userindex, N)
user=user_ratings(:,userindex);
user_ratings(:,userindex)=[];
distances=zeros(size(user_ratings,2),1);
for i=1:size(user_ratings,2)
   distances(i)=sample_pearsonnan(user, user_ratings(:,i));
    %distances(i)=calculatenorm_nan(user-user_ratings(:,i));
end
[~, indices]=sort(distances); %asc order, dhladh min distance top
init_pop=user_ratings(:,indices(1:N))';
init_pop(isnan(init_pop))=0;
%h 0 h 3

%8a prepei na valw tyxaies times sta nan tou initpop? find out on the next
%episode

end

function MAE = calculateMAE(outputtest, desired_outputtest)
d = abs((outputtest-desired_outputtest));
d = d(~isnan(d));
N=mean(d,1);
MAE=mean(N);

end
