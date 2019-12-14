Times = 50;
x = magic(1000);
%x = ones([20,20]);
%x(1,3) = 2;
%x(4,17) = 2;
%x(4,9) = 10;
tic;
%for i = 1 : Times;
%    [y,we] = cnnPool([3 3],x,'meanpool');
%end
fprintf('meanpool time used:%s\n',toc);
for i = 1 : Times;
    [y,we] = cnnPool([3 3],x,'maxpool');
end
fprintf('maxpool time used:%s\n',toc);