function s = mips(f,x, m,i)
    x = double(x);
    m = double(m);
    f = squeeze(double(f));
    f = bwareaopen(f,1500,6);
    f = double(f);
    x = x.*f;
    x = max(x,[],3);
%     x = x(:,:,i);
%     x = x(:,:,5);
%     for i = 1:size(x,3)

    %magimage1 ,transmat1 = calcBG(m,x);
    magimage1 = im2single(m);
    magimage1=magimage1/(max(magimage1(:))*7/10); % pixels > 1 will be white, <0 will be black. adjust the 0.7 (or take out) to scale it
    magimage1=repmat(magimage1,[1 1 3]);
    transmat1=x; transmat1(isnan(transmat1)) = 0; transmat1(transmat1~=0)=1;
    fig = figure;
    imagesc(magimage1);
    axis equal
    hold on
    j1 = imagesc(x);
    alpha(j1,transmat1)
    colormap jet
    axis off
    colorbar
    caxis([0 1])
    filename = sprintf('images/tests_%d.png', i);
    saveas(fig,filename)
%     end
    s = fig;

end