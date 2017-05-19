m=importdata('left-only-data.csv');
yf=fft(m.data);
x=linspace(0,4454,4454);
colorstring='bgrcmyk'
hold on
for i=1:7
    plot(x,fy(:,i),'Color',colorstring(i))
end