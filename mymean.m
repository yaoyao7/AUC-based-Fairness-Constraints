function ave = mymean(x)
    temp=mean(x);
    if isnan(mean(temp))
        ave=0;
    else
        ave=temp;
    end
end