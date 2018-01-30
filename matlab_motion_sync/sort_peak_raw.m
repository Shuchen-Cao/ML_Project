% this function is used to get the kicking ground peaks

function [pks, locs] = sort_peak_raw(pks_raw, locs_raw)

pks(1) = pks_raw(1);
locs(1) = locs_raw(1);

len = size(pks_raw, 1);
for i = 2:len
    if locs_raw(i) - locs_raw(1) > 50
        pks(2) = pks_raw(i);
        locs(2) = locs_raw(i);
        break;
    end
end
        
pks(4) = pks_raw(end);
locs(4) = locs_raw(end);

for i = len - 1:-1:1
    if locs_raw(end) - locs_raw(i) < 50
        pks(4) = pks_raw(i);
        locs(4) = locs_raw(i);
    else
        break;
    end
end

if i == 1
    i = len ;
end

for j = i:-1:1
    if locs_raw(i+1) - locs_raw(j) > 50
        pks(3) = pks_raw(j);
        locs(3) = locs_raw(j);
        break;
    end
end

for k = j - 1:-1:1
    if locs_raw(j) - locs_raw(k) < 50
        pks(3) = pks_raw(k);
        locs(3) = locs_raw(k);
    else
        break;
    end
end

end

