function name = get_gait_names(i_gait)
switch i_gait
    case 1
        name = 'normal';
    case 2
        name = 'toein';
    case 3
        name = 'toeout';
    case 4
        name = 'largeSW';
    case 5
        name = 'largeTS';
    case 6
        name = 'toein_largeSW';
    case 7
        name = 'toeout_largeSW';
end