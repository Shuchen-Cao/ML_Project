function [KAM,ARM] = get_KAM(knee_l, knee_r, CoP, force, subject, leg_side)
%intput:
%Knee_Left(number[1X3]),Knee_Right(number[1X3]),COP(number[1X3]),Force(number[1X3]),subjuct(struct),whichLeg(string)
%output:
%KAM(number[1X1]),ARM(number[1X3])
%function:
%calculate KAM of each frame
weight = subject.weight;
height = subject.height;
%% Find a vector "B" vertical to "Knee_vector" on the X-Y plane
% 可证明dot(B, knee_vector)恒为0，但是这个变量名起的有点蠢
knee_center = (knee_l + knee_r)/2;
knee_vector = knee_r - knee_l;
knee_vector_xy = [knee_vector(1) knee_vector(2) 0];
b = [1 0 0];
b = b - dot(b, knee_vector_xy) / dot(knee_vector_xy, knee_vector_xy) * knee_vector_xy;
B = b / norm(b) ;
%% Get the KAM value along with direction of "B"
ARM = CoP - knee_center;
KAM = cross(ARM, force);
result = dot(KAM, B);
if strcmp(leg_side,'l')
    KAM = 100 * result / (weight * height);  % 100 for cm to m - height
else
    KAM =- 100 * result / (weight * height);
end
end



