%% This function extracts n training samples of size sz from Signal S.
% The input S can be either a 1-D signal or 2-D signal (image). For a 1-D
% signal n training sequences having sz samples will be extracted from
% random positions. For a 2-D Signal n training patches of size
% [sz(1),sz(2)] or [sz,sz] will be extracted from n random positions.
% (c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
% Muenchen, 2012. Contact: simon.hawe@tum.de
function [X] = extract_training_set(S, n, sz)

if isscalar(sz) && min(size(S)) > 1
    sz = [sz,sz];
end

patches = im2col(S,sz,'sliding');

patches_ext = bsxfun(@minus,patches,mean(patches));
patches_ext = sqrt(sum(patches_ext.^2));
sel = isnan(1./patches_ext) | isinf(1./patches_ext);
patches(:,sel)=[];
patches_ext(:,sel) = [];
[~,b]=sort(patches_ext.*rand(size(patches_ext)),'descend');
%[~,bb]=sort(patches_ext,'descend');
X = patches(:,b(1:min(end,n)));

%patches_ext(:,isnan(1./sum(patches_ext)))=[];
%p = 1:size(patches,2);
%sel = randperm(numel(p));
%X = patches(:,p(sel(1:min(end,n))));



