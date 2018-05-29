pkg load image

addpath matting/
addpath gaimc/
N = 7;

prefix = '../../images/content/d_content';
img_type = '.png';
laplacian_name = 'd_laplacian';

for i = 1:N
    in_name = [prefix int2str(i) img_type];

    input = im2double(imread(in_name));
    [h w c] = size(input);

    A = getLaplacian1(input, zeros(h, w), 1e-7, 1);

    n = nnz(A);
    [Ai, Aj, Aval] = find(A);
    % CSC = [Ai, Aj, Aval];
    % save([laplacian_name int2str(i) '.mat'], 'CSC');

    [rp ci ai] = sparse_to_csr(A);
    Ai = sort(Ai);
    Aj = ci;
    Aval = ai;
    CSR = [Ai, Aj, Aval];
    save('-v6', [laplacian_name int2str(i) '.mat'], 'CSR');

    clear A Ai Aj Aval CSR ai c ci h in_name input n rp w;
end
