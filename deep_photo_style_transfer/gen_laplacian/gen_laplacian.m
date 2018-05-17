pkg load image

addpath matting/
addpath gaimc/
N = 1;

for i = 1:N
    prefix = '../../ionut_resized';
    in_name = [prefix '.jpg'];

    input = im2double(imread(in_name));
    [h w c] = size(input);

    A = getLaplacian1(input, zeros(h, w), 1e-7, 1);

    n = nnz(A);
    [Ai, Aj, Aval] = find(A);
    CSC = [Ai, Aj, Aval];
    %save(['Input_Laplacian_3x3_1e-7_CSC' int2str(i) '.mat'], 'CSC');

    [rp ci ai] = sparse_to_csr(A);
    Ai = sort(Ai);
    Aj = ci;
    Aval = ai;
    CSR = [Ai, Aj, Aval];
    % save(['Input_Laplacian_3x3_1e-7_CSR' int2str(i) '.mat'], 'CSR'); % save -6 laplacian.mat CSR
    save -6 ['Input_Laplacian_3x3_1e-7_CSR' int2str(i) '.mat'] CSR
end
