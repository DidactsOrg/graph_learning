clc; clear all; close all; 
S = load('S.mat');           % Precomputed XX^T matrix (normalized) 
S= S.S;                      % This can be commented if S is loaded in double format

load('top_PMT_coordinates.mat'); x_s = points(:,1); y_s = points(:,2);  % PMT coordinates
figure(1)
surf(S); view(2); xlim([1 127]);ylim([1 127]); colorbar; title('Sorted dataset'); xlabel('PMT index'); ylabel('PMT index');shading interp;

%% Block coordinate descent : 

% tri = delaunay(x_s,y_s);
% g = digraph(tri, tri(:, [2 3 1]));
% A = adjacency(g);
% A = A | A';
% Ad = full(double(A));  % Delaunay triangulation adjacency 

load('Ad.mat');          % Delaunay triangulation adjacency 


% parameters for Block coordinate descent algorithm 
% my_eps_outer = 1e-7; my_eps_inner = 1e-12; max_cycles = 100; scale = 1; isNormalized = 0;
% 
% A_init = ones(size(S)) - eye(size(S));   % Initialization with a full connected matrix       
% A_init1 = Ad;                            % Initialization with a Delaunay matrix 
% alpha = 0.01;                            % Sparsity regularizer for off-diagonal elements 
% 
% [L_est,~,convergence] = estimate_cgl(S,A_init1,alpha,my_eps_outer,my_eps_inner,max_cycles,2);
% 
% L = L_est; 

%% Graph learning using convex optimization software CVX : ( available in both MATLAB (http://cvxr.com/cvx/) and Python (https://www.cvxpy.org/)) 
n = 127; id = eye(n); 

% Standard graph leaning based on smoothness-prior 

cvx_begin
variable L_est(n,n) symmetric;
minimize(trace(L_est*S))        % For sparse off-diagonal elemnts ( trace(L_est*S) + 0.1*norm(L_est(~id),1))
subject to
     L_est == semidefinite(n);
     L_est*ones(n,1) == 0;
     L_est(~id)<= 0;
cvx_end

L = L_est*10^11;                % overall scaling (no change in the structure) 
%                        
                          
 
%% Laplacian to adjacency 

edge_th = 0.05;                 % threshold to select an edge 
A = L - diag(diag(L));          % replacing the diagonal elements to zero 
A = -A; 
A(A<edge_th) = 0; 
A(A>0)=1;


%% Plotting 
figure(3) 
surf(A); view(2); xlim([1 127]);ylim([1 127]); colorbar; title('Weighted adjacency'); xlabel('PMT index'); ylabel('PMT index');
load('top_PMT_coordinates.mat'); x_s = points(:,1); y_s = points(:,2);

figure(4)
gplot(A,[x_s y_s]);xlabel('x (cm)'); ylabel('y (cm)');title('Learned graph structure'); 