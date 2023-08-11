% DOCUMENTACION GET Z :   

%   Funcion busca devolver a z para ser usado como input en otras funciones

% PARAMETROS : 

%   mu : float
%       Drift tipo de cambio
%   sigma : float
%       Desv estandar innovaciones

function z = get_z()

mu = 0;
sigma = 1;

z = normrnd(mu, sigma);

return
