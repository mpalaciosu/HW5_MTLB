% DOCUMENTACION GET SPOT :   

%   Funcion busca devolver el spot del dia siguiente dado una serie de
%   parametros.

% PARAMETROS : 

%   spot_t0 : float
%       Spot del dia anterior
%   mu : float
%       Drift tipo de cambio
%   sigma : float
%       Desv estandar tipo de cambio
%   delta_t : float
%       Cambio en el tiempo
%   z: float
%       Innovaciones Gauss(0,1)

function spot_t1 = get_spot(spot_i, mu, sigma, delta_t, z)

spot_t1 = spot_i * exp((mu - (sigma ^ 2) / 2) * delta_t + sigma * sqrt(delta_t) * z);

return 
