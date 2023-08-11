% DOCUMENTACION GET Hi :   

%   Funcion busca devolver el valor del hedging portfolio para t+1

% PARAMETROS : 

%   deltai : float
%       Posicion larga en el activo riesgoso
%   q : float
%       Foreign risk free rate
%   delta_t : float
%        Cambio en el tiempo
%   spot_t1 : float
%       Spot para el periodo t+1
%   Bi : float
%       Posicion en la cuenta corriente MMA
%   r : float
%       Domestic risk free rate

function H = getH(delta_anterior, q_diario, delta_t, spot_actual, B_anterior, r_diario)

H = delta_anterior * exp(q_diario * delta_t) * spot_actual + B_anterior * exp(r_diario * delta_t);

return