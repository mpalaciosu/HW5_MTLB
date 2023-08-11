% TAREA 5 : BLACK-SCHOLES TESTING
% MARKET CONDITIONS : 

clear
clc

rng(777) % seed

K = 500; % Strike price
S0 = 499.75; % underlying spot
r = 0.05; % risk free domestic
q = 0.01; % risk free foreign
sigma = 0.08;
T = 1;
deltat = 1 / 252; 

% QUESTION 1 : 

d1 = getd1(S0, K, r, q, T, sigma, 0);
d2 = getd2(d1, sigma, T, 0);

Price = PriceOption(1, S0, q, T, d1, K, r, d2, 0);

% PROBAMOS QUE ESTE TODO OK :
H0 = Price;
DELTA0 = getdelta(q, T, d1, 0);
B0 = getB(H0, DELTA0, S0);

% QUESTION 2 : 
M = 1000;
mu = 0.15;

z = zeros(M, 252);
S = zeros(M, 252);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal
lnST = log(ST); % Ln Spot terminal

hold on
figure(1)
histfit(lnST, 50);
title('Distribution Ln(S_T)');
legend('Empirical Distribution', 'Theoretical Distribution');
hold off

% PREGUNTA 3
d1 = zeros(1000, 252);
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

% Pregunta 4 : 
mediaY = zeros(1,252);
stdevY = zeros(1,252);
mediaX = zeros(1,252);
stdevX = zeros(1,252);

for i = 2:252
    mediaY(i) = mean(Y(:,i-1));
    stdevY(i) = std(Y(:,i-1));
    mediaX(i) = mean(X(:,i-1));
    stdevX(i) = std(X(:,i-1));
end

figure(2)
hold on
plot(mediaY);
plot(stdevY);
title('Mean and Std for Y');
legend('Mean Y', 'Std Y')
hold off

figure(3)
hold on
plot(mediaX);
plot(stdevX);
title('Mean and Std for X');
legend('Mean X', 'Std X')
hold off

% Pregunta 5 :
figure(4)
hold on
histogram(delta(:,126), 50, Normalization="probability");
title('Distribution of delta on the 125th day');
legend('Empirical Distribution');
hold off

Y252 = Y(:, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redo questions 2, 3 and 4 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Question 6 (week) :

% QUESTION 6.2 : 
M = 1000;
mu = 0.15;

z = zeros(M, 48);
S = zeros(M, 48);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:48
        z(i,j) = get_z();
        if j < 48
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal
lnST = log(ST); % Ln Spot terminal

hold on
figure(5)
histfit(lnST, 50);
title('Distribution Ln(S_T)');
legend('Empirical Distribution', 'Theoretical Distribution');
hold off

% PREGUNTA 6.3

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 48);
delta = zeros(1000, 48);
H = zeros(1000, 48);
B = zeros(1000, 48);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 48);
V = zeros(1000,48);
V(:,1) = Price;
Y = zeros(1000, 48);
X = zeros(1000,48);

for i = 1:M % Path
    for j = 1:48 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:48
          if j<48 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:48
        if j < 48
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

% Pregunta 6.4 : 
mediaY = zeros(1,48);
stdevY = zeros(1,48);
mediaX = zeros(1,48);
stdevX = zeros(1,48);

for i = 2:48
    mediaY(i) = mean(Y(:,i-1));
    stdevY(i) = std(Y(:,i-1));
    mediaX(i) = mean(X(:,i-1));
    stdevX(i) = std(X(:,i-1));
end

figure(6)
hold on
plot(mediaY);
plot(stdevY);
title('Median and Std for Y');
legend('Mean Y', 'Std Y')
hold off

figure(7)
hold on
plot(mediaX);
plot(stdevX);
title('Median and Std for X');
legend('Mean X', 'Std X')
hold off

Y48 = Y(:, 11);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redo questions 2, 3 and 4 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Question 6 (month) :

% QUESTION 6.2.1 : 
M = 1000;
mu = 0.15;
Q = 50;

z = zeros(M, 12);
S = zeros(M, 12);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:12
        z(i,j) = get_z();
        if j < 12
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal
lnST = log(ST); % Ln Spot terminal

hold on
figure(8)
histfit(lnST, 50);
title('Distribution Ln(S_T)');
legend('Empirical Distribution', 'Theoretical Distribution');
hold off

% PREGUNTA 6.3.1 :

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 12);
delta = zeros(1000, 12);
H = zeros(1000, 12);
B = zeros(1000, 12);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
q_diario = q;
r_diario = r;
d2 = zeros(1000, 12);
V = zeros(1000,12);
V(:,1) = Price;
Y = zeros(1000, 12);
X = zeros(1000,12);

for i = 1:M % Path
    for j = 1:12 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:12
          if j<12 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:12
        if j < 12
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

% Pregunta 6.4.1 : 
mediaY = zeros(1,12);
stdevY = zeros(1,12);
mediaX = zeros(1,12);
stdevX = zeros(1,12);

for i = 2:12
    mediaY(i) = mean(Y(:,i-1));
    stdevY(i) = std(Y(:,i-1));
    mediaX(i) = mean(X(:,i-1));
    stdevX(i) = std(X(:,i-1));
end

figure(9)
hold on
plot(mediaY);
plot(stdevY);
title('Median and Std for Y');
legend('Mean Y', 'Std Y')
hold off

figure(10)
hold on
plot(mediaX);
plot(stdevX);
title('Median and Std for X');
legend('Mean X', 'Std X')
hold off

Y12 = Y(:,11);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redo 2, 3 and 4 with mu = -0.15 and mu = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUESTION 7.2.1 : 
M = 1000;
mu = -0.15;

z = zeros(M, 252);
S = zeros(M, 252);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal
lnST = log(ST); % Ln Spot terminal

hold on
figure(11)
histfit(lnST, 50);
title('Distribution Ln(S_T)');
legend('Empirical Distribution', 'Theoretical Distribution');
hold off

% PREGUNTA 7.3.1

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 252);
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

% Pregunta 7.4.1 : 
mediaY = zeros(1,252);
stdevY = zeros(1,252);
mediaX = zeros(1,252);
stdevX = zeros(1,252);

for i = 2:252
    mediaY(i) = mean(Y(:,i-1));
    stdevY(i) = std(Y(:,i-1));
    mediaX(i) = mean(X(:,i-1));
    stdevX(i) = std(X(:,i-1));
end

figure(12)
hold on
plot(mediaY);
plot(stdevY);
title('Median and Std for Y');
legend('Mean Y', 'Std Y')
hold off

figure(13)
hold on
plot(mediaX);
plot(stdevX);
title('Median and Std for X');
legend('Mean X', 'Std X')
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redo 2, 3 and 4 with mu = -0.15 and mu = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUESTION 7.2.2 : 
M = 1000;
mu = 0;

z = zeros(M, 252);
S = zeros(M, 252);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal
lnST = log(ST); % Ln Spot terminal

hold on
figure(14)
histfit(lnST, 50);
title('Distribution Ln(S_T)');
legend('Empirical Distribution', 'Theoretical Distribution');
hold off

% PREGUNTA 7.3.2

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 252);
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

% Pregunta 7.4.2 : 
mediaY = zeros(1,252);
stdevY = zeros(1,252);
mediaX = zeros(1,252);
stdevX = zeros(1,252);

for i = 2:252
    mediaY(i) = mean(Y(:,i-1));
    stdevY(i) = std(Y(:,i-1));
    mediaX(i) = mean(X(:,i-1));
    stdevX(i) = std(X(:,i-1));
end

figure(15)
hold on
plot(mediaY);
plot(stdevY);
title('Median and Std for Y');
legend('Mean Y', 'Std Y')
hold off

figure(16)
hold on
plot(mediaX);
plot(stdevX);
title('Median and Std for X');
legend('Mean X', 'Std X')
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUESTION 8 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
rng(777) % seed

K = 500; % Strike price
S0 = 499.75; % underlying spot
r = 0.05; % risk free domestic
q = 0.01; % risk free foreign
sigma = 0.1;
T = 1;
deltat = 1 / 252; 

% QUESTION 1 : 

d1 = getd1(S0, K, r, q, T, sigma, 0);
d2 = getd2(d1, sigma, T, 0);

Price = PriceOption(1, S0, q, T, d1, K, r, d2, 0);

% PROBAMOS QUE ESTE TODO OK :
H0 = Price;
DELTA0 = getdelta(q, T, d1, 0);
B0 = getB(H0, DELTA0, S0);

% QUESTION 2 : 
M = 1000;
mu = 0.15;

z = zeros(M, 252);
S = zeros(M, 252);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, 0.08, deltat, z(i,j));
        end
    end
end

% PLOT : 

% PREGUNTA 3

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 252);
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 9 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
rng(777) % seed

K = 500; % Strike price
S0 = 499.75; % underlying spot
r = 0.05; % risk free domestic
q = 0.01; % risk free foreign
sigma = 0.06;
T = 1;
deltat = 1 / 252; 

% QUESTION 1 : 

d1 = getd1(S0, K, r, q, T, sigma, 0);
d2 = getd2(d1, sigma, T, 0);

Price = PriceOption(1, S0, q, T, d1, K, r, d2, 0);

% PROBAMOS QUE ESTE TODO OK :
H0 = Price;
DELTA0 = getdelta(q, T, d1, 0);
B0 = getB(H0, DELTA0, S0);

% QUESTION 2 : 
M = 1000;
mu = 0.15;

z = zeros(M, 252);
S = zeros(M, 252);

for i = 1:M
    S(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, 0.08, deltat, z(i,j));
        end
    end
end

% PREGUNTA 3

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 252);
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 


            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, S(i,j), q, T, d1(i,j), K, r, d2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUESTION 10 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
rng(777) % seed

K = 500; % Strike price
S0 = 499.75; % underlying spot
r = 0.05; % risk free domestic
q = 0.01; % risk free foreign
sigma = 0.08;
T = 1;
deltat = 1 / 252; 

% QUESTION 1 : 

d1 = getd1(S0, K, r, q, T, sigma, 0);
d2 = getd2(d1, sigma, T, 0);

Price = PriceOption(1, S0, q, T, d1, K, r, d2, 0);

% PROBAMOS QUE ESTE TODO OK :
H0 = Price;
X0 = Price - H0;
DELTA0 = getdelta(q, T, d1, 0);
B0 = getB(H0, DELTA0, S0);

% QUESTION 2 : 
M = 1000;
mu = 0.15;

z = zeros(M, 252);
S = zeros(M, 252);
fS = zeros(M,252);

for i = 1:M
    S(i,1) = S0;
    fS(i,1) = S0;
end

% GENERATE SPOT :
for i = 1:M
    for j = 1:252
        z(i,j) = get_z();
        if j < 252
            S(i,j+1) = get_spot(S(i,j), mu, sigma, deltat, z(i,j));
            fS(i,j+1) = get_spot(fS(i,j), mu, 0.1, deltat, z(i,j));
        end
    end
end

% PLOT : 
ST = S(:, end); % Spot terminal

% PREGUNTA 3

% Defino todas las matrices como ceros pero reemplazo el primer valor por
% los obtenidos en la confirmacion de la pregunta 1.

% Justify which of these quantities a priori massively depend on the physical drift µ.

d1 = zeros(1000, 252);
fd1 = d1;
delta = zeros(1000, 252);
H = zeros(1000, 252);
B = zeros(1000, 252);
H(:,1) = H0;
delta(:,1) = DELTA0;
B(:,1) = B0;
d2 = zeros(1000, 252);
fd2 = d2;
V = zeros(1000,252);
V(:,1) = Price;
Y = zeros(1000, 252);
X = zeros(1000,252);

for i = 1:M % Path
    for j = 1:252 % Columnas
        % Designat t : 
        t = (j - 1) * deltat;

        % Calcular d1 : 
        d1(i,j) = getd1(S(i,j), K, r, q, T , sigma, t);
        fd1(i,j) = getd1(S(i,j), K, r, q, T , 0.1, t);

        % Calcular delta : 
        delta(i,j) = getdelta(q, T, d1(i,j), t);
    end
end
    
for i = 1:M
    for j = 1:252
          if j<252 

            % Calcular H : 
            H(i,j+1) = getH(delta(i,j), q, deltat, S(i,j+1), B(i,j), r);

            % Calcular B : 
            B(i,j+1) = getB(H(i,j+1), delta(i,j+1), S(i,j+1)); 

          end

            % inicializar t : 
            t = (j - 1) * deltat;

            % Calcular d2 : 
            d2(i,j) = getd2(d1(i,j), sigma, T, t);
            fd2(i,j) = getd2(fd1(i,j), 0.1, T, t);

            % Calular V : 
            V(i,j) = PriceOption(1, fS(i,j), q, T, fd1(i,j), K, r, fd2(i,j), t);  
    end
end

% Y = PnL = deltaH -deltaV
for i = 1:M
    for j = 1:252
        if j < 252
        
        % Calculamos PnL :
        Y(i,j) = (H(i, j+1) - H(i,j) - (V(i,j+1)-V(i,j)));  

        end

        % Calculamos X : 
        X(i,j) = H(i,j) - V(i,j);

    end
end



