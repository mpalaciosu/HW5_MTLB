function d2 = getd2(d1, sigma, T, t)

d2 = d1 - sigma * sqrt(T-t);

return