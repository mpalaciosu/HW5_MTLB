function d1 = getd1(spot, K, r, q, T, sigma, t)

d1 = ((log(spot / K) + (r - q) * (T - t)) / (sigma * sqrt(T - t))) + (sigma * sqrt(T - t)) / 2;

return