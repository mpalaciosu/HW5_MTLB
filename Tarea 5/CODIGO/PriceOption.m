function Price = PriceOption(callput_ticker, spot, q, T, d1, K, r, d2, t)

Price = callput_ticker * spot * exp(-q * T - t) * normcdf(callput_ticker * d1) - callput_ticker * K * exp(-r * T - t) * normcdf(callput_ticker * d2);

return