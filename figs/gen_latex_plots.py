symbs20 = ['AAPL', 'AFL', 'AMZN', 'BA', 'CVX', 'DAL', 'DIS', 'FB', 'GE', 'GM', 'GOOG',
         'GS', 'JNJ', 'JPM', 'MAR', 'KO', 'MCD', 'NKE', 'PG', 'VZ', 'WMT']

for symb in symbs20:
    print("\\begin{figure}[!htb] ")
    print("\t\\centering ")
    print("\t\t \\includegraphics[width=\\textwidth]{stock/figs/plot_%s.pdf}"%symb)
    print("\t\\caption{Showcase %s of predicted v.s. actual daily prices of one stock over test period, 2019/08/16-2020/02/14.}"%symb)
    print("\t\\label{fig:plot_%s}"%symb)
    print("\\end{figure}\n")
