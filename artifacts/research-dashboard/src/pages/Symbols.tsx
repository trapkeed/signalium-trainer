import { useSymbolPerformance } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge } from "@/components/ui-utils";
import { useState } from "react";
import { Search } from "lucide-react";

export default function Symbols() {
  const [search, setSearch] = useState("");
  const { data, isLoading, error } = useSymbolPerformance();

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const symbols = data?.symbols || [];
  
  const filtered = symbols.filter(s => s.symbol.toLowerCase().includes(search.toLowerCase()));
  
  // Sort for best/worst leaderboards
  const bySharpe = [...symbols].sort((a, b) => b.sharpe - a.sharpe);
  const best = bySharpe.slice(0, 5);
  const worst = bySharpe.slice().reverse().slice(0, 5);

  const formatPercent = (val: number | null | undefined) => 
    val == null ? "N/A" : `${(val * 100).toFixed(1)}%`;
    
  const formatNum = (val: number | null | undefined) => 
    val == null ? "N/A" : val.toFixed(3);

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Symbol Leaderboard</h1>
          <p className="text-sm text-muted-foreground mt-1">Cross-sectional performance analytics</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="border-primary/30">
          <CardHeader title="Top Performers" subtitle="Highest Sharpe Ratios" />
          <CardContent noPadding>
            <div className="divide-y divide-border">
              {best.map(s => (
                <div key={s.symbol} className="p-3 flex items-center justify-between bg-primary/5">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded bg-secondary flex items-center justify-center font-mono font-bold text-foreground">
                      {s.symbol}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-mono text-primary">Sharpe: {formatNum(s.sharpe)}</span>
                      <span className="text-xs text-muted-foreground">Win Rate: {formatPercent(s.win_rate)}</span>
                    </div>
                  </div>
                  <div className="text-right flex flex-col">
                    <span className="text-sm font-mono text-foreground">${formatNum(s.total_pnl)}</span>
                    <span className="text-[10px] text-muted-foreground uppercase">{s.trade_count} trades</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-destructive/30">
          <CardHeader title="Bottom Performers" subtitle="Lowest Sharpe Ratios" />
          <CardContent noPadding>
            <div className="divide-y divide-border">
              {worst.map(s => (
                <div key={s.symbol} className="p-3 flex items-center justify-between bg-destructive/5">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded bg-secondary flex items-center justify-center font-mono font-bold text-foreground">
                      {s.symbol}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-mono text-destructive">Sharpe: {formatNum(s.sharpe)}</span>
                      <span className="text-xs text-muted-foreground">Win Rate: {formatPercent(s.win_rate)}</span>
                    </div>
                  </div>
                  <div className="text-right flex flex-col">
                    <span className="text-sm font-mono text-foreground">${formatNum(s.total_pnl)}</span>
                    <span className="text-[10px] text-muted-foreground uppercase">{s.trade_count} trades</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader 
          title="All Symbols" 
          action={
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder="Filter symbols..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="pl-9 pr-4 py-1.5 bg-input border border-border rounded text-sm font-mono text-foreground focus:outline-none focus:border-primary w-64 transition-colors"
              />
            </div>
          }
        />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-y border-border bg-secondary/30">
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">Symbol</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Trades</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Sharpe</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Win Rate</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Avg Conf</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Reliability</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Total PnL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.length > 0 ? filtered.map(s => (
                <tr key={s.symbol} className="hover:bg-secondary/20 transition-colors">
                  <td className="px-4 py-3 text-sm font-mono font-bold text-foreground">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: `hsl(${Math.abs(s.symbol.split('').reduce((a,b)=>a+b.charCodeAt(0),0)%360)} 70% 50%)` }}></span>
                      {s.symbol}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-right text-muted-foreground">{s.trade_count}</td>
                  <td className={`px-4 py-3 text-sm font-mono text-right ${s.sharpe >= 1 ? 'text-primary' : s.sharpe < 0 ? 'text-destructive' : 'text-foreground'}`}>
                    {formatNum(s.sharpe)}
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-right text-foreground">{formatPercent(s.win_rate)}</td>
                  <td className="px-4 py-3 text-sm font-mono text-right text-foreground">{formatPercent(s.avg_confidence_at_entry)}</td>
                  <td className="px-4 py-3 text-sm font-mono text-right text-foreground">
                    {s.confidence_reliability != null ? (
                      <Badge variant={s.confidence_reliability > 0.8 ? "success" : s.confidence_reliability < 0.5 ? "destructive" : "default"}>
                        {formatNum(s.confidence_reliability)}
                      </Badge>
                    ) : "-"}
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-right text-foreground">${formatNum(s.total_pnl)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={7} className="p-8">
                    <EmptyScreen message="No symbols found matching filter" />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
