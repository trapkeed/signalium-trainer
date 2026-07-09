import { useState } from "react";
import { useHealthAlerts, useHealthStatus, useReasoningDistribution } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Filter, Clock, AlertTriangle } from "lucide-react";

export default function Drift() {
  const [severityFilter, setSeverityFilter] = useState<string>("");
  const [dimFilter, setDimFilter] = useState<string>("");

  const { data: statusData, isLoading: isLoadingStatus } = useHealthStatus();
  const { data: alertsData, isLoading: isLoadingAlerts } = useHealthAlerts({ 
    severity: severityFilter || undefined,
    dimension: dimFilter || undefined,
    limit: 100 
  });
  
  // Just showing global reasoning drift for now
  const { data: distData, isLoading: isLoadingDist } = useReasoningDistribution({ bucket_seconds: 3600 });

  if (isLoadingStatus && isLoadingAlerts && isLoadingDist) return <LoadingScreen />;

  const dimensions = Object.entries(statusData?.status?.dimensions || {});
  const alerts = alertsData?.alerts || [];
  
  // Format timeseries for reasoning distribution
  const series = distData?.series || [];
  const chartData = series.map(b => {
    const formatted: { time: number; timeStr: string; [head: string]: number | string } = {
      time: b.bucket_start * 1000, 
      timeStr: new Date(b.bucket_start * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
    };
    for (const [head, mean] of Object.entries(b.head_means)) {
      formatted[head] = mean;
    }
    return formatted;
  });

  // Extract all unique heads from the first bucket for plotting lines
  const headNames = series.length > 0 ? Object.keys(series[0].head_means) : [];
  const colors = [
    "hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))", 
    "hsl(var(--chart-4))", "hsl(var(--chart-5))", "hsl(var(--primary))"
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Signal Drift Monitor</h1>
          <p className="text-sm text-muted-foreground mt-1">Telemetry and output distribution analysis</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-1 flex flex-col gap-6">
          <Card>
            <CardHeader title="Current Dimensions" />
            <CardContent noPadding>
              <div className="divide-y divide-border max-h-[300px] overflow-y-auto">
                {dimensions.length > 0 ? dimensions.map(([name, dim]) => (
                  <div key={name} className="p-3 hover:bg-secondary/20 transition-colors">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-mono text-foreground capitalize">{name}</span>
                      <Badge variant={
                        dim.severity === "CRITICAL" ? "destructive" :
                        dim.severity === "WARNING" ? "warning" : "success"
                      }>
                        {dim.severity}
                      </Badge>
                    </div>
                    {dim.message && <p className="text-xs text-muted-foreground">{dim.message}</p>}
                    <p className="text-[10px] text-muted-foreground font-mono mt-2 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {dim.last_checked ? formatTime(dim.last_checked) : "Never"}
                    </p>
                  </div>
                )) : (
                  <div className="p-4"><EmptyScreen message="No health data" /></div>
                )}
              </div>
            </CardContent>
          </Card>
          
          <Card className="flex-1 flex flex-col min-h-[400px]">
            <CardHeader 
              title="Alert Log" 
              action={
                <div className="flex gap-2">
                  <select 
                    value={severityFilter} 
                    onChange={e => setSeverityFilter(e.target.value)}
                    className="text-xs font-mono bg-input border border-border rounded px-2 py-1 text-foreground"
                  >
                    <option value="">All Severities</option>
                    <option value="CRITICAL">Critical</option>
                    <option value="WARNING">Warning</option>
                  </select>
                </div>
              }
            />
            <CardContent noPadding className="flex-1 overflow-hidden flex flex-col">
              <div className="flex-1 overflow-y-auto divide-y divide-border">
                {alerts.length > 0 ? alerts.map(alert => (
                  <div key={alert.alert_id} className={`p-4 ${alert.severity === 'CRITICAL' ? 'bg-destructive/5' : ''}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant={alert.severity === "CRITICAL" ? "destructive" : "warning"}>{alert.severity}</Badge>
                        <span className="text-xs font-mono text-foreground">{alert.dimension}</span>
                      </div>
                      <span className="text-[10px] text-muted-foreground font-mono">{formatTime(alert.created_at)}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{alert.message}</p>
                  </div>
                )) : (
                  <div className="p-8 h-full flex items-center justify-center">
                    <EmptyScreen message="No alerts match filters" icon={AlertTriangle} />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="xl:col-span-2">
          <Card className="h-[740px] flex flex-col">
            <CardHeader 
              title="Output Head Means (Reasoning Drift)" 
              subtitle="Time series of raw model outputs. Significant deviation indicates signal degradation."
            />
            <CardContent className="flex-1 min-h-0 pt-6 pr-8">
              {isLoadingDist ? <LoadingScreen /> : series.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis 
                      dataKey="timeStr" 
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11, fontFamily: "monospace" }}
                      stroke="hsl(var(--border))"
                      tickMargin={10}
                    />
                    <YAxis 
                      domain={['auto', 'auto']}
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11, fontFamily: "monospace" }}
                      stroke="hsl(var(--border))"
                      tickFormatter={(val) => val.toFixed(3)}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'hsl(var(--popover))', borderColor: 'hsl(var(--border))', borderRadius: '4px' }}
                      itemStyle={{ fontFamily: 'monospace', fontSize: '12px' }}
                      labelStyle={{ color: 'hsl(var(--muted-foreground))', marginBottom: '8px', fontFamily: 'monospace', fontSize: '11px' }}
                    />
                    <Legend 
                      wrapperStyle={{ fontFamily: 'monospace', fontSize: '11px', paddingTop: '20px' }}
                    />
                    {headNames.map((head, i) => (
                      <Line 
                        key={head}
                        type="monotone" 
                        dataKey={head} 
                        name={head}
                        stroke={colors[i % colors.length]} 
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center">
                  <EmptyScreen message="No reasoning distribution data" />
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
