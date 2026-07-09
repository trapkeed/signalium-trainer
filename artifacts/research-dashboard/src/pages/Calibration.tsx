import { useCalibration } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, ValueDisplay } from "@/components/ui-utils";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Label } from "recharts";

export default function Calibration() {
  const { data, isLoading, error } = useCalibration();

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const ece = data?.expected_calibration_error;
  const curve = data?.curve || [];

  // Filter out buckets with no trades for cleaner plotting
  const plotData = curve
    .filter(b => b.trade_count > 0 && b.mean_confidence != null && b.observed_win_rate != null)
    .map(b => ({
      ...b,
      mean_confidence: Number(b.mean_confidence?.toFixed(3)),
      observed_win_rate: Number(b.observed_win_rate?.toFixed(3)),
      size: Math.max(b.trade_count * 0.1, 10) // scale circle by trade count
    }));

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Confidence Calibration</h1>
          <p className="text-sm text-muted-foreground mt-1">Reliability diagram: predicted vs observed</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-1 border-primary/20 bg-primary/5">
          <CardContent className="flex flex-col justify-center h-full min-h-[120px]">
            <ValueDisplay 
              label="Expected Calibration Error (ECE)" 
              value={ece != null ? (ece * 100).toFixed(2) + "%" : "N/A"} 
              subValue={ece != null ? (ece < 0.05 ? "Excellent" : ece < 0.1 ? "Acceptable" : "Poor") : ""}
            />
          </CardContent>
        </Card>
        
        <Card className="md:col-span-1">
          <CardContent className="flex flex-col justify-center h-full min-h-[120px]">
            <ValueDisplay 
              label="Total Calibration Trades" 
              value={data?.n_trades?.toLocaleString() || "0"} 
            />
          </CardContent>
        </Card>

        <Card className="md:col-span-1">
          <CardContent className="flex flex-col justify-center h-full min-h-[120px]">
            <ValueDisplay 
              label="Analyzed Buckets" 
              value={plotData.length} 
              subValue={`out of ${curve.length} total`}
            />
          </CardContent>
        </Card>
      </div>

      <Card className="h-[600px] flex flex-col">
        <CardHeader title="Reliability Diagram" />
        <CardContent className="flex-1 min-h-0 pt-6 pr-8">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={true} horizontal={true} />
              
              <XAxis 
                type="number" 
                dataKey="mean_confidence" 
                name="Mean Confidence" 
                domain={[0, 1]} 
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12, fontFamily: "monospace" }}
                stroke="hsl(var(--border))"
              >
                <Label value="Predicted Confidence" position="insideBottom" offset={-15} fill="hsl(var(--muted-foreground))" fontSize={12} className="font-mono uppercase" />
              </XAxis>
              
              <YAxis 
                type="number" 
                dataKey="observed_win_rate" 
                name="Observed Win Rate" 
                domain={[0, 1]}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12, fontFamily: "monospace" }}
                stroke="hsl(var(--border))"
              >
                <Label value="Observed Win Rate" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} fill="hsl(var(--muted-foreground))" fontSize={12} className="font-mono uppercase" />
              </YAxis>

              <Tooltip 
                cursor={{ strokeDasharray: '3 3', stroke: 'hsl(var(--muted-foreground))' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-popover border border-border p-3 rounded shadow-xl font-mono text-xs space-y-1">
                        <div className="text-muted-foreground uppercase mb-2 border-b border-border pb-1">Bucket: {data.bucket_lo.toFixed(2)} - {data.bucket_hi.toFixed(2)}</div>
                        <div className="flex justify-between gap-4"><span className="text-muted-foreground">Confidence:</span> <span className="text-foreground">{(data.mean_confidence * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between gap-4"><span className="text-muted-foreground">Win Rate:</span> <span className="text-foreground">{(data.observed_win_rate * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between gap-4"><span className="text-muted-foreground">Trades:</span> <span className="text-foreground">{data.trade_count}</span></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              
              {/* Perfect Calibration Line */}
              <ReferenceLine 
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} 
                stroke="hsl(var(--muted-foreground))" 
                strokeDasharray="5 5"
              />

              <Scatter 
                name="Buckets" 
                data={plotData} 
                fill="hsl(var(--primary))" 
                shape="circle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
