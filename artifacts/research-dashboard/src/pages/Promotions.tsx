import { useDeploymentHistory, useCurrentDeployment } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { GitMerge, Zap, ArrowDown, Activity } from "lucide-react";

export default function Promotions() {
  const { data: historyData, isLoading: isLoadingHist, error: errorHist } = useDeploymentHistory({ limit: 100 });
  const { data: deploymentData, isLoading: isLoadingDep } = useCurrentDeployment();

  if (isLoadingHist || isLoadingDep) return <LoadingScreen />;
  if (errorHist) return <ErrorScreen error={errorHist} />;

  const events = historyData?.events || [];
  const active = deploymentData?.active;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Deployment & Promotions</h1>
        <p className="text-sm text-muted-foreground mt-1">Audit log of model state transitions</p>
      </div>

      {active && (
        <Card className="border-primary/50 shadow-[0_0_15px_rgba(0,229,155,0.1)] relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4">
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
            </span>
          </div>
          <CardHeader 
            title="CURRENT ACTIVE MODEL" 
            icon={<Zap className="w-4 h-4 text-primary" />}
          />
          <CardContent className="bg-primary/5">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-[10px] text-muted-foreground uppercase font-mono block mb-1">Version</span>
                <span className="text-xl font-mono font-bold text-foreground">{active.model_version}</span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground uppercase font-mono block mb-1">Deployed</span>
                <span className="text-sm font-mono text-foreground">{active.deployed_at ? formatTime(active.deployed_at) : "N/A"}</span>
              </div>
              <div className="col-span-2">
                <span className="text-[10px] text-muted-foreground uppercase font-mono block mb-1">Weights Bundle</span>
                <code className="text-xs text-primary bg-primary/10 px-2 py-1 rounded border border-primary/20 block truncate">
                  {active.weights_url || "local://memory"}
                </code>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="relative">
        {events.length > 0 ? (
          <div className="space-y-6 before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-border">
            {events.map((ev, i) => (
              <div key={`${ev.model_version}-${ev.timestamp}-${i}`} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                {/* Timeline dot */}
                <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-background bg-secondary shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                  {ev.event_type === "PROMOTED_TO_ACTIVE" ? (
                    <Zap className="w-4 h-4 text-primary" />
                  ) : ev.event_type === "DEMOTED" ? (
                    <ArrowDown className="w-4 h-4 text-destructive" />
                  ) : (
                    <Activity className="w-4 h-4 text-accent" />
                  )}
                </div>

                {/* Event Card */}
                <Card className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] hover:border-muted-foreground/50 transition-colors">
                  <div className="p-4 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <Badge variant={
                        ev.event_type === "PROMOTED_TO_ACTIVE" ? "success" : 
                        ev.event_type === "DEMOTED" ? "destructive" : "default"
                      }>
                        {ev.event_type.replace(/_/g, ' ')}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground font-mono">{formatTime(ev.timestamp)}</span>
                    </div>
                    
                    <div>
                      <span className="text-xs text-muted-foreground font-mono block mb-1">Target Version</span>
                      <span className="text-base font-mono font-bold text-foreground">{ev.model_version}</span>
                    </div>
                    
                    {!!(ev.metadata && Object.keys(ev.metadata as Record<string, unknown>).length > 0) && (
                      <div className="mt-2 pt-2 border-t border-border/50 text-xs font-mono text-muted-foreground">
                        <pre className="whitespace-pre-wrap">
                          {JSON.stringify(ev.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </Card>
              </div>
            ))}
          </div>
        ) : (
          <EmptyScreen message="No promotion history found" icon={GitMerge} />
        )}
      </div>
    </div>
  );
}
