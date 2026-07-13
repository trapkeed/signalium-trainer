import { useDeployments, useDeploymentHistory } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, ValueDisplay, formatTime } from "@/components/ui-utils";
import { PackageCheck, Clock, ArrowUpCircle } from "lucide-react";

function statusVariant(status: string): "success" | "warning" | "destructive" | "outline" {
  const s = status?.toUpperCase();
  if (s === "ACTIVE" || s === "DEPLOYED") return "success";
  if (s === "PENDING" || s === "STAGING") return "warning";
  if (s === "FAILED" || s === "ROLLED_BACK") return "destructive";
  return "outline";
}

export default function DeploymentCandidates() {
  const { data: deploymentsData, isLoading: loadingDep, error: errDep } = useDeployments();
  const { data: historyData, isLoading: loadingHist, error: errHist } = useDeploymentHistory({ limit: 30 });

  if (loadingDep || loadingHist) return <LoadingScreen />;
  if (errDep) return <ErrorScreen error={errDep} />;
  if (errHist) return <ErrorScreen error={errHist} />;

  const deployments = deploymentsData?.deployments ?? [];
  const events = historyData?.events ?? [];

  const active = deployments.find((d) => d.status?.toUpperCase() === "ACTIVE");

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Deployment Candidates</h1>
          <p className="text-sm text-muted-foreground mt-1">Model version promotion history and state</p>
        </div>
      </div>

      {/* Active deployment banner */}
      {active && (
        <div className="p-5 rounded-lg border border-primary/30 bg-primary/5 flex items-center gap-6">
          <div className="p-3 bg-primary/20 rounded-full text-primary">
            <ArrowUpCircle className="w-7 h-7" />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 flex-1">
            <ValueDisplay label="Active Model" value={active.model_version} />
            <ValueDisplay label="Status" value={<Badge variant="success">{active.status}</Badge>} />
            <ValueDisplay
              label="Deployed At"
              value={active.deployed_at ? formatTime(active.deployed_at) : "—"}
            />
            <ValueDisplay
              label="Weights"
              value={
                <span className="truncate max-w-[120px] block text-sm" title={active.weights_url as string | undefined}>
                  {(active.weights_url as string | undefined)?.split("/").pop() ?? "—"}
                </span>
              }
            />
          </div>
        </div>
      )}

      {/* All deployment records */}
      <Card>
        <CardHeader
          title="Deployment Registry"
          icon={<PackageCheck className="w-4 h-4 text-primary" />}
          subtitle={`${deployments.length} model version${deployments.length !== 1 ? "s" : ""}`}
        />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-y border-border bg-secondary/30">
                {["Model Version", "Status", "Deployed At", "Weights"].map((h) => (
                  <th key={h} className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {deployments.length > 0 ? (
                deployments.map((dep) => (
                  <tr key={dep.model_version} className="hover:bg-secondary/20 transition-colors">
                    <td className="px-4 py-3">
                      <span className="text-sm font-mono font-bold text-foreground">{dep.model_version}</span>
                    </td>
                    <td className="px-4 py-3">
                      <Badge variant={statusVariant(dep.status)}>{dep.status || "—"}</Badge>
                    </td>
                    <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
                      {dep.deployed_at ? formatTime(dep.deployed_at) : "—"}
                    </td>
                    <td className="px-4 py-3 text-xs font-mono text-muted-foreground truncate max-w-[200px]">
                      {(dep.weights_url as string | undefined)?.split("/").pop() ?? "—"}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="p-8">
                    <EmptyScreen message="No deployment records found" icon={PackageCheck} />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Event timeline */}
      <Card>
        <CardHeader
          title="Deployment Event Log"
          icon={<Clock className="w-4 h-4 text-muted-foreground" />}
          subtitle="Promotion, rollback, and lifecycle events"
        />
        <div className="divide-y divide-border">
          {events.length > 0 ? (
            events.map((ev, idx) => (
              <div key={idx} className="flex items-center gap-4 px-4 py-3 hover:bg-secondary/20 transition-colors">
                <div className={`w-2 h-2 rounded-full shrink-0 ${
                  ev.event_type?.toUpperCase().includes("PROMOTE") ? "bg-primary" :
                  ev.event_type?.toUpperCase().includes("ROLLBACK") ? "bg-destructive" :
                  ev.event_type?.toUpperCase().includes("DEPLOY") ? "bg-primary" : "bg-muted-foreground"
                }`} />
                <span className="text-xs font-mono font-medium text-foreground uppercase w-32 shrink-0">
                  {ev.event_type}
                </span>
                <span className="text-sm font-mono text-muted-foreground flex-1">
                  {ev.model_version}
                </span>
                <span className="text-xs font-mono text-muted-foreground flex items-center gap-1 shrink-0">
                  <Clock className="w-3 h-3" />
                  {formatTime(ev.timestamp)}
                </span>
              </div>
            ))
          ) : (
            <div className="p-8">
              <EmptyScreen message="No deployment events recorded" icon={Clock} />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
