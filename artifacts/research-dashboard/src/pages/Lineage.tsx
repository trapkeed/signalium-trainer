import { useTrainingRuns } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, formatTime } from "@/components/ui-utils";
import { GitCommit, Database, Settings, FileBox } from "lucide-react";

export default function Lineage() {
  const { data, isLoading, error } = useTrainingRuns();

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const runs = data?.training_runs || [];

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Artifact Lineage</h1>
          <p className="text-sm text-muted-foreground mt-1">Traceability from dataset to model</p>
        </div>
      </div>

      <Card>
        <CardHeader title="Training Runs Log" />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-y border-border bg-secondary/30">
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">Run ID / Exp</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">Model Version</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">Lineage Hashes</th>
                <th className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {runs.length > 0 ? runs.map(run => (
                <tr key={run.run_id} className="hover:bg-secondary/20 transition-colors align-top">
                  <td className="px-4 py-4">
                    <div className="flex flex-col gap-1">
                      <span className="text-sm font-mono font-bold text-foreground">{run.run_id}</span>
                      <span className="text-xs font-mono text-muted-foreground">Exp: {run.experiment_id}</span>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    {run.model_version ? (
                      <span className="text-sm font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">{run.model_version}</span>
                    ) : (
                      <span className="text-xs text-muted-foreground italic">No artifact</span>
                    )}
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2 text-xs font-mono text-foreground">
                        <Database className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground w-12">Data:</span>
                        <span className="truncate max-w-[150px] block">{run.dataset_version}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs font-mono text-foreground">
                        <FileBox className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground w-12">Feat:</span>
                        <span className="truncate max-w-[150px] block">{run.feature_version}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs font-mono text-foreground">
                        <GitCommit className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground w-12">Code:</span>
                        <span className="truncate max-w-[150px] block text-accent">{run.code_commit}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs font-mono text-foreground">
                        <Settings className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground w-12">Conf:</span>
                        <span className="truncate max-w-[150px] block">{run.config_hash}</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-xs font-mono text-muted-foreground">
                    {formatTime(run.created_at)}
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={4} className="p-8">
                    <EmptyScreen message="No training runs found" />
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
