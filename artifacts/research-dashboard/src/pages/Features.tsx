import { useState } from "react";
import { useFeatures } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge } from "@/components/ui-utils";

export default function Features() {
  const [statusFilter, setStatusFilter] = useState("");
  const { data, isLoading, error } = useFeatures(statusFilter || undefined);

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const features = data?.features || [];

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Feature Discovery</h1>
          <p className="text-sm text-muted-foreground mt-1">Candidate features and importance scoring</p>
        </div>
        <div className="flex gap-2">
          <select 
            value={statusFilter} 
            onChange={e => setStatusFilter(e.target.value)}
            className="text-xs font-mono bg-input border border-border rounded px-3 py-1.5 text-foreground focus:outline-none focus:border-primary"
          >
            <option value="">All Statuses</option>
            <option value="CANDIDATE">Candidate</option>
            <option value="EVALUATING">Evaluating</option>
            <option value="ACCEPTED">Accepted</option>
            <option value="REJECTED">Rejected</option>
            <option value="DEPRECATED">Deprecated</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {features.length > 0 ? features.map(feat => (
          <Card key={feat.feature_id} className="flex flex-col">
            <CardContent className="p-5 flex-1 flex flex-col">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-mono font-bold text-foreground">{feat.name}</h3>
                  <span className="text-xs font-mono text-muted-foreground uppercase">{feat.category}</span>
                </div>
                <Badge variant={
                  feat.status === "ACCEPTED" ? "success" :
                  feat.status === "REJECTED" ? "destructive" :
                  feat.status === "DEPRECATED" ? "outline" : "warning"
                }>
                  {feat.status}
                </Badge>
              </div>

              <div className="mt-auto pt-4 border-t border-border space-y-3">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-muted-foreground">Importance Score</span>
                    <span className="text-primary font-bold">{feat.importance_score.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 w-full bg-secondary rounded overflow-hidden">
                    <div 
                      className="h-full bg-primary rounded" 
                      style={{ width: `${Math.min(100, Math.max(0, feat.importance_score * 100))}%` }}
                    />
                  </div>
                </div>

                {feat.notes && (
                  <p className="text-xs text-muted-foreground italic border-l-2 border-border pl-2 py-1">
                    "{feat.notes}"
                  </p>
                )}
                
                <div className="text-[10px] text-muted-foreground font-mono text-right mt-2">
                  ID: {feat.feature_id.split('-')[0]}
                </div>
              </div>
            </CardContent>
          </Card>
        )) : (
          <div className="col-span-full">
            <EmptyScreen message="No features matching filter" />
          </div>
        )}
      </div>
    </div>
  );
}
