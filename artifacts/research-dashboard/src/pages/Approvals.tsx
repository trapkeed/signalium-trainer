import { useState } from "react";
import { useApprovals, useApproveApproval, useRejectApproval } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { Check, X, Shield, Clock } from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";

export default function Approvals() {
  const [statusFilter, setStatusFilter] = useState("PENDING");
  const { data, isLoading, error } = useApprovals({ status: statusFilter !== "ALL" ? statusFilter : undefined });
  
  const approveReq = useApproveApproval();
  const rejectReq = useRejectApproval();

  const [rejectId, setRejectId] = useState<string | null>(null);
  const [rejectReason, setRejectReason] = useState("");

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const approvals = data?.approval_requests || [];

  const handleApprove = (id: string) => {
    if (confirm("AUTHORIZATION REQUIRED\n\nAre you sure you want to approve this action? This will proceed to the next pipeline stage.")) {
      approveReq.mutate({ requestId: id });
    }
  };

  const submitReject = (e: React.FormEvent) => {
    e.preventDefault();
    if (rejectId && rejectReason.trim()) {
      rejectReq.mutate({ requestId: rejectId, reason: rejectReason.trim() });
      setRejectId(null);
      setRejectReason("");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Manual Gates</h1>
          <p className="text-sm text-muted-foreground mt-1">Review and authorize pipeline transitions</p>
        </div>
        <div className="flex gap-2">
          {["PENDING", "APPROVED", "REJECTED", "ALL"].map(s => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-1.5 text-xs font-mono uppercase rounded transition-colors ${
                statusFilter === s 
                  ? "bg-secondary text-foreground font-bold" 
                  : "text-muted-foreground hover:bg-secondary/50"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4">
        {approvals.length > 0 ? approvals.map(req => (
          <Card key={req.request_id} className={`border-l-4 ${
            req.status === "PENDING" ? "border-l-accent bg-accent/5" :
            req.status === "APPROVED" ? "border-l-primary" :
            req.status === "REJECTED" ? "border-l-destructive" :
            "border-l-muted"
          }`}>
            <CardContent className="p-5 flex flex-col md:flex-row gap-6 justify-between">
              <div className="space-y-3 flex-1">
                <div className="flex items-center gap-3">
                  <Badge variant={
                    req.status === "PENDING" ? "warning" :
                    req.status === "APPROVED" ? "success" :
                    req.status === "REJECTED" ? "destructive" : "outline"
                  }>
                    {req.status}
                  </Badge>
                  <span className="text-xs font-mono text-muted-foreground">ID: {req.request_id}</span>
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1 font-mono">
                    <Clock className="w-3 h-3" /> {formatTime(req.created_at)}
                  </span>
                </div>
                
                <div>
                  <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
                    <Shield className="w-4 h-4 text-muted-foreground" />
                    {req.gate_type}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1">{req.action_summary}</p>
                </div>

                <div className="grid grid-cols-2 gap-4 text-xs font-mono bg-background/50 p-3 rounded border border-border">
                  <div><span className="text-muted-foreground block mb-0.5">Resource ID</span> {req.resource_id}</div>
                  <div><span className="text-muted-foreground block mb-0.5">Resource Type</span> {req.resource_type || "N/A"}</div>
                  <div><span className="text-muted-foreground block mb-0.5">Requested By</span> {req.requested_by || "System"}</div>
                  {req.resolved_by && <div><span className="text-muted-foreground block mb-0.5">Resolved By</span> {req.resolved_by}</div>}
                </div>
              </div>

              {req.status === "PENDING" && (
                <div className="flex flex-col gap-3 shrink-0 md:w-48 justify-center border-t md:border-t-0 md:border-l border-border pt-4 md:pt-0 md:pl-6">
                  <button 
                    onClick={() => handleApprove(req.request_id)}
                    disabled={approveReq.isPending || rejectReq.isPending}
                    className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 h-10 rounded font-mono text-sm uppercase tracking-wide flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                  >
                    <Check className="w-4 h-4" /> Approve
                  </button>
                  <button 
                    onClick={() => setRejectId(req.request_id)}
                    disabled={approveReq.isPending || rejectReq.isPending}
                    className="w-full bg-destructive/10 hover:bg-destructive/20 text-destructive border border-destructive/30 h-10 rounded font-mono text-sm uppercase tracking-wide flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                  >
                    <X className="w-4 h-4" /> Reject
                  </button>
                </div>
              )}
            </CardContent>
          </Card>
        )) : (
          <EmptyScreen message={`No ${statusFilter !== "ALL" ? statusFilter.toLowerCase() : ""} approvals found`} icon={Shield} />
        )}
      </div>

      {/* Reject Dialog */}
      <Dialog.Root open={!!rejectId} onOpenChange={(open) => !open && setRejectId(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-background/80 backdrop-blur-sm" />
          <Dialog.Content className="fixed top-[50%] left-[50%] max-h-[85vh] w-[90vw] max-w-[500px] translate-x-[-50%] translate-y-[-50%] rounded-lg bg-card border border-border p-6 shadow-2xl focus:outline-none flex flex-col gap-4">
            <Dialog.Title className="text-lg font-mono font-bold uppercase text-destructive flex items-center gap-2">
              <X className="w-5 h-5" /> Reject Authorization
            </Dialog.Title>
            <Dialog.Description className="text-sm text-muted-foreground">
              Provide a reason for rejecting this pipeline transition. This will be logged in the audit trail.
            </Dialog.Description>
            
            <form onSubmit={submitReject} className="flex flex-col gap-4 mt-2">
              <textarea
                value={rejectReason}
                onChange={e => setRejectReason(e.target.value)}
                placeholder="Reason for rejection..."
                className="w-full min-h-[100px] bg-input border border-border rounded p-3 text-sm font-mono text-foreground focus:outline-none focus:border-destructive transition-colors resize-none"
                autoFocus
                required
              />
              <div className="flex justify-end gap-3 mt-2">
                <button
                  type="button"
                  onClick={() => setRejectId(null)}
                  className="px-4 py-2 text-sm font-mono uppercase text-muted-foreground hover:bg-secondary rounded transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!rejectReason.trim()}
                  className="px-4 py-2 text-sm font-mono uppercase bg-destructive text-destructive-foreground rounded hover:bg-destructive/90 transition-colors disabled:opacity-50"
                >
                  Confirm Reject
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
