import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "@/lib/api";

// Summary & Deployment
export const useSummary = () => useQuery({ queryKey: ["summary"], queryFn: api.getSummary, refetchInterval: 30000 });
export const useCurrentDeployment = () => useQuery({ queryKey: ["current-deployment"], queryFn: api.getCurrentDeployment, refetchInterval: 10000 });

// Health
export const useHealthStatus = (model_version?: string) => useQuery({ 
  queryKey: ["health-status", model_version], 
  queryFn: () => api.getHealthStatus(model_version),
  refetchInterval: 15000
});

export const useHealthAlerts = (params?: { model_version?: string; severity?: string; dimension?: string; limit?: number }) => useQuery({
  queryKey: ["health-alerts", params],
  queryFn: () => api.getHealthAlerts(params),
  refetchInterval: 15000
});

// Approvals
export const useApprovals = (params?: { gate_type?: string; status?: string; limit?: number }) => useQuery({
  queryKey: ["approvals", params],
  queryFn: () => api.listApprovals(params),
  refetchInterval: 10000
});

export const useApproveApproval = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ requestId, reason }: { requestId: string; reason?: string }) => api.approveApproval(requestId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["approvals"] });
    }
  });
};

export const useRejectApproval = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ requestId, reason }: { requestId: string; reason: string }) => api.rejectApproval(requestId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["approvals"] });
    }
  });
};

// Symbol Performance
export const useSymbolPerformance = (params?: { model_version?: string; since?: number; until?: number }) => useQuery({
  queryKey: ["symbol-performance", params],
  queryFn: () => api.getSymbolPerformance(params)
});

// Calibration
export const useCalibration = (params?: { model_version?: string; since?: number; until?: number; bucket_width?: number }) => useQuery({
  queryKey: ["calibration", params],
  queryFn: () => api.getCalibration(params)
});

// Reasoning
export const useReasoningDistribution = (params?: { model_version?: string; symbol?: string; since?: number; until?: number; bucket_seconds?: number }) => useQuery({
  queryKey: ["reasoning-distribution", params],
  queryFn: () => api.getReasoningDistribution(params)
});

// History / Lineage / Features
export const useDeploymentHistory = (params?: { model_version?: string; event_type?: string; limit?: number }) => useQuery({
  queryKey: ["deployment-history", params],
  queryFn: () => api.getDeploymentHistory(params)
});

export const useTrainingRuns = (params?: { experiment_id?: string; pipeline_job_id?: string; limit?: number }) => useQuery({
  queryKey: ["training-runs", params],
  queryFn: () => api.listTrainingRuns(params)
});

export const useFeatures = (status?: string) => useQuery({
  queryKey: ["features", status],
  queryFn: () => api.listFeatures(status)
});
