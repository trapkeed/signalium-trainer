/**
 * Lightweight in-memory task queue.
 * No external dependencies (no Redis, no BullMQ).
 * Suitable for low-to-moderate workloads; swap for a proper queue in high-load production.
 */

export interface Task {
  id: string;
  type: string;
  payload: unknown;
  attempts: number;
  createdAt: Date;
}

type Handler = (task: Task) => Promise<void>;

const handlers = new Map<string, Handler>();
const queue: Task[] = [];
let running = false;

export function registerHandler(type: string, handler: Handler): void {
  handlers.set(type, handler);
}

export function enqueue(type: string, payload: unknown): void {
  queue.push({
    id: crypto.randomUUID(),
    type,
    payload,
    attempts: 0,
    createdAt: new Date(),
  });
  if (!running) void drain();
}

async function drain(): Promise<void> {
  running = true;
  while (queue.length > 0) {
    const task = queue.shift()!;
    const handler = handlers.get(task.type);
    if (!handler) continue;
    try {
      task.attempts += 1;
      await handler(task);
    } catch (err) {
      if (task.attempts < 3) {
        // Re-queue with backoff
        setTimeout(() => queue.push(task), task.attempts * 2000);
      }
    }
  }
  running = false;
}
