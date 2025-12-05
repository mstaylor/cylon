//! Checkpoint trigger implementations.
//!
//! Triggers determine when checkpoints should be taken based on various criteria.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use super::config::TriggerConfig;
use super::traits::CheckpointTrigger;
use super::types::{CheckpointContext, CheckpointUrgency, OperationType};

/// Operation count-based trigger.
///
/// Triggers a checkpoint after a specified number of operations.
pub struct OperationCountTrigger {
    /// Number of operations before triggering
    threshold: u64,
    /// Current operation count
    count: AtomicU64,
    /// Bytes processed since last checkpoint
    bytes: AtomicU64,
    /// Whether a checkpoint has been forced
    forced: AtomicBool,
    /// Time of last checkpoint
    last_checkpoint: RwLock<Instant>,
}

impl OperationCountTrigger {
    /// Create a new operation count trigger
    pub fn new(threshold: u64) -> Self {
        Self {
            threshold,
            count: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            forced: AtomicBool::new(false),
            last_checkpoint: RwLock::new(Instant::now()),
        }
    }

    /// Get the current operation count
    pub fn current_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl CheckpointTrigger for OperationCountTrigger {
    fn record_operation(&self, _op_type: OperationType, bytes_processed: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes_processed, Ordering::Relaxed);
    }

    fn should_checkpoint(&self) -> bool {
        self.forced.load(Ordering::Relaxed)
            || self.count.load(Ordering::Relaxed) >= self.threshold
    }

    fn force_checkpoint(&self) {
        self.forced.store(true, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.forced.store(false, Ordering::Relaxed);
        if let Ok(mut last) = self.last_checkpoint.write() {
            *last = Instant::now();
        }
    }

    fn urgency(&self) -> CheckpointUrgency {
        if self.forced.load(Ordering::Relaxed) {
            CheckpointUrgency::Critical
        } else {
            let count = self.count.load(Ordering::Relaxed);
            let ratio = count as f64 / self.threshold as f64;

            if ratio >= 1.0 {
                CheckpointUrgency::Medium
            } else if ratio >= 0.8 {
                CheckpointUrgency::Low
            } else {
                CheckpointUrgency::None
            }
        }
    }

    fn get_context(&self) -> CheckpointContext {
        let time_since = self
            .last_checkpoint
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        CheckpointContext {
            operations_since_checkpoint: self.count.load(Ordering::Relaxed),
            bytes_since_checkpoint: self.bytes.load(Ordering::Relaxed),
            time_since_checkpoint: time_since,
            remaining_time_budget: None,
            memory_usage: None,
            memory_limit: None,
            last_checkpoint_id: None,
        }
    }
}

/// Time budget-based trigger for serverless environments.
///
/// Triggers a checkpoint when the remaining time budget drops below a threshold.
/// Essential for Lambda/Fargate where functions have hard time limits.
pub struct TimeBudgetTrigger {
    /// Start time of this execution
    start_time: Instant,
    /// Total time budget for this execution
    total_budget: Duration,
    /// Reserve time before deadline to trigger checkpoint
    reserve_time: Duration,
    /// Operations since last checkpoint
    operations: AtomicU64,
    /// Bytes processed since last checkpoint
    bytes: AtomicU64,
    /// Whether a checkpoint has been forced
    forced: AtomicBool,
    /// Time of last checkpoint
    last_checkpoint: RwLock<Instant>,
}

impl TimeBudgetTrigger {
    /// Create a new time budget trigger
    ///
    /// # Arguments
    /// * `total_budget` - Total time available for this execution
    /// * `reserve_time` - Time to reserve for checkpointing before deadline
    pub fn new(total_budget: Duration, reserve_time: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            total_budget,
            reserve_time,
            operations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            forced: AtomicBool::new(false),
            last_checkpoint: RwLock::new(Instant::now()),
        }
    }

    /// Create a trigger for AWS Lambda with the given timeout
    pub fn for_lambda(timeout_seconds: u64, reserve_seconds: u64) -> Self {
        Self::new(
            Duration::from_secs(timeout_seconds),
            Duration::from_secs(reserve_seconds),
        )
    }

    /// Get remaining time budget
    pub fn remaining_time(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        self.total_budget.saturating_sub(elapsed)
    }

    /// Check if we're in the critical zone (should checkpoint immediately)
    pub fn is_critical(&self) -> bool {
        self.remaining_time() <= self.reserve_time
    }
}

impl CheckpointTrigger for TimeBudgetTrigger {
    fn record_operation(&self, _op_type: OperationType, bytes_processed: u64) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes_processed, Ordering::Relaxed);
    }

    fn should_checkpoint(&self) -> bool {
        self.forced.load(Ordering::Relaxed) || self.is_critical()
    }

    fn force_checkpoint(&self) {
        self.forced.store(true, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.operations.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.forced.store(false, Ordering::Relaxed);
        if let Ok(mut last) = self.last_checkpoint.write() {
            *last = Instant::now();
        }
    }

    fn urgency(&self) -> CheckpointUrgency {
        if self.forced.load(Ordering::Relaxed) {
            return CheckpointUrgency::Critical;
        }

        let remaining = self.remaining_time();
        let reserve = self.reserve_time;

        if remaining <= reserve {
            CheckpointUrgency::Critical
        } else if remaining <= reserve * 2 {
            CheckpointUrgency::Medium
        } else if remaining <= reserve * 3 {
            CheckpointUrgency::Low
        } else {
            CheckpointUrgency::None
        }
    }

    fn get_context(&self) -> CheckpointContext {
        let time_since = self
            .last_checkpoint
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        CheckpointContext {
            operations_since_checkpoint: self.operations.load(Ordering::Relaxed),
            bytes_since_checkpoint: self.bytes.load(Ordering::Relaxed),
            time_since_checkpoint: time_since,
            remaining_time_budget: Some(self.remaining_time()),
            memory_usage: None,
            memory_limit: None,
            last_checkpoint_id: None,
        }
    }
}

/// Interval-based trigger.
///
/// Triggers a checkpoint at regular time intervals.
pub struct IntervalTrigger {
    /// Interval between checkpoints
    interval: Duration,
    /// Operations since last checkpoint
    operations: AtomicU64,
    /// Bytes processed since last checkpoint
    bytes: AtomicU64,
    /// Whether a checkpoint has been forced
    forced: AtomicBool,
    /// Time of last checkpoint
    last_checkpoint: RwLock<Instant>,
}

impl IntervalTrigger {
    /// Create a new interval trigger
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            operations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            forced: AtomicBool::new(false),
            last_checkpoint: RwLock::new(Instant::now()),
        }
    }

    /// Create an interval trigger with minutes
    pub fn minutes(minutes: u64) -> Self {
        Self::new(Duration::from_secs(minutes * 60))
    }
}

impl CheckpointTrigger for IntervalTrigger {
    fn record_operation(&self, _op_type: OperationType, bytes_processed: u64) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes_processed, Ordering::Relaxed);
    }

    fn should_checkpoint(&self) -> bool {
        if self.forced.load(Ordering::Relaxed) {
            return true;
        }

        self.last_checkpoint
            .read()
            .map(|t| t.elapsed() >= self.interval)
            .unwrap_or(true)
    }

    fn force_checkpoint(&self) {
        self.forced.store(true, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.operations.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.forced.store(false, Ordering::Relaxed);
        if let Ok(mut last) = self.last_checkpoint.write() {
            *last = Instant::now();
        }
    }

    fn urgency(&self) -> CheckpointUrgency {
        if self.forced.load(Ordering::Relaxed) {
            return CheckpointUrgency::Critical;
        }

        let elapsed = self
            .last_checkpoint
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        let ratio = elapsed.as_secs_f64() / self.interval.as_secs_f64();

        if ratio >= 1.0 {
            CheckpointUrgency::Medium
        } else if ratio >= 0.8 {
            CheckpointUrgency::Low
        } else {
            CheckpointUrgency::None
        }
    }

    fn get_context(&self) -> CheckpointContext {
        let time_since = self
            .last_checkpoint
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        CheckpointContext {
            operations_since_checkpoint: self.operations.load(Ordering::Relaxed),
            bytes_since_checkpoint: self.bytes.load(Ordering::Relaxed),
            time_since_checkpoint: time_since,
            remaining_time_budget: None,
            memory_usage: None,
            memory_limit: None,
            last_checkpoint_id: None,
        }
    }
}

/// Composite trigger that combines multiple triggers.
///
/// Triggers when ANY of the component triggers fires (OR logic).
pub struct CompositeTrigger {
    triggers: Vec<Box<dyn CheckpointTrigger>>,
}

impl CompositeTrigger {
    /// Create a new composite trigger
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
        }
    }

    /// Add a trigger
    pub fn add<T: CheckpointTrigger + 'static>(mut self, trigger: T) -> Self {
        self.triggers.push(Box::new(trigger));
        self
    }

    /// Create a composite trigger from configuration
    pub fn from_config(config: &TriggerConfig) -> Self {
        let mut composite = Self::new();

        if let Some(ops) = config.operation_threshold {
            composite = composite.add(OperationCountTrigger::new(ops));
        }

        if let Some(interval) = config.interval {
            composite = composite.add(IntervalTrigger::new(interval));
        }

        if let (Some(threshold), Some(total)) =
            (config.time_budget_threshold, config.total_time_budget)
        {
            composite = composite.add(TimeBudgetTrigger::new(total, threshold));
        }

        composite
    }
}

impl Default for CompositeTrigger {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointTrigger for CompositeTrigger {
    fn record_operation(&self, op_type: OperationType, bytes_processed: u64) {
        for trigger in &self.triggers {
            trigger.record_operation(op_type, bytes_processed);
        }
    }

    fn should_checkpoint(&self) -> bool {
        self.triggers.iter().any(|t| t.should_checkpoint())
    }

    fn force_checkpoint(&self) {
        for trigger in &self.triggers {
            trigger.force_checkpoint();
        }
    }

    fn reset(&self) {
        for trigger in &self.triggers {
            trigger.reset();
        }
    }

    fn urgency(&self) -> CheckpointUrgency {
        // Return the highest urgency among all triggers
        self.triggers
            .iter()
            .map(|t| t.urgency())
            .max()
            .unwrap_or(CheckpointUrgency::None)
    }

    fn get_context(&self) -> CheckpointContext {
        // Aggregate context from all triggers
        let mut context = CheckpointContext::default();

        for trigger in &self.triggers {
            let t_ctx = trigger.get_context();
            context.operations_since_checkpoint += t_ctx.operations_since_checkpoint;
            context.bytes_since_checkpoint += t_ctx.bytes_since_checkpoint;

            // Take the maximum time since checkpoint
            if t_ctx.time_since_checkpoint > context.time_since_checkpoint {
                context.time_since_checkpoint = t_ctx.time_since_checkpoint;
            }

            // Take the minimum remaining time budget (most urgent)
            if let Some(remaining) = t_ctx.remaining_time_budget {
                context.remaining_time_budget = Some(
                    context
                        .remaining_time_budget
                        .map(|r| r.min(remaining))
                        .unwrap_or(remaining),
                );
            }
        }

        context
    }
}
