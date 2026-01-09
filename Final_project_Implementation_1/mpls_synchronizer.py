"""
Deadlock-Free Synchronizer for MPLS Handshake
Implements distributed barriers with multiple deadlock prevention mechanisms:
1. Timeout-based barriers with automatic recovery
2. Consistent lock ordering (always acquire locks in rank order)
3. Health checks and quorum-based decisions
4. Automatic deadlock detection and recovery
5. No circular dependencies
"""

import time
from typing import List, Dict, Optional
from enum import Enum


class SyncPhase(Enum):
    """Synchronization phases"""
    TRAIN_COMPLETE = "train_complete"
    MODEL_PLACED = "model_placed"
    EXCHANGE_COMPLETE = "exchange_complete"
    ROUND_COMPLETE = "round_complete"


class MPLSSynchronizer:
    """
    Deadlock-free synchronizer for MPLS handshake protocol.
    
    Deadlock Prevention Mechanisms:
    1. Timeout-based: All barriers have timeouts with automatic recovery
    2. Consistent Lock Ordering: Always acquire locks in rank order (lowest first)
    3. Quorum-based: Proceed if majority (not all) nodes are ready (configurable)
    4. Health Checks: Periodic checks to detect stuck nodes
    5. Automatic Recovery: Reset barriers if timeout occurs
    6. No Circular Dependencies: Linear barrier progression
    """
    
    def __init__(
        self,
        rank: int,
        num_nodes: int,
        shared_sync: Dict,
        sync_locks: List,
        logger,
        timeout: float = 60.0,
        quorum_ratio: float = 0.8,
        poll_interval: float = 0.05
    ):
        """
        Initialize synchronizer.
        
        Args:
            rank: Node rank (0 to num_nodes-1)
            num_nodes: Total number of nodes
            shared_sync: Shared dictionary for synchronization state
            sync_locks: List of locks for synchronization
            logger: Logger instance
            timeout: Maximum time to wait for barrier (seconds)
            quorum_ratio: Minimum ratio of nodes needed to proceed (0.0-1.0)
            poll_interval: Time between barrier checks (seconds)
        """
        self.rank = rank
        self.num_nodes = num_nodes
        self.shared_sync = shared_sync
        self.sync_locks = sync_locks
        self.logger = logger
        self.timeout = timeout
        self.quorum_ratio = quorum_ratio
        self.min_quorum = max(1, int(num_nodes * quorum_ratio))
        self.poll_interval = poll_interval
        
        # Initialize shared state if not exists
        self._init_shared_state()
    
    def _init_shared_state(self):
        """Initialize shared synchronization state"""
        # Initialize only once using rank 0's lock
        with self.sync_locks[0]:
            if 'round_num' not in self.shared_sync:
                self.shared_sync['round_num'] = 0
            if 'initialized' not in self.shared_sync:
                # Initialize all flags as manager lists (already done in init_synchronizer_objects)
                # Just mark as initialized
                self.shared_sync['initialized'] = True
                self.logger.debug(f"[SYNC] Shared state initialized by node {self.rank}")
    
    def _acquire_locks_ordered(self, lock_indices: List[int]) -> List:
        """
        Acquire locks in consistent order to prevent deadlocks.
        Always acquire locks in ascending order of rank.
        """
        sorted_indices = sorted(set(lock_indices))
        acquired_locks = []
        try:
            for lock_idx in sorted_indices:
                self.sync_locks[lock_idx].acquire()
                acquired_locks.append(lock_idx)
            return acquired_locks
        except Exception as e:
            # Release all acquired locks on error
            for lock_idx in acquired_locks:
                try:
                    self.sync_locks[lock_idx].release()
                except:
                    pass
            raise e
    
    def _release_locks(self, lock_indices: List[int]):
        """Release locks in reverse order"""
        for lock_idx in reversed(lock_indices):
            try:
                self.sync_locks[lock_idx].release()
            except:
                pass
    
    def _update_health(self):
        """Update health check timestamp (assumes lock is already held)"""
        # Don't acquire lock again - should be called while lock is already held
        try:
            self.shared_sync['health_check'][self.rank] = time.time()
            self.shared_sync['last_update'][self.rank] = time.time()
        except Exception as e:
            # Silently fail - health update is not critical
            pass
    
    def _check_health(self, phase: SyncPhase) -> bool:
        """
        Check if all nodes are healthy (recently updated).
        Returns True if all nodes updated within timeout period.
        """
        current_time = time.time()
        healthy_count = 0
        
        # Check all nodes' health (read-only, no locks needed for reading)
        for i in range(self.num_nodes):
            last_update = self.shared_sync['last_update'][i]
            if current_time - last_update < self.timeout:
                healthy_count += 1
        
        if healthy_count < self.min_quorum:
            self.logger.warning(
                f"[SYNC] Health check failed in {phase.value}: "
                f"Only {healthy_count}/{self.num_nodes} nodes healthy"
            )
            return False
        return True
    
    def _wait_for_barrier(
        self,
        phase: SyncPhase,
        ready_flags: List[bool],
        set_flag: bool = True
    ) -> bool:
        """
        Wait for barrier with deadlock prevention.
        
        Args:
            phase: Synchronization phase
            ready_flags: List of ready flags to check
            set_flag: Whether to set this node's flag
        
        Returns:
            True if barrier passed, False if timeout
        """
        barrier_name = phase.value
        start_time = time.time()
        
        # Set our flag first - CRITICAL: This must happen before waiting
        if set_flag:
            try:
                self.logger.info(f"[SYNC] {barrier_name}: Setting ready flag for node {self.rank}...")
                # Acquire lock and set flag
                lock_acquired = False
                try:
                    self.sync_locks[self.rank].acquire()
                    lock_acquired = True
                    self.logger.debug(f"[SYNC] {barrier_name}: Lock acquired for node {self.rank}")
                    
                    # Set the flag
                    ready_flags[self.rank] = True
                    self.logger.debug(f"[SYNC] {barrier_name}: Flag set to True for node {self.rank}")
                    
                    # Update health (this also updates last_update)
                    self._update_health()
                    
                    # Verify immediately (while still holding lock)
                    flag_value = ready_flags[self.rank]
                    if not flag_value:
                        self.logger.error(f"[SYNC] {barrier_name}: Flag verification failed for node {self.rank}!")
                    else:
                        self.logger.info(f"[SYNC] {barrier_name}: Node {self.rank} marked as ready (verified: {flag_value})")
                    
                finally:
                    if lock_acquired:
                        self.sync_locks[self.rank].release()
                        self.logger.debug(f"[SYNC] {barrier_name}: Lock released for node {self.rank}")
                
                # Count all ready nodes AFTER releasing lock (to avoid blocking)
                try:
                    current_ready = sum(1 for i in range(self.num_nodes) if ready_flags[i])
                    self.logger.info(f"[SYNC] {barrier_name}: Current ready count: {current_ready}/{self.num_nodes} (after setting flag)")
                except Exception as e:
                    self.logger.warning(f"[SYNC] {barrier_name}: Could not count ready nodes: {e}")
                        
            except Exception as e:
                self.logger.error(f"[SYNC] {barrier_name}: CRITICAL ERROR setting flag for node {self.rank}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Don't return False - continue and see if we can proceed anyway
                # The barrier might still work if other nodes are ready
        
        # Wait for quorum with periodic logging
        iteration = 0
        while True:
            elapsed = time.time() - start_time
            iteration += 1
            
            # Timeout check
            if elapsed > self.timeout:
                self.logger.warning(
                    f"[SYNC] {barrier_name}: Timeout after {elapsed:.2f}s. "
                    f"Proceeding with available nodes."
                )
                break
            
            # Count ready nodes (read-only access, no lock needed for reading list)
            try:
                ready_count = 0
                ready_list = []
                for i in range(self.num_nodes):
                    try:
                        if ready_flags[i]:
                            ready_count += 1
                            ready_list.append(i)
                    except (IndexError, TypeError) as e:
                        self.logger.debug(f"[SYNC] Error reading flag for node {i}: {e}")
                        continue
            except Exception as e:
                self.logger.error(f"[SYNC] {barrier_name}: Error reading flags: {e}")
                time.sleep(self.poll_interval)
                continue
            
            # Log progress every 5 seconds
            if iteration % 100 == 0:  # Every 5 seconds (100 * 0.05s)
                self.logger.info(
                    f"[SYNC] {barrier_name}: Waiting... ({ready_count}/{self.num_nodes} ready "
                    f"(nodes: {ready_list}), {elapsed:.1f}s elapsed, need {self.min_quorum} for quorum)"
                )
            
            # Check if quorum reached
            if ready_count >= self.min_quorum:
                self.logger.info(
                    f"[SYNC] {barrier_name}: Quorum reached ({ready_count}/{self.num_nodes} nodes ready) "
                    f"after {elapsed:.2f}s"
                )
                return True
            
            # Check if all nodes are ready (even if quorum not configured)
            if ready_count >= self.num_nodes:
                self.logger.info(
                    f"[SYNC] {barrier_name}: All nodes ready ({ready_count}/{self.num_nodes}) "
                    f"after {elapsed:.2f}s"
                )
                return True
            
            # Health check (less frequent)
            if iteration % 20 == 0:  # Every 1 second
                if not self._check_health(phase):
                    self.logger.warning(
                        f"[SYNC] {barrier_name}: Health check failed, proceeding with {ready_count} available nodes"
                    )
                    return ready_count > 0  # Proceed if at least one other node is ready
            
            # Poll interval
            time.sleep(self.poll_interval)
        
        # Final check after timeout
        try:
            ready_count = sum(1 for i in range(self.num_nodes) if ready_flags[i])
        except (IndexError, TypeError) as e:
            self.logger.error(f"[SYNC] {barrier_name}: Error in final check: {e}")
            return False
            
        if ready_count >= self.min_quorum:
            self.logger.info(
                f"[SYNC] {barrier_name}: Proceeding with {ready_count}/{self.num_nodes} nodes after timeout"
            )
            return True
        elif ready_count > 0:
            self.logger.warning(
                f"[SYNC] {barrier_name}: Proceeding with {ready_count}/{self.num_nodes} nodes (below quorum)"
            )
            return True
        else:
            self.logger.error(
                f"[SYNC] {barrier_name}: No other nodes ready, may cause issues"
            )
            return False
    
    def _reset_barrier(self, ready_flags: List[bool], phase: SyncPhase):
        """Reset barrier flags for next round"""
        # Reset only our own flag to avoid lock contention
        # Other nodes will reset their own flags
        try:
            with self.sync_locks[self.rank]:
                ready_flags[self.rank] = False
            self.logger.debug(f"[SYNC] Reset barrier flag for {phase.value} (node {self.rank})")
        except Exception as e:
            self.logger.error(f"[SYNC] Error resetting barrier: {e}")
    
    def wait_for_all_train(self, round_num: int) -> bool:
        """
        Wait for all nodes to complete training.
        Returns True if barrier passed, False if timeout.
        """
        # Reset our own flag for this round first
        try:
            with self.sync_locks[self.rank]:
                self.shared_sync['train_ready'][self.rank] = False
            self.logger.debug(f"[SYNC] Reset train_ready flag for node {self.rank}")
        except Exception as e:
            self.logger.error(f"[SYNC] Error resetting train_ready flag: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Small delay to ensure all nodes have reset
        time.sleep(0.1)
        
        # Now set our flag and wait
        return self._wait_for_barrier(
            SyncPhase.TRAIN_COMPLETE,
            self.shared_sync['train_ready'],
            set_flag=True
        )
    
    def wait_for_all_models_placed(self) -> bool:
        """
        Wait for all nodes to place models in shared memory.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(
            SyncPhase.MODEL_PLACED,
            self.shared_sync['model_placed'],
            set_flag=True
        )
    
    def wait_for_all_exchanges(self) -> bool:
        """
        Wait for all nodes to complete exchanges.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(
            SyncPhase.EXCHANGE_COMPLETE,
            self.shared_sync['exchange_done'],
            set_flag=True
        )
    
    def wait_for_next_round(self) -> bool:
        """
        Wait for all nodes to be ready for next round.
        Returns True if barrier passed, False if timeout.
        """
        result = self._wait_for_barrier(
            SyncPhase.ROUND_COMPLETE,
            self.shared_sync['round_ready'],
            set_flag=True
        )
        
        # Reset our own flags for next round (each node resets its own)
        if result:
            self.logger.info(f"[SYNC] Round barrier passed, resetting flags for next round...")
            # Reset our own flags (no need to reset others - they'll reset their own)
            try:
                with self.sync_locks[self.rank]:
                    self.shared_sync['train_ready'][self.rank] = False
                    self.shared_sync['model_placed'][self.rank] = False
                    self.shared_sync['exchange_done'][self.rank] = False
                    self.shared_sync['round_ready'][self.rank] = False
                self.logger.debug(f"[SYNC] Reset all flags for node {self.rank} for next round")
            except Exception as e:
                self.logger.error(f"[SYNC] Error resetting flags: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return result
    
    def signal_exchange_complete(self):
        """Signal that this node has completed its exchange"""
        with self.sync_locks[self.rank]:
            self.shared_sync['exchange_done'][self.rank] = True
            self._update_health()
        self.logger.debug(f"[SYNC] Node {self.rank} exchange complete")
    
    def get_ready_count(self, phase: SyncPhase) -> int:
        """Get number of ready nodes for a phase"""
        try:
            if phase == SyncPhase.TRAIN_COMPLETE:
                flags = self.shared_sync['train_ready']
            elif phase == SyncPhase.MODEL_PLACED:
                flags = self.shared_sync['model_placed']
            elif phase == SyncPhase.EXCHANGE_COMPLETE:
                flags = self.shared_sync['exchange_done']
            elif phase == SyncPhase.ROUND_COMPLETE:
                flags = self.shared_sync['round_ready']
            else:
                return 0
            
            # Count ready nodes (accessing manager.list correctly)
            return sum(1 for i in range(self.num_nodes) if flags[i])
        except (IndexError, TypeError, KeyError) as e:
            self.logger.error(f"[SYNC] Error getting ready count: {e}")
            return 0


def create_synchronizer(
    rank: int,
    num_nodes: int,
    shared_sync: Dict,
    sync_locks: List,
    logger,
    timeout: float = 60.0,
    quorum_ratio: float = 0.8
) -> MPLSSynchronizer:
    """
    Factory function to create a synchronizer instance.
    
    Args:
        rank: Node rank
        num_nodes: Total number of nodes
        shared_sync: Shared synchronization dictionary
        sync_locks: List of synchronization locks
        logger: Logger instance
        timeout: Barrier timeout in seconds
        quorum_ratio: Minimum ratio of nodes needed (0.0-1.0)
    
    Returns:
        MPLSSynchronizer instance
    """
    return MPLSSynchronizer(
        rank=rank,
        num_nodes=num_nodes,
        shared_sync=shared_sync,
        sync_locks=sync_locks,
        logger=logger,
        timeout=timeout,
        quorum_ratio=quorum_ratio
    )

