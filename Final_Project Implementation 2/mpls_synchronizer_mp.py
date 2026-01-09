"""
Message Passing Synchronizer for MPLS Handshake
Implements distributed barriers using message passing instead of shared memory.

Key differences from shared memory version:
- Uses message passing for barrier coordination
- Coordinator (rank 0) collects ready signals and broadcasts proceed
- No locks needed (message queues handle synchronization)
- More scalable and truly distributed
"""

import time
from typing import List, Dict, Optional
from enum import Enum
from message_passing import MessagePassingManager, MessageType


class SyncPhase(Enum):
    """Synchronization phases"""
    TRAIN_COMPLETE = "train_complete"
    MODEL_PLACED = "model_placed"
    EXCHANGE_COMPLETE = "exchange_complete"
    ROUND_COMPLETE = "round_complete"


class MPLSSynchronizerMP:
    """
    Message passing synchronizer for MPLS handshake protocol.
    
    Uses coordinator-based barriers:
    - All nodes send SYNC_READY to coordinator (rank 0)
    - Coordinator waits for quorum/all nodes
    - Coordinator broadcasts SYNC_PROCEED to all nodes
    - Nodes wait for SYNC_PROCEED before proceeding
    """
    
    def __init__(
        self,
        rank: int,
        num_nodes: int,
        mp_manager: MessagePassingManager,
        logger,
        timeout: float = 60.0,
        quorum_ratio: float = 0.8,
        poll_interval: float = 0.05
    ):
        """
        Initialize message passing synchronizer.
        
        Args:
            rank: Node rank (0 to num_nodes-1)
            num_nodes: Total number of nodes
            mp_manager: MessagePassingManager instance
            logger: Logger instance
            timeout: Maximum time to wait for barrier (seconds)
            quorum_ratio: Minimum ratio of nodes needed to proceed (0.0-1.0)
            poll_interval: Time between barrier checks (seconds)
        """
        self.rank = rank
        self.num_nodes = num_nodes
        self.mp_manager = mp_manager
        self.logger = logger
        self.timeout = timeout
        self.quorum_ratio = quorum_ratio
        self.min_quorum = max(1, int(num_nodes * quorum_ratio))
        self.poll_interval = poll_interval
        self.is_coordinator = (rank == 0)
    
    def _wait_for_barrier(
        self,
        phase: SyncPhase,
        set_flag: bool = True
    ) -> bool:
        """
        Wait for barrier using message passing.
        
        Args:
            phase: Synchronization phase
            set_flag: Whether to send ready signal (usually True)
        
        Returns:
            True if barrier passed, False if timeout
        """
        barrier_name = phase.value
        start_time = time.time()
        
        if self.is_coordinator:
            # Coordinator: collect ready signals and broadcast proceed
            return self._coordinator_barrier(phase, start_time)
        else:
            # Worker: send ready and wait for proceed
            return self._worker_barrier(phase, start_time, set_flag)
    
    def _coordinator_barrier(self, phase: SyncPhase, start_time: float) -> bool:
        """Coordinator side of barrier: collect ready signals and broadcast proceed"""
        barrier_name = phase.value
        received_ready = set()
        received_ready.add(0)  # Coordinator is always ready
        
        self.logger.info(f"[SYNC] {barrier_name}: Coordinator waiting for ready signals...")
        
        # Collect ready signals from all workers
        while len(received_ready) < self.num_nodes:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                self.logger.warning(
                    f"[SYNC] {barrier_name}: Timeout after {elapsed:.2f}s. "
                    f"Received {len(received_ready)}/{self.num_nodes} ready signals. "
                    f"Proceeding with quorum ({self.min_quorum} needed)."
                )
                break
            
            # Receive ready signals
            message = self.mp_manager.receive(0, timeout=self.poll_interval)
            if message and message.type == MessageType.SYNC_READY:
                if message.metadata.get('phase') == barrier_name:
                    received_ready.add(message.sender)
                    self.logger.debug(
                        f"[SYNC] {barrier_name}: Received ready from p{message.sender} "
                        f"({len(received_ready)}/{self.num_nodes})"
                    )
            
            # Log progress periodically
            if len(received_ready) % max(1, self.num_nodes // 4) == 0:
                self.logger.info(
                    f"[SYNC] {barrier_name}: {len(received_ready)}/{self.num_nodes} nodes ready "
                    f"({elapsed:.1f}s elapsed)"
                )
        
        # Check quorum
        if len(received_ready) >= self.min_quorum:
            self.logger.info(
                f"[SYNC] {barrier_name}: Quorum reached ({len(received_ready)}/{self.num_nodes} nodes). "
                f"Broadcasting proceed..."
            )
            
            # Broadcast proceed signal
            proceed_count = self.mp_manager.broadcast(
                0, MessageType.SYNC_PROCEED,
                metadata={'phase': barrier_name, 'ready_count': len(received_ready)}
            )
            self.logger.debug(f"[SYNC] {barrier_name}: Broadcasted proceed to {proceed_count} nodes")
            return True
        else:
            self.logger.error(
                f"[SYNC] {barrier_name}: Quorum not reached ({len(received_ready)}/{self.min_quorum} needed)"
            )
            return False
    
    def _worker_barrier(self, phase: SyncPhase, start_time: float, set_flag: bool) -> bool:
        """Worker side of barrier: send ready and wait for proceed"""
        barrier_name = phase.value
        
        # Send ready signal to coordinator
        if set_flag:
            self.logger.info(f"[SYNC] {barrier_name}: Sending ready signal to coordinator...")
            self.mp_manager.send(
                self.rank, 0, MessageType.SYNC_READY,
                metadata={'phase': barrier_name}
            )
            self.logger.debug(f"[SYNC] {barrier_name}: Ready signal sent")
        
        # Wait for proceed signal from coordinator
        self.logger.info(f"[SYNC] {barrier_name}: Waiting for proceed signal from coordinator...")
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                self.logger.warning(
                    f"[SYNC] {barrier_name}: Timeout waiting for proceed signal after {elapsed:.2f}s"
                )
                return False
            
            # Receive proceed signal
            message = self.mp_manager.receive(self.rank, timeout=self.poll_interval)
            if message and message.type == MessageType.SYNC_PROCEED:
                if message.metadata.get('phase') == barrier_name:
                    ready_count = message.metadata.get('ready_count', self.num_nodes)
                    self.logger.info(
                        f"[SYNC] {barrier_name}: Received proceed signal "
                        f"({ready_count}/{self.num_nodes} nodes ready)"
                    )
                    return True
            
            # Log progress periodically
            if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:
                self.logger.debug(
                    f"[SYNC] {barrier_name}: Still waiting for proceed... ({elapsed:.1f}s elapsed)"
                )
    
    def wait_for_all_train(self, round_num: int) -> bool:
        """
        Wait for all nodes to complete training.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(SyncPhase.TRAIN_COMPLETE, set_flag=True)
    
    def wait_for_all_models_placed(self) -> bool:
        """
        Wait for all nodes to place models.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(SyncPhase.MODEL_PLACED, set_flag=True)
    
    def wait_for_all_exchanges(self) -> bool:
        """
        Wait for all nodes to complete exchanges.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(SyncPhase.EXCHANGE_COMPLETE, set_flag=True)
    
    def wait_for_next_round(self) -> bool:
        """
        Wait for all nodes to be ready for next round.
        Returns True if barrier passed, False if timeout.
        """
        return self._wait_for_barrier(SyncPhase.ROUND_COMPLETE, set_flag=True)
    
    def signal_exchange_complete(self):
        """Signal that this node has completed its exchange"""
        # In message passing, this is handled by wait_for_all_exchanges
        # which sends ready signal. This method is kept for API compatibility.
        self.logger.debug(f"[SYNC] Node {self.rank} exchange complete (signaled via barrier)")
    
    def get_ready_count(self, phase: SyncPhase) -> int:
        """
        Get number of ready nodes for a phase.
        In message passing, coordinator tracks this, workers don't know.
        Returns estimated count based on last proceed message.
        """
        # For workers, we don't have this information directly
        # Could be enhanced to include in proceed message metadata
        if self.is_coordinator:
            # Coordinator could track this, but for simplicity return num_nodes
            return self.num_nodes
        else:
            # Workers don't know, return estimated value
            return self.num_nodes


def create_synchronizer_mp(
    rank: int,
    num_nodes: int,
    mp_manager: MessagePassingManager,
    logger,
    timeout: float = 60.0,
    quorum_ratio: float = 0.8
) -> MPLSSynchronizerMP:
    """
    Factory function to create a message passing synchronizer instance.
    
    Args:
        rank: Node rank
        num_nodes: Total number of nodes
        mp_manager: MessagePassingManager instance
        logger: Logger instance
        timeout: Barrier timeout in seconds
        quorum_ratio: Minimum ratio of nodes needed (0.0-1.0)
    
    Returns:
        MPLSSynchronizerMP instance
    """
    return MPLSSynchronizerMP(
        rank=rank,
        num_nodes=num_nodes,
        mp_manager=mp_manager,
        logger=logger,
        timeout=timeout,
        quorum_ratio=quorum_ratio
    )

