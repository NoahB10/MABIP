"""
Async utility functions and task management.
"""
import asyncio
import logging
from typing import Set, Coroutine, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncTaskManager:
    """
    Manages async tasks with automatic cleanup.
    Tracks all running tasks and ensures proper cancellation on shutdown.
    """
    
    def __init__(self, name: str = "TaskManager"):
        self.name = name
        self.tasks: Set[asyncio.Task] = set()
        self._shutdown = False
    
    def add_task(self, task: asyncio.Task, name: str = None) -> asyncio.Task:
        """
        Backwards-compatible wrapper to track an already created task.
        """
        if name and hasattr(task, "set_name"):
            try:
                task.set_name(name)
            except Exception:
                pass
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return task
    
    def create_task(self, coro: Coroutine, name: str = None) -> asyncio.Task:
        """
        Create and track an async task.
        
        Args:
            coro: Coroutine to run
            name: Optional task name for debugging
            
        Returns:
            Created asyncio.Task
        """
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        logger.debug(f"[{self.name}] Created task: {name or task.get_name()}")
        return task
    
    def _task_done(self, task: asyncio.Task):
        """Callback when task completes."""
        self.tasks.discard(task)
        
        # Log any exceptions
        if not task.cancelled():
            try:
                exc = task.exception()
                if exc:
                    logger.error(
                        f"[{self.name}] Task {task.get_name()} failed: {exc}",
                        exc_info=exc
                    )
            except asyncio.CancelledError:
                pass
    
    async def shutdown(self, timeout: float = 5.0):
        """
        Cancel all running tasks and wait for completion.
        
        Args:
            timeout: Maximum time to wait for tasks to finish
        """
        if self._shutdown:
            return
        
        self._shutdown = True
        logger.info(f"[{self.name}] Shutting down {len(self.tasks)} tasks...")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks with timeout
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{self.name}] Shutdown timeout - {len(self.tasks)} tasks still running"
                )
        
        logger.info(f"[{self.name}] Shutdown complete")
    
    @property
    def task_count(self) -> int:
        """Return number of active tasks."""
        return len(self.tasks)

    def get_running_tasks(self):
        """Return a snapshot list of currently tracked tasks."""
        return list(self.tasks)

    async def cancel_all_tasks(self, timeout: float = 5.0):
        """Alias for shutdown to match GUI expectations."""
        await self.shutdown(timeout=timeout)

    async def cleanup(self, timeout: float = 5.0):
        """Alias used in tests."""
        await self.shutdown(timeout=timeout)


async def interruptible_sleep(duration: float, stop_event: asyncio.Event, 
                              check_interval: float = 0.5) -> bool:
    """
    Sleep with periodic stop checks.
    
    Args:
        duration: Total sleep duration in seconds
        stop_event: Event to check for stop request
        check_interval: How often to check stop event
        
    Returns:
        True if sleep completed, False if interrupted
    """
    elapsed = 0.0
    
    while elapsed < duration:
        # Check if stop requested
        if stop_event.is_set():
            return False
        
        # Sleep for check_interval or remaining time
        sleep_time = min(check_interval, duration - elapsed)
        await asyncio.sleep(sleep_time)
        elapsed += sleep_time
    
    return True


class AsyncRateLimiter:
    """
    Rate limiter for async operations.
    Ensures minimum time between operations.
    """
    
    def __init__(self, min_interval: float):
        """
        Args:
            min_interval: Minimum seconds between operations
        """
        self.min_interval = min_interval
        self.last_call: float = 0.0
        self._lock = asyncio.Lock()
    
    async def wait(self):
        """Wait until enough time has passed since last call."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_call
            
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            
            self.last_call = asyncio.get_event_loop().time()


def format_task_info(task: asyncio.Task) -> str:
    """Format task information for logging."""
    name = task.get_name()
    state = "running"
    
    if task.done():
        if task.cancelled():
            state = "cancelled"
        elif task.exception():
            state = f"failed: {task.exception()}"
        else:
            state = "completed"
    
    return f"{name}: {state}"


async def run_with_timeout(coro: Coroutine, timeout: float, 
                           timeout_message: str = "Operation timed out") -> Any:
    """
    Run coroutine with timeout and custom error message.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        timeout_message: Error message if timeout occurs
        
    Returns:
        Result of coroutine
        
    Raises:
        TimeoutError: If operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(timeout_message)
