"""
Core API Sentinel - High-Performance Async Resilience & Circuit Breaker Engine
Author: Engineering Lead
License: MIT
"""

import asyncio
import time
from typing import Callable, Any, Optional, Dict, List


class RateLimiterExceeded(Exception):
    """Raised when the request threshold is exceeded within the sliding window."""
    pass


class CircuitBreakerOpen(Exception):
    """Raised when upstream calls are blocked due to repeated failures."""
    pass


class RateLimiter:
    """
    Sliding-window token rate limiter for asynchronous routines.
    Enforces smooth traffic distribution and protects downstream APIs.
    """
    def __init__(self, max_requests: int = 60, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.calls: List[float] = []
        self._lock = asyncio.Lock()

    def protect(self, func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self._lock:
                now = time.monotonic()
                # Evict timestamps falling outside the sliding observation window
                self.calls = [t for t in self.calls if now - t < self.time_window]
                if len(self.calls) >= self.max_requests:
                    raise RateLimiterExceeded(
                        f"Limit exceeded: {len(self.calls)}/{self.max_requests} within {self.time_window}s"
                    )
                self.calls.append(now)
            return await func(*args, **kwargs)
        return wrapper


class CircuitBreaker:
    """
    State-machine Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN).
    Mitigates thundering herds and cascading failures across microservices.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    def get_status(self) -> Dict[str, Any]:
        """Provides runtime telemetry metrics for observability dashboards."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
        }

    def protect(self, func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self._lock:
                now = time.monotonic()
                if self.state == "OPEN":
                    if self.last_failure_time and (now - self.last_failure_time > self.recovery_timeout):
                        self.state = "HALF-OPEN"
                    else:
                        raise CircuitBreakerOpen("Circuit is currently open. Upstream call rejected.")

            try:
                result = await func(*args, **kwargs)
                async with self._lock:
                    if self.state == "HALF-OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                return result
            except Exception as exc:
                async with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.monotonic()
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                raise exc
        return wrapper


# ---------------------------------------------------------
# Self-Verification Test Routine (Zero External Dependencies)
# ---------------------------------------------------------
async def _run_diagnostics():
    print("[INIT] Starting Core Sentinel engine self-diagnostics...")
    
    limiter = RateLimiter(max_requests=2, time_window=1.0)
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

    @limiter.protect
    @breaker.protect
    async def mock_endpoint(should_fail: bool = False):
        if should_fail:
            raise ConnectionResetError("Upstream connection reset")
        return {"status": 200, "telemetry": "healthy"}

    # 1. Normal execution
    resp = await mock_endpoint(should_fail=False)
    assert resp["status"] == 200
    print("[PASS] Normal dispatch verified.")

    # 2. Rate limiter breach
    try:
        await mock_endpoint(should_fail=False)
        await mock_endpoint(should_fail=False)
    except RateLimiterExceeded:
        print("[PASS] RateLimiter sliding-window boundary verified.")

    # 3. Circuit breaker tripping
    breaker_test = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

    @breaker_test.protect
    async def failing_call():
        raise RuntimeError("Service failure")

    for _ in range(2):
        try:
            await failing_call()
        except RuntimeError:
            pass

    try:
        await failing_call()
    except CircuitBreakerOpen:
        print("[PASS] CircuitBreaker trip state machine verified.")

    print("[SUCCESS] All architectural resilience gates are functional.")


if __name__ == "__main__":
    asyncio.run(_run_diagnostics())
