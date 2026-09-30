import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List

from app.core.config import settings
from app.schemas.request import SystemOneRequest
from app.schemas.response import SystemOneResponse
from app.services.inference_service import inference_service

logger = logging.getLogger("jev-lite.scheduler")

@dataclass
class BatchItem:
    """ Represnts a pending request and its associated completion Future."""
    request : SystemOneRequest
    future: asyncio.Future
    created_at: float


class DynamicBatchScheduler:
    """
    Asynchronous Dynamic Batching Scheduler.
    Gathers concurrent user requests over a micro-window (10ms) and
    dispatch them in optimal batches to the GPU engine
    """
    def __init__(self):
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue(maxsize=settings.scheduler_max_queue_size)
        self.max_batch_size = settings.scheduler_max_batch_size
        self.max_delay_sec = settings.scheduler_max_delay_ms / 1000.0  # convert ms to sec
        self.worker_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self):
        """ Starts the background worker loop in FastAPI lifespan."""
        if self._is_running:
            return
        self._is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Dynamic Batch Scheduler started ( max_batch_size = {self.max_batch_size}, window = {settings.scheduler_max_delay_ms}ms)")

    async def stop(self):
        """ Gracefully drains remaining requests and shut down. """
        if not self._is_running:
            return

        logger.info("stopping Dynamci Batch scheduler...")
        self._is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("scheduler worker stopped.")

    async def submit(self, request: SystemOneRequest) -> SystemOneResponse:
        """
        Public Gateway: Pushes request to queue and awaits the future.
        called by FqastAPI HTTP endpoints.
        """

        if not self._is_running:
            raise RuntimeError("Scheduler is not running.")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchItem(request=request, future=future, created_at=time.perf_counter())

        try:
            # Pushes to queue in O(1) time (< 0.1ms)
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("scheduler queue is full! Appplying backpressure.")
            raise RuntimeError("System is under heavy load. please try again shortly (HTTP 429).")

        return await future

    async def _worker_loop(self):
        """
        the continous dual-trigger loop ( time vs size).
        """
        while self._is_running:
            # 1. wait for the very first item to arrive (sleeps with 0% CPU if idle)
            first_item = await self.queue.get()
            batch : List[BatchItem] = [first_item]
            start_time = time.perf_counter()

            while len(batch) < self.max_batch_size:
                elapsed = time.perf_counter() - start_time
                remaining_time = self.max_delay_sec - elapsed

                if remaining_time <= 0:
                    break

                try:

                    next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break # timeout  reached 

            # dispatch the colledted batch to gpu
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchItem]):
        """ execute the batch and resolve each user's Future. """
        logger.info(f"Dispatching dynamic batch of {len(batch)} requests to GPU.")

        for item in batch:
            try:
                # evaluate request using our inference service
                response = await inference_service.predict(item.request)
                if not item.future.cancelled():
                    item.future.set_result(response)
            except Exception as e:
                logger.error(f"Error evaluating request: {str(e)}")
                if not item.future.cancelled():
                    item.future.set_exception(e)


scheduler = DynamicBatchScheduler()