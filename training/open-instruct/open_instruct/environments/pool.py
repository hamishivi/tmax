"""Pool of Ray actors with acquire/release semantics."""

import asyncio
import contextlib
import os
import random
import time
from typing import Any

import ray

from open_instruct import logger_utils
from open_instruct.environments.backends import is_docker_host_connectivity_error

logger = logger_utils.setup_logger(__name__)

DEFAULT_ACQUIRE_TIMEOUT_S = float(os.getenv("SWERL_POOL_ACQUIRE_TIMEOUT_S", "86400"))
ACQUIRE_CONCURRENCY = 1000
RELEASE_CONCURRENCY = 128
DEFAULT_PODMAN_HOST_COOLDOWN_S = 300.0
DEFAULT_PODMAN_HOST_COOLDOWN_JITTER_S = 30.0
DEFAULT_ENV_ACTOR_CREATE_BATCH_SIZE = 64
DEFAULT_ENV_ACTOR_CREATE_BATCH_SLEEP_S = 0.0
DEFAULT_ENV_ACTOR_SETUP_MAX_RETRIES = 5


def _podman_docker_hosts_from_env() -> list[str]:
    hosts = os.getenv("SWERL_PODMAN_DOCKER_HOSTS", "")
    return [host.strip() for host in hosts.split(",") if host.strip()]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, value, default)
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _actor_key(actor: ray.actor.ActorHandle) -> str:
    actor_id = getattr(actor, "_actor_id", None)
    if actor_id is not None:
        hex_fn = getattr(actor_id, "hex", None)
        if callable(hex_fn):
            return hex_fn()
        return str(actor_id)
    return str(actor)


def _is_podman_host_failure(error: BaseException) -> bool:
    return is_docker_host_connectivity_error(error)


def _actor_reusable_after_error(error: BaseException) -> bool:
    return not isinstance(error, ray.exceptions.RayActorError)


@ray.remote(concurrency_groups={"acquire": ACQUIRE_CONCURRENCY, "release": RELEASE_CONCURRENCY})
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is an async Ray actor. acquire() blocks until an actor is available
    (no polling needed — release() wakes up waiting acquirers via asyncio.Queue).
    """

    def __init__(
        self,
        pool_size: int,
        actor_class: type,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_S,
        **actor_kwargs: Any,
    ):
        self._acquire_timeout = acquire_timeout
        self._remote_class = ray.remote(actor_class)
        self._actor_kwargs = dict(actor_kwargs)
        docker_hosts = _podman_docker_hosts_from_env()
        self._docker_hosts = (
            docker_hosts if actor_kwargs.get("backend") == "docker" and "docker_host" not in actor_kwargs else []
        )
        self._host_cursor = 0
        self._host_inflight = {host: 0 for host in self._docker_hosts}
        self._host_unhealthy_until: dict[str, float] = {}
        self._actor_host_leases: dict[str, str] = {}
        self._host_cooldown_s = _env_float("SWERL_PODMAN_HOST_COOLDOWN_S", DEFAULT_PODMAN_HOST_COOLDOWN_S)
        self._host_cooldown_jitter_s = _env_float(
            "SWERL_PODMAN_HOST_COOLDOWN_JITTER_S", DEFAULT_PODMAN_HOST_COOLDOWN_JITTER_S
        )
        self._close_on_release = _env_flag("SWERL_ENV_CLOSE_ON_RELEASE", False)

        create_batch_size = max(1, _env_int("SWERL_ENV_ACTOR_CREATE_BATCH_SIZE", DEFAULT_ENV_ACTOR_CREATE_BATCH_SIZE))
        create_batch_sleep_s = max(
            0.0, _env_float("SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S", DEFAULT_ENV_ACTOR_CREATE_BATCH_SLEEP_S)
        )
        setup_max_retries = max(1, _env_int("SWERL_ENV_ACTOR_SETUP_MAX_RETRIES", DEFAULT_ENV_ACTOR_SETUP_MAX_RETRIES))
        logger.info(
            "Creating pool of %s %s actors (batch_size=%s, batch_sleep_s=%s, setup_max_retries=%s)",
            pool_size,
            actor_class.__name__,
            create_batch_size,
            create_batch_sleep_s,
            setup_max_retries,
        )
        if self._docker_hosts:
            logger.info(
                "Balancing %s %s actors across %s Podman Docker hosts at reset time",
                pool_size,
                actor_class.__name__,
                len(self._docker_hosts),
            )
        if self._close_on_release:
            logger.info("EnvironmentPool will close actor backends on release to free idle sandbox resources.")
        self._actors = []
        for batch_start in range(0, pool_size, create_batch_size):
            batch_end = min(pool_size, batch_start + create_batch_size)
            self._actors.extend(self._create_actor_batch(batch_end - batch_start, setup_max_retries))
            logger.info("Created %s/%s %s actors", len(self._actors), pool_size, actor_class.__name__)
            if create_batch_sleep_s > 0 and batch_end < pool_size:
                time.sleep(create_batch_sleep_s)

        self._available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)
        logger.info("Pool ready: %s %s actors", pool_size, actor_class.__name__)

    def _create_actor_batch(self, count: int, max_retries: int) -> list[ray.actor.ActorHandle]:
        actors: list[ray.actor.ActorHandle] = []
        last_error: BaseException | None = None
        attempt = 0
        while len(actors) < count and attempt < max_retries:
            attempt += 1
            needed = count - len(actors)
            candidates = [self._remote_class.remote(**self._actor_kwargs) for _ in range(needed)]
            setup_refs = [actor.setup.remote() for actor in candidates]
            failed = 0
            for actor, setup_ref in zip(candidates, setup_refs):
                try:
                    ray.get(setup_ref)
                    actors.append(actor)
                except Exception as e:
                    failed += 1
                    last_error = e
                    with contextlib.suppress(Exception):
                        ray.kill(actor, no_restart=True)
            if failed:
                logger.warning(
                    "Environment actor setup failed during pool creation (attempt %s/%s, failed=%s, ready=%s/%s): %s",
                    attempt,
                    max_retries,
                    failed,
                    len(actors),
                    count,
                    last_error,
                )
                time.sleep(min(5.0, 0.5 * attempt))
        if len(actors) != count:
            for actor in actors:
                with contextlib.suppress(Exception):
                    ray.kill(actor, no_restart=True)
            raise RuntimeError(
                f"Failed to create {count} environment actors after {max_retries} attempts; "
                f"created {len(actors)}: {last_error}"
            ) from last_error
        return actors

    async def _acquire_actor(self) -> ray.actor.ActorHandle:
        try:
            return await asyncio.wait_for(self._available.get(), timeout=self._acquire_timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Pool acquire timed out after {self._acquire_timeout}s. "
                f"Pool has {len(self._actors)} actors, {self._available.qsize()} available. "
                f"An actor may have crashed without being released."
            ) from e

    @ray.method(concurrency_group="acquire")
    async def acquire(self) -> ray.actor.ActorHandle:
        return await self._acquire_actor()

    @ray.method(concurrency_group="acquire")
    async def acquire_reset(self, reset_kwargs: dict[str, Any]) -> tuple[ray.actor.ActorHandle, list[dict]]:
        """Acquire an actor and reset it, rotating Podman hosts on host-level failures."""
        actor = await self._acquire_actor()
        try:
            target_tools = await self._reset_actor(actor, reset_kwargs)
        except Exception as e:
            if _actor_reusable_after_error(e):
                await self._release_actor(actor)
            else:
                # a crashed actor previously
                # stayed in self._actors but never re-entered the available queue —
                # a silent capacity leak. Discard it (kill + deregister) and spawn a
                # replacement off-loop so pool capacity stays stable across failures.
                await self._discard_actor(actor, reason=f"reset failure: {e}")
            raise
        return actor, target_tools

    @ray.method(concurrency_group="release")
    async def release(self, actor: ray.actor.ActorHandle) -> None:
        await self._release_actor(actor)

    @ray.method(concurrency_group="release")
    async def discard(self, actor: ray.actor.ActorHandle, reason: str = "") -> None:
        await self._discard_actor(actor, reason)

    def size(self) -> int:
        return len(self._actors)

    async def _release_actor(self, actor: ray.actor.ActorHandle) -> None:
        actor_key = _actor_key(actor)
        host = self._actor_host_leases.pop(actor_key, None)
        if host is not None:
            self._release_host(host)
        if self._close_on_release:
            try:
                await actor.close.remote()
            except Exception as e:
                await self._discard_actor(actor, reason=f"close-on-release failure: {e}")
                return
        await self._available.put(actor)

    async def _discard_actor(self, actor: ray.actor.ActorHandle, reason: str = "") -> None:
        actor_key = _actor_key(actor)
        if not any(_actor_key(candidate) == actor_key for candidate in self._actors):
            return
        host = self._actor_host_leases.pop(actor_key, None)
        if host is not None:
            self._release_host(host)

        self._actors = [candidate for candidate in self._actors if _actor_key(candidate) != actor_key]
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)

        logger.warning(
            "Discarded environment actor %s. reason=%s pool_size=%s available=%s",
            actor_key,
            reason,
            len(self._actors),
            self._available.qsize(),
        )
        # Every discard path must restore capacity, not only failed resets.
        # Remove the old actor before yielding so duplicate discards cannot
        # create multiple replacements. Setup failures remain visible.
        loop = asyncio.get_running_loop()
        replacement = await loop.run_in_executor(None, self._create_actor_batch, 1, 2)
        for new_actor in replacement:
            self._actors.append(new_actor)
            self._available.put_nowait(new_actor)

    async def _reset_actor(self, actor: ray.actor.ActorHandle, reset_kwargs: dict[str, Any]) -> list[dict]:
        if not self._docker_hosts:
            _, target_tools = await actor.reset.remote(**reset_kwargs)
            return target_tools

        attempted_hosts: set[str] = set()
        last_error: BaseException | None = None
        while len(attempted_hosts) < len(self._docker_hosts):
            host = self._lease_next_host(attempted_hosts)
            if host is None:
                break
            try:
                kwargs = dict(reset_kwargs)
                kwargs["docker_host"] = host
                _, target_tools = await actor.reset.remote(**kwargs)
                self._actor_host_leases[_actor_key(actor)] = host
                return target_tools
            except Exception as e:
                last_error = e
                self._release_host(host)
                attempted_hosts.add(host)
                if not _is_podman_host_failure(e):
                    raise
                self._mark_host_unhealthy(host, e)
                logger.warning(
                    "Reset failed on Podman Docker host %s; trying another host if available (%s/%s): %s",
                    host,
                    len(attempted_hosts),
                    len(self._docker_hosts),
                    e,
                )

        if last_error is not None:
            raise RuntimeError(
                f"Reset failed after trying {len(attempted_hosts)} Podman Docker hosts: {last_error}"
            ) from last_error
        raise RuntimeError("Reset failed before trying any Podman Docker hosts.")

    def _lease_next_host(self, exclude: set[str]) -> str | None:
        candidates = [host for host in self._healthy_hosts() if host not in exclude]
        if not candidates:
            candidates = [host for host in self._docker_hosts if host not in exclude]
        if not candidates:
            return None

        min_inflight = min(self._host_inflight.get(host, 0) for host in candidates)
        least_loaded = {host for host in candidates if self._host_inflight.get(host, 0) == min_inflight}
        ordered_hosts = self._docker_hosts[self._host_cursor :] + self._docker_hosts[: self._host_cursor]
        for host in ordered_hosts:
            if host in least_loaded:
                self._host_cursor = (self._docker_hosts.index(host) + 1) % len(self._docker_hosts)
                self._host_inflight[host] = self._host_inflight.get(host, 0) + 1
                return host
        return None

    def _healthy_hosts(self) -> list[str]:
        now = time.monotonic()
        return [host for host in self._docker_hosts if self._host_unhealthy_until.get(host, 0.0) <= now]

    def _release_host(self, host: str) -> None:
        self._host_inflight[host] = max(0, self._host_inflight.get(host, 0) - 1)

    def _mark_host_unhealthy(self, host: str, error: BaseException) -> None:
        cooldown = self._host_cooldown_s + random.uniform(0.0, self._host_cooldown_jitter_s)
        self._host_unhealthy_until[host] = time.monotonic() + cooldown
        logger.warning(
            "Temporarily disabling Podman Docker host %s for %.1fs after reset failure: %s", host, cooldown, error
        )
