"""Small display history for tracked entities."""

from collections import deque


class TrackHistory:

    def __init__(
        self,
        max_points: int,
        timestamp: float,
    ):

        self.first_seen = timestamp

        self.last_seen = timestamp

        self.centers = deque(
            maxlen=max_points
        )

    def update(
        self,
        center,
        timestamp: float,
    ):

        self.centers.append(
            center
        )

        self.last_seen = timestamp


class EntityHistoryStore:

    def __init__(
        self,
        max_points: int = 24,
    ):

        self.max_points = max_points

        self._history = {}

    def update(
        self,
        entity_id: int,
        center,
        timestamp: float,
    ):

        history = self._history.get(
            entity_id
        )

        if history is None:

            history = TrackHistory(
                max_points=(
                    self.max_points
                ),
                timestamp=timestamp,
            )

            self._history[
                entity_id
            ] = history

        history.update(
            center=center,
            timestamp=timestamp,
        )

    def trail(
        self,
        entity_id: int,
    ):

        history = self._history.get(
            entity_id
        )

        if history is None:
            return ()

        return tuple(
            history.centers
        )

    def track_age(
        self,
        entity_id: int,
        timestamp: float,
    ) -> float:

        history = self._history.get(
            entity_id
        )

        if history is None:
            return 0.0

        return max(
            0.0,
            timestamp
            - history.first_seen,
        )

    def prune(
        self,
        timestamp: float,
        max_missing_age: float = 2.0,
    ):

        dead_ids = []

        for (
            entity_id,
            history,
        ) in self._history.items():

            age = (
                timestamp
                - history.last_seen
            )

            if age > max_missing_age:

                dead_ids.append(
                    entity_id
                )

        for entity_id in dead_ids:

            del self._history[
                entity_id
            ]

    def keep_only(self, active_ids):
        active = set(active_ids)
        self._history = {key: value for key, value in self._history.items() if key in active}
