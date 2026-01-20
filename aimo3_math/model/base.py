


class Model:
    """A dummy model."""

    def __init__(self):
        self._model = None

    def load(self):
        """Simulate model loading."""
        print("Loading model...")
        # Just return a "model" that always answers with 0
        return lambda problem: 0

    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self._model is None:
            self._model = self.load()
        return self._model(problem)
