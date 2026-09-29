"""Stage 09: shared-engine HTTP serving. Nothing starts on import."""


def create_app(model=None, max_pending=64, engine=None):
    """FastAPI app; optional engine injection permits deterministic lifecycle tests.

    Required routes: GET /health and POST /generate. See the complete JSON and
    validation contract in the stage 09 chapter. Expose app.state.engine.
    """
    raise NotImplementedError("Stage 09: one engine, bounded admissions, completion and cleanup")
