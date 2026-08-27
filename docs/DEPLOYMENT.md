# Deployment

## Local Streamlit

```powershell
.venv\Scripts\Activate.ps1
streamlit run dashboard\app.py
```

The original `dashboard/executive_inventory_dashboard.py` remains available as
a fallback while the five-layer application is being reviewed.

## Docker

Docker Desktop must be running before these commands are used.

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Open `http://localhost:8501`. The container exposes Streamlit's health endpoint
at `http://localhost:8501/_stcore/health`.

Stop the application with:

```powershell
docker compose down
```

The Compose configuration mounts generated decision outputs as read-only data,
so the container cannot overwrite pipeline results.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It:

1. Installs the pinned Python environment.
2. Compiles source files.
3. Runs the regression suite.
4. validates the configuration-driven pipeline graph.
5. Builds the Docker image.

The workflow does not execute the full 1.46-million-row pipeline because those
assets are intentionally generated outside Git. Full pipeline execution remains
an integration or scheduled workload.
