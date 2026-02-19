"""
FastAPI server for SigmaHand trajectory visualization.

Run with:
    uvicorn carbonsix_drake_sim.visualization.server:app --reload --port 8000

Or directly:
    python -m carbonsix_drake_sim.visualization.server
"""

from typing import List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydrake.all import Meshcat
import uvicorn

from .sigma_hand_visualizer import SigmaHandVisualizer


# ============================================================================
# Pydantic Models
# ============================================================================


class HandConfig(BaseModel):
    """Configuration for a single hand visualization."""

    color: Union[List[float], str] = Field(
        ..., description="RGBA color [r, g, b, a] or 'real' for textured model"
    )


class ConfigRequest(BaseModel):
    """Request to initialize/reset the visualizer with a new config."""

    config: dict[str, HandConfig] = Field(
        ..., description="Mapping of hand names to their display config"
    )
    meshcat_port: int = Field(default=7500, description="Meshcat server port")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "config": {
                        "tracking": {"color": [0.8, 0.8, 0.0, 0.4]},
                        "actual": {"color": "real"},
                    },
                    "meshcat_port": 7500,
                }
            ]
        }
    }


class TrajectoryRequest(BaseModel):
    """Request to add a trajectory for a configured hand."""

    name: str = Field(..., description="Name of the hand (must match config key)")
    time: List[float] = Field(..., description="Array of timestamps")
    ee_pose: List[List[float]] = Field(
        ..., description="Array of poses, each [x, y, z, qw, qx, qy, qz]"
    )
    hand: List[float] = Field(
        ..., description="Array of hand positions [0=open, 1=closed]"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "tracking",
                    "time": [0.0, 0.5, 1.0],
                    "ee_pose": [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    "hand": [0.0, 0.5, 1.0],
                }
            ]
        }
    }


class WaypointConfigRequest(BaseModel):
    """Request to generate waypoint config with colormap."""

    num_waypoints: int = Field(..., description="Number of waypoints")
    cmap_name: str = Field(default="turbo", description="Matplotlib colormap name")
    alpha: float = Field(default=0.3, description="Transparency (0-1)")
    prefix: str = Field(default="waypoint", description="Prefix for waypoint names")
    include_animated: bool = Field(
        default=True, description="Include animated robot following trajectory"
    )
    animated_name: str = Field(
        default="animated", description="Name for animated robot"
    )


class WaypointsRequest(BaseModel):
    """Request to add static waypoint visualizations."""

    ee_poses: List[List[float]] = Field(
        ..., description="Array of waypoint poses, each [x, y, z, qw, qx, qy, qz]"
    )
    hands: List[float] = Field(
        ..., description="Hand position at each waypoint [0=open, 1=closed]"
    )
    times: List[float] = Field(..., description="Timestamps for trajectory duration")
    prefix: str = Field(default="waypoint", description="Prefix for waypoint names")
    animated_name: str = Field(
        default="animated", description="Name for animated robot"
    )
    line_color: Optional[List[float]] = Field(
        default=None, description="RGBA color for path line"
    )
    line_width: float = Field(default=4.0, description="Width of path line")


class RecordRequest(BaseModel):
    """Request to record and publish trajectory."""

    dt: float = Field(default=0.02, description="Time step between frames")
    save_html: Optional[str] = Field(
        default=None, description="Path to save static HTML"
    )
    loop: bool = Field(default=True, description="Loop animation continuously")


class TimeQueryRequest(BaseModel):
    """Request to query or set animation at a specific time."""

    t: float = Field(..., description="Time in seconds")


class PlayRequest(BaseModel):
    """Request to play the animation."""

    realtime_rate: float = Field(
        default=1.0, description="Playback speed (1.0 = real-time)"
    )
    dt: float = Field(default=0.02, description="Time step between frames")


# ============================================================================
# Application State
# ============================================================================


class VisualizerState:
    """Holds the current visualizer and shared Meshcat instance."""

    def __init__(self):
        self.visualizer: Optional[SigmaHandVisualizer] = None
        self.meshcat: Optional[Meshcat] = None
        self.meshcat_port: int = 7500


state = VisualizerState()


# ============================================================================
# FastAPI App
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    from pydrake.all import Meshcat

    print("SigmaHand Visualization Server starting...")
    # Create shared Meshcat instance on startup
    state.meshcat = Meshcat(port=state.meshcat_port)
    print(f"Meshcat server running at http://localhost:{state.meshcat_port}")
    yield
    print("SigmaHand Visualization Server shutting down...")


app = FastAPI(
    title="SigmaHand Trajectory Visualizer",
    description="API for visualizing SigmaHand robot trajectories in Meshcat",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Check if the server is running and visualizer status."""
    return {
        "status": "ok",
        "visualizer_initialized": state.visualizer is not None,
        "trajectories": list(state.visualizer.trajectories.keys())
        if state.visualizer
        else [],
        "configured_hands": list(state.visualizer.config.keys())
        if state.visualizer
        else [],
    }


@app.post("/init")
async def initialize_visualizer(request: ConfigRequest):
    """
    Initialize or reset the visualizer with a new configuration.

    This must be called before adding trajectories.
    Note: meshcat_port in request is ignored; the shared Meshcat instance is used.
    """
    # Clean up old visualizer if it exists
    if state.visualizer is not None:
        state.visualizer.cleanup()

    # Convert Pydantic models to plain dicts
    config = {name: {"color": cfg.color} for name, cfg in request.config.items()}

    try:
        state.visualizer = SigmaHandVisualizer(config, meshcat=state.meshcat)
        return {
            "status": "initialized",
            "configured_hands": list(config.keys()),
            "meshcat_url": f"http://localhost:{state.meshcat_port}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/init_with_waypoints")
async def initialize_with_waypoints(request: WaypointConfigRequest):
    """
    Initialize visualizer with waypoint config using colormap colors.

    Convenience endpoint that generates waypoint config and initializes.
    """
    # Clean up old visualizer if it exists
    if state.visualizer is not None:
        state.visualizer.cleanup()

    waypoint_config = SigmaHandVisualizer.generate_waypoint_config(
        num_waypoints=request.num_waypoints,
        cmap_name=request.cmap_name,
        alpha=request.alpha,
        prefix=request.prefix,
        include_animated=request.include_animated,
        animated_name=request.animated_name,
    )

    try:
        state.visualizer = SigmaHandVisualizer(waypoint_config, meshcat=state.meshcat)
        return {
            "status": "initialized",
            "configured_hands": list(waypoint_config.keys()),
            "meshcat_url": f"http://localhost:{state.meshcat_port}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trajectory")
async def add_trajectory(request: TrajectoryRequest):
    """
    Add a trajectory for a configured hand.

    The hand name must match a key in the config used during initialization.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        state.visualizer.add_trajectory(
            name=request.name,
            time=request.time,
            ee_pose=request.ee_pose,
            hand=request.hand,
        )
        return {
            "status": "trajectory_added",
            "name": request.name,
            "num_timesteps": len(request.time),
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waypoints")
async def add_waypoints(request: WaypointsRequest):
    """
    Add static waypoint visualizations.

    Each waypoint creates a static hand at the specified pose.
    Requires waypoint config entries (use /init_with_waypoints or include in /init).
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        state.visualizer.add_waypoints(
            ee_poses=request.ee_poses,
            hands=request.hands,
            times=request.times,
            prefix=request.prefix,
            animated_name=request.animated_name,
            line_color=request.line_color,
            line_width=request.line_width,
        )
        return {
            "status": "waypoints_added",
            "num_waypoints": len(request.ee_poses),
            "prefix": request.prefix,
            "has_animated": request.animated_name in state.visualizer.config,
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/publish")
async def publish_trajectory(request: RecordRequest = RecordRequest()):
    """
    Publish the trajectory visualization to Meshcat with a time slider.

    Call this after adding all trajectories. Use /set_time or /sync to control playback.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        state.visualizer.publish_trajectory(save_html=request.save_html)
        time_range = state.visualizer.get_time_range()
        return {
            "status": "published",
            "mode": "slider",
            "html_saved": request.save_html,
            "time_range": time_range,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/record")
@app.post("/publish_recording")
async def publish_recording(request: RecordRequest = RecordRequest()):
    """
    Record and publish animation using Meshcat's built-in animation player.

    This creates an animation that can be played using Meshcat's animation controls.
    Set loop=true (default) for continuous looping.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        state.visualizer.publish_recording(
            dt=request.dt, save_html=request.save_html, loop=request.loop
        )
        time_range = state.visualizer.get_time_range()
        return {
            "status": "recorded",
            "mode": "animation",
            "loop": request.loop,
            "dt": request.dt,
            "html_saved": request.save_html,
            "time_range": time_range,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state")
async def get_state_at_time(request: TimeQueryRequest):
    """
    Query the animation state at a specific time.

    Returns the pose and hand position for each configured hand at time t.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        result = state.visualizer.get_state_at_time(request.t)
        return {
            "status": "ok",
            "t": request.t,
            "state": result,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_time")
async def set_visualization_time(request: TimeQueryRequest):
    """
    Set the live visualization to a specific time.

    Updates the Meshcat viewer to show the state at time t.
    This is a live update, not a recorded animation.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        state.visualizer.set_time(request.t)
        return {
            "status": "ok",
            "t": request.t,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/play")
async def play_animation(request: PlayRequest = PlayRequest()):
    """
    Play the animation once at the specified rate.

    This is a blocking call that plays through the entire trajectory.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        time_range = state.visualizer.get_time_range()
        state.visualizer.play_animation(
            realtime_rate=request.realtime_rate,
            dt=request.dt,
            loop=False,
        )
        return {
            "status": "complete",
            "realtime_rate": request.realtime_rate,
            "time_range": time_range,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/time_range")
async def get_time_range():
    """
    Get the valid time range for all trajectories.

    Returns the start and end time for each configured hand's trajectory.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    result = state.visualizer.get_time_range()
    return {
        "status": "ok",
        "time_ranges": result,
    }


@app.post("/sync")
async def sync_from_slider():
    """
    Sync visualization to current Meshcat slider values.

    Reads the time slider in Meshcat and updates the visualization.
    Call this to read the current time slider value and update the visualization.
    """
    if state.visualizer is None:
        raise HTTPException(
            status_code=400, detail="Visualizer not initialized. Call /init first."
        )

    try:
        t = state.visualizer.meshcat.GetSliderValue("time")
        state.visualizer._update_visualization_at_time(t)
        return {
            "status": "ok",
            "t": t,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_trajectories():
    """Clear all trajectories (but keep the visualizer config)."""
    if state.visualizer is None:
        raise HTTPException(status_code=400, detail="Visualizer not initialized.")

    state.visualizer.trajectories.clear()
    return {"status": "trajectories_cleared"}


@app.post("/reset")
async def reset_scene():
    """Reset the entire Meshcat scene (clears all visuals and trajectories)."""
    if state.visualizer is None:
        raise HTTPException(status_code=400, detail="Visualizer not initialized.")

    state.visualizer.clear_scene()
    return {"status": "scene_reset"}


@app.get("/config")
async def get_config():
    """Get the current visualizer configuration."""
    if state.visualizer is None:
        raise HTTPException(status_code=400, detail="Visualizer not initialized.")

    return {
        "config": state.visualizer.config,
        "trajectories": list(state.visualizer.trajectories.keys()),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "carbonsix_drake_sim.visualization.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
