import { useState } from "react";
import { buildMap, getMap, API_BASE } from "../api";
import Viewer2D from "./Viewer2D";

export default function MapView() {
  const [file, setFile] = useState(null);
  const [grid, setGrid] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const onBuild = async (e) => {
    e.preventDefault();
    if (!file) {
      setErr("Select a .npz scan first.");
      return;
    }
    setErr("");
    setLoading(true);
    try {
      const res = await buildMap(file);
      setGrid(res.grid);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setErr("");
    try {
      const res = await getMap();
      setGrid(res.grid);
    } catch (e) {
      setErr(e.message);
    }
  };

  const occupiedCount =
    grid?.reduce(
      (sum, row) => sum + row.reduce((s, v) => s + (v === 1 ? 1 : 0), 0),
      0
    ) ?? 0;

  return (
    <div>
      <h2>2. Indoor Map (Occupancy Grid)</h2>
      <p>
        Build a 2D occupancy grid from one LiDAR scan. White = free, Dark =
        obstacle.
      </p>
      <p style={{ marginBottom: "0.5rem" }}>
        Don&apos;t have a file?{" "}
        <a href={`${API_BASE}/sample_npz`} target="_blank" rel="noreferrer">
          Download sample .npz
        </a>{" "}
        from the server and upload it here.
      </p>
      <form
        onSubmit={onBuild}
        style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
      >
        <input
          type="file"
          accept=".npz"
          onChange={(e) => setFile(e.target.files[0] || null)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Building..." : "Build Map"}
        </button>
        <button type="button" onClick={onRefresh}>
          Refresh Map
        </button>
      </form>
      {err && <p style={{ color: "red" }}>{err}</p>}
      {grid && (
        <div style={{ marginTop: "1rem" }}>
          <p style={{ fontSize: "0.9rem", color: "#555" }}>
            Grid size: {grid.length} × {grid[0].length} &nbsp;|&nbsp; Occupied
            cells: {occupiedCount}
          </p>
          <div style={{ width: "100%", maxWidth: 500 }}>
            <Viewer2D grid={grid} />
          </div>
        </div>
      )}
    </div>
  );
}
