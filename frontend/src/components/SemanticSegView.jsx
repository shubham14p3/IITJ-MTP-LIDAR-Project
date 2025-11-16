import { useState } from "react";
import { segmentPointCloud, API_BASE } from "../api";

export default function SemanticSegView() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setErr("Please select a .npz file with 'points'.");
      return;
    }
    setErr("");
    setLoading(true);
    try {
      const res = await segmentPointCloud(file);
      setResult(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>1. Semantic Segmentation</h2>
      <p>
        Upload a 3DSES .npz scan (from <code>data/raw/3dses_npz</code>) to get
        per-point labels. For speed, only a subset of points is used.
      </p>
      <p style={{ marginBottom: "0.5rem" }}>
        Need a test scan?{" "}
        <a href={`${API_BASE}/sample_npz`} target="_blank" rel="noreferrer">
          Download sample .npz
        </a>
        .
      </p>
      <form
        onSubmit={onSubmit}
        style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
      >
        <input
          type="file"
          accept=".npz"
          onChange={(e) => setFile(e.target.files[0] || null)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Running..." : "Segment"}
        </button>
      </form>
      {err && <p style={{ color: "red" }}>{err}</p>}
      {result && (
        <div style={{ marginTop: "1rem" }}>
          <p>Total points used: {result.num_points}</p>
          <p>
            First 30 labels:{" "}
            <code>{result.labels.slice(0, 30).join(", ")}</code>
          </p>
        </div>
      )}
    </div>
  );
}
