import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

function useCountUp(target, ms = 900) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (target == null) return setV(0);
    let raf;
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min((t - t0) / ms, 1);
      setV(target * (1 - Math.pow(1 - p, 4)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

const RING = 2 * Math.PI * 54;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) handleFile(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  const handleFile = (selected) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("That file isn't an image. Choose a PNG or JPG.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error();
      const data = await response.json();
      setResult(data);
      setHistory((h) => [{ file, url: preview, data }, ...h.filter((x) => x.url !== preview)].slice(0, 6));
    } catch {
      setError("Can't reach the classifier. Start the FastAPI server on port 8000, then try again.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const restore = (h) => {
    setFile(h.file);
    setPreview(h.url);
    setResult(h.data);
    setError("");
  };

  const copy = async () => {
    await navigator.clipboard?.writeText(`${result.predicted_class} (${result.confidence}%)`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const conf = useCountUp(result ? Number(result.confidence) : null);
  const ranked = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="shell">
      <header className="top">
        <div className="wordmark">
          <span className="mark" aria-hidden="true" />
          Tenfold
        </div>
        <div className="meta">
          <span className="live"><i /> Classifier online</span>
          <span>ResNet18 on CIFAR-10</span>
        </div>
      </header>

      <section className="intro">
        <h1>
          Drop in a photo.
          <br />
          Get one of ten answers.
        </h1>
        <p>
          The model recognizes airplanes, cars, birds, cats, deer, dogs, frogs,
          horses, ships and trucks, and shows how sure it is about each.
        </p>
      </section>

      <main className="stage">
        {/* Specimen plate */}
        <section className="plate-col">
          <div
            className={`plate ${dragging ? "is-drag" : ""} ${loading ? "is-scan" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <span className="crop tl" /><span className="crop tr" />
            <span className="crop bl" /><span className="crop br" />

            {!preview ? (
              <label className="drop">
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFile(e.target.files[0])}
                />
                <span className="drop-title">Drop an image here</span>
                <span className="drop-sub">or click to browse. PNG, JPG, JPEG.</span>
              </label>
            ) : (
              <>
                <img src={preview} alt="Selected upload" />
                {loading && <span className="scanline" />}
              </>
            )}
          </div>

          <div className="plate-foot">
            <span className="fname">{file ? file.name : "No image selected"}</span>
            {file && (
              <button className="link" onClick={reset} disabled={loading}>
                Remove
              </button>
            )}
          </div>

          {error && <div className="error" role="alert">{error}</div>}

          <button className="cta" disabled={!file || loading} onClick={handlePredict}>
            {loading ? "Classifying" : result ? "Classify again" : "Classify image"}
          </button>
        </section>

        {/* Verdict */}
        <section className="verdict-col" aria-live="polite">
          {!result && (
            <div className="idle">
              <p className="idle-title">
                {loading ? "Reading the image" : "Your result appears here"}
              </p>
              <p className="idle-sub">
                {loading
                  ? "The network is scoring all ten classes."
                  : "Add an image, then choose Classify image."}
              </p>
              <ul className="ghost-list">
                {CLASSES.map((c) => (
                  <li key={c}>
                    <span>{c}</span>
                    <b />
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result && (
            <div className="reveal">
              <div className="hero-row">
                <div>
                  <p className="kicker">This looks like a</p>
                  <h2 className="answer">{result.predicted_class}</h2>
                  <p className="sure">
                    Next closest: {ranked[1]?.[0]} at {ranked[1]?.[1]}%
                  </p>
                </div>
                <div className="ring" role="img" aria-label={`${result.confidence}% confidence`}>
                  <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="54" className="ring-bg" />
                    <circle
                      cx="60" cy="60" r="54" className="ring-fg"
                      strokeDasharray={RING}
                      strokeDashoffset={RING * (1 - Number(result.confidence) / 100)}
                    />
                  </svg>
                  <div className="ring-num">
                    <b>{conf.toFixed(1)}</b>
                    <span>% sure</span>
                  </div>
                </div>
              </div>

              <ol className="board">
                {ranked.map(([name, prob], i) => (
                  <li key={name} className={i === 0 ? "top-row" : ""}>
                    <span className="n">{name}</span>
                    <span className="track">
                      <span className="fill" style={{ width: `${Math.max(prob, 0.6)}%` }} />
                    </span>
                    <span className="v">{prob}%</span>
                  </li>
                ))}
              </ol>
              <button className="ghost" onClick={copy}>{copied ? "Copied" : "Copy result"}</button>
            </div>
          )}
        </section>
      </main>

      {history.length > 0 && (
        <section className="history">
          <h3>This session</h3>
          <div className="rail">
            {history.map((h) => (
              <button key={h.url} className={`chip ${h.url === preview ? "on" : ""}`} onClick={() => restore(h)}>
                <img src={h.url} alt="" />
                <span>{h.data.predicted_class}</span>
                <em>{h.data.confidence}%</em>
              </button>
            ))}
          </div>
        </section>
      )}

      <footer className="foot">
        Paste an image with Ctrl/⌘+V, or drop one on the plate. Trained on 32×32 images, so simple, centered subjects work best.
      </footer>
    </div>
  );
}

export default App;