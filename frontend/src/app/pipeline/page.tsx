"use client";

import { useState, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Cpu, Box, Leaf, FileText, CheckCircle,
  Loader2, ChevronRight, Download, ArrowLeft, Printer, Wand2
} from "lucide-react";

const GLBViewer = dynamic(() => import("@/components/GLBViewer"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-80 rounded-lg border border-teal-700 bg-[#0a1628] flex items-center justify-center">
      <Loader2 className="w-8 h-8 animate-spin text-teal-400" />
    </div>
  ),
});

const API = "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

interface SegResult {
  image_id: string;
  original_url: string;
  overlay_url: string;
  raw_url: string;
  vegetation_pct: number;
  pavement_pct: number;
  building_pct: number;
  sky_pct: number;
  shadow_pct: number;
  sky_view_factor: number;
  ndvi: number;
  ndbi: number;
  solar_exposure: number;
  status: string;
}

interface TrellisResult {
  model_url: string;
  image_url: string;
  filename: string;
}

interface Intervention {
  title: string;
  reason: string;
  what_to_do: string;
  budget_inr: number;
  temp_reduction_c: number;
  sustainability_score: number;
  implementation_weeks: number;
  co_benefits: string[];
}

const STEPS = [
  { id: 1, icon: Upload,  label: "Upload",        desc: "Upload campus photo" },
  { id: 2, icon: Cpu,     label: "SAM2 Analysis", desc: "Surface segmentation" },
  { id: 3, icon: Box,     label: "3D Model",      desc: "TRELLIS base mesh" },
  { id: 4, icon: Leaf,    label: "Interventions", desc: "Gemini AI strategies" },
  { id: 5, icon: Wand2,   label: "Visualize",     desc: "FLUX Lightning render" },
  { id: 6, icon: FileText,label: "Report",        desc: "Full analysis" },
];

// ── Shared sub-components ─────────────────────────────────────────────────────

function StatBar({ label, value, color, textColor }: {
  label: string; value: number; color: string; textColor?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span className={`font-mono font-bold ${textColor ?? "text-white"}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 bg-teal-900 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(value, 100)}%` }}
          transition={{ duration: 0.85, ease: "easeOut" }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
    </div>
  );
}

function StepIndicator({ step, activeStep }: { step: typeof STEPS[0]; activeStep: number }) {
  const done    = activeStep > step.id;
  const current = activeStep === step.id;
  return (
    <div className="relative z-10 flex flex-col items-center gap-2">
      <div className={`w-9 h-9 rounded-full flex items-center justify-center border-2 transition-all
        ${done    ? "bg-heat-green border-heat-green text-teal-900" :
          current ? "bg-teal-800 border-heat-green text-heat-green shadow-[0_0_12px_rgba(6,214,160,0.4)]" :
                    "bg-[var(--background)] border-teal-700 text-gray-600"}`}
      >
        {done ? <CheckCircle className="w-4 h-4" /> : <step.icon className="w-4 h-4" />}
      </div>
      <span className={`text-[9px] uppercase tracking-wider font-semibold hidden md:block
        ${current ? "text-heat-green" : done ? "text-gray-400" : "text-gray-600"}`}>
        {step.label}
      </span>
    </div>
  );
}

// ── SAM2 Stats Panel (reused in step 2 and report) ────────────────────────────

function SegStats({ seg }: { seg: SegResult }) {
  return (
    <div className="w-full space-y-4">
      {/* Side-by-side images */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-[10px] font-mono text-gray-500 mb-1.5">ORIGINAL</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${API}${seg.original_url}`} alt="original"
            className="w-full rounded-lg border border-teal-800 object-cover" />
        </div>
        <div>
          <p className="text-[10px] font-mono text-gray-500 mb-1.5">SEGMENTATION OVERLAY</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${API}${seg.overlay_url}`} alt="overlay"
            className="w-full rounded-lg border border-teal-800 object-cover" />
        </div>
      </div>

      {/* Colour legend pills */}
      <div className="flex flex-wrap gap-2">
        {[
          { label: "Vegetation", color: "bg-green-500",  val: seg.vegetation_pct },
          { label: "Road",       color: "bg-slate-400",  val: seg.pavement_pct },
          { label: "Building",   color: "bg-orange-500", val: seg.building_pct },
          { label: "Sky",        color: "bg-sky-400",    val: seg.sky_pct },
          { label: "Shadow",     color: "bg-indigo-700", val: seg.shadow_pct },
        ].map(c => (
          <div key={c.label}
            className="flex items-center gap-1.5 text-xs text-gray-400 bg-teal-900/50 px-2.5 py-1.5 rounded-full border border-teal-800">
            <div className={`w-2.5 h-2.5 rounded-sm ${c.color}`} />
            {c.label}
            <span className="text-white font-bold ml-0.5">{c.val.toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* Percentage bars */}
      <div className="bg-teal-900/30 border border-teal-800 rounded-xl p-4 space-y-3">
        <StatBar label="Vegetation / Trees" value={seg.vegetation_pct} color="bg-green-500"  textColor="text-green-400" />
        <StatBar label="Road / Pavement"    value={seg.pavement_pct}   color="bg-slate-400" />
        <StatBar label="Building Facade"    value={seg.building_pct}   color="bg-orange-500" textColor="text-orange-400" />
        <StatBar label="Sky"                value={seg.sky_pct}         color="bg-sky-400"   textColor="text-sky-400" />
        <StatBar label="Shadow"             value={seg.shadow_pct}      color="bg-indigo-700" />
      </div>

      {/* Scalar metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
        {[
          { label: "Sky View Factor", value: seg.sky_view_factor.toFixed(3), note: "Heat escape index" },
          { label: "NDVI (proxy)",    value: seg.ndvi.toFixed(3),            note: "Vegetation greenness" },
          { label: "NDBI (proxy)",    value: seg.ndbi.toFixed(3),            note: "Built-up index" },
          { label: "Solar Exposure",  value: seg.solar_exposure.toFixed(3),  note: "Heat absorption" },
        ].map(s => (
          <div key={s.label} className="bg-teal-900/50 border border-teal-700 rounded-lg p-3">
            <p className="text-[10px] text-gray-500">{s.label}</p>
            <p className="text-xl font-black text-heat-green">{s.value}</p>
            <p className="text-[10px] text-gray-600 mt-0.5">{s.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Intervention Card (selectable) ────────────────────────────────────────────

function InterventionCard({
  item, index, selected, onToggle,
}: {
  item: Intervention; index: number; selected: boolean; onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`w-full text-left border rounded-xl p-4 transition-all duration-200 cursor-pointer
        ${selected
          ? "border-heat-green bg-heat-green/10 shadow-[0_0_12px_rgba(6,214,160,0.15)]"
          : "border-teal-800 bg-teal-900/30 hover:border-teal-600"}`}
    >
      <div className="flex items-start gap-3">
        {/* Checkbox */}
        <div className={`mt-0.5 w-5 h-5 rounded border-2 flex-shrink-0 flex items-center justify-center transition-all
          ${selected ? "bg-heat-green border-heat-green" : "border-teal-600"}`}>
          {selected && <CheckCircle className="w-3 h-3 text-teal-900" />}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-gray-600">0{index + 1}</span>
              <h4 className={`font-bold text-sm ${selected ? "text-white" : "text-gray-300"}`}>
                {item.title}
              </h4>
            </div>
            <div className="text-right flex-shrink-0">
              <p className="text-heat-green font-black text-lg leading-none">-{item.temp_reduction_c}°C</p>
              <p className="text-gray-400 text-xs mt-0.5">₹{item.budget_inr.toLocaleString("en-IN")}</p>
            </div>
          </div>
          <p className="text-gray-500 text-xs mt-1.5 italic leading-relaxed">{item.what_to_do}</p>

          {/* Sustainability dots */}
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[10px] text-gray-600">Sustainability</span>
            <div className="flex gap-0.5">
              {Array.from({ length: 10 }).map((_, j) => (
                <div key={j}
                  className={`w-2.5 h-1.5 rounded-sm ${j < item.sustainability_score ? "bg-heat-green" : "bg-teal-800"}`}
                />
              ))}
            </div>
            <span className="text-[10px] text-heat-green font-bold">{item.sustainability_score}/10</span>
          </div>
        </div>
      </div>
    </button>
  );
}

// ── Report Component ──────────────────────────────────────────────────────────

function Report({
  segResult, trellisResult, interventions, budget,
  selectedIdxs, interventionImage, interventionTrellis, onReset,
}: {
  segResult: SegResult;
  trellisResult: TrellisResult | null;
  interventions: Intervention[];
  budget: number;
  selectedIdxs: number[];
  interventionImage: string | null;
  interventionTrellis: TrellisResult | null;
  onReset: () => void;
}) {
  const applied   = interventions.filter((_, i) => selectedIdxs.includes(i));
  const totalDrop = applied.reduce((s, i) => s + i.temp_reduction_c, 0);
  const totalCost = applied.reduce((s, i) => s + i.budget_inr, 0);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="w-full max-w-4xl mx-auto space-y-8 print:space-y-6">

      {/* Header */}
      <div className="border border-teal-700 rounded-xl p-6 bg-gradient-to-br from-teal-900/60 to-[#0a1628]">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <p className="text-[10px] font-mono tracking-widest text-heat-green mb-1">
              HEATSCAPE — URBAN THERMAL ANALYSIS REPORT
            </p>
            <h2 className="text-2xl font-black text-white uppercase tracking-tight">
              Heat Island Mitigation Plan
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              SAM2 Segmentation · Gemini Vision · FLUX Lightning · TRELLIS 3D
            </p>
          </div>
          <button onClick={() => window.print()}
            className="flex items-center gap-2 border border-teal-600 text-teal-400 px-4 py-2 rounded text-sm font-semibold hover:bg-teal-800 transition-all">
            <Printer className="w-4 h-4" /> Export PDF
          </button>
        </div>
      </div>

      {/* 01 — Surface Analysis */}
      <div className="border border-teal-700 rounded-xl overflow-hidden">
        <div className="bg-teal-900/40 px-5 py-3 border-b border-teal-700 flex justify-between items-center">
          <h3 className="text-white font-bold uppercase tracking-wider text-sm">01 — Surface Analysis</h3>
          <a href={`${API}${segResult.overlay_url}`} download
            className="text-xs text-teal-400 hover:underline flex items-center gap-1">
            <Download className="w-3 h-3" /> Download Mask
          </a>
        </div>
        <div className="p-5">
          <SegStats seg={segResult} />
        </div>
      </div>

      {/* 02 — Before / After Visualisation */}
      {interventionImage && (
        <div className="border border-teal-700 rounded-xl overflow-hidden">
          <div className="bg-teal-900/40 px-5 py-3 border-b border-teal-700">
            <h3 className="text-white font-bold uppercase tracking-wider text-sm">
              02 — AI Visualisation (FLUX Lightning)
            </h3>
          </div>
          <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <p className="text-xs text-gray-500 font-mono">BEFORE</p>
                <a href={`${API}${segResult.original_url}`} download
                  className="text-[10px] text-teal-500 hover:underline">Download</a>
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`${API}${segResult.original_url}`} alt="before"
                className="w-full rounded-lg border border-teal-800" />
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <p className="text-xs text-heat-green font-mono">AFTER (INTERVENTIONS)</p>
                <a href={`${API}${interventionImage}`} download
                  className="text-[10px] text-heat-green hover:underline">Download</a>
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`${API}${interventionImage}`} alt="after"
                className="w-full rounded-lg border border-heat-green/50 shadow-[0_0_15px_rgba(6,214,160,0.15)]" />
            </div>
          </div>
        </div>
      )}

      {/* 03 — 3D Comparison */}
      {(trellisResult || interventionTrellis) && (
        <div className="border border-teal-700 rounded-xl overflow-hidden">
          <div className="bg-teal-900/40 px-5 py-3 border-b border-teal-700">
            <h3 className="text-white font-bold uppercase tracking-wider text-sm">
              03 — 3D Morphology (TRELLIS)
            </h3>
          </div>
          <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
            {trellisResult && (
              <div>
                <div className="flex justify-between items-center mb-2">
                  <p className="text-xs text-gray-500 font-mono">BEFORE 3D</p>
                  <a href={`${API}${trellisResult.model_url}`} download
                    className="text-[10px] text-teal-500 hover:underline">Download GLB</a>
                </div>
                <GLBViewer url={`${API}${trellisResult.model_url}`} height="h-64" />
              </div>
            )}
            {interventionTrellis && (
              <div>
                <div className="flex justify-between items-center mb-2">
                  <p className="text-xs text-heat-green font-mono">AFTER 3D</p>
                  <a href={`${API}${interventionTrellis.model_url}`} download
                    className="text-[10px] text-heat-green hover:underline">Download GLB</a>
                </div>
                <GLBViewer url={`${API}${interventionTrellis.model_url}`} height="h-64" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* 04 — Interventions breakdown */}
      <div className="border border-teal-700 rounded-xl overflow-hidden">
        <div className="bg-teal-900/40 px-5 py-3 border-b border-teal-700">
          <h3 className="text-white font-bold uppercase tracking-wider text-sm">
            04 — Applied Intervention Strategies
          </h3>
        </div>
        <div className="p-5 space-y-4">
          {applied.map((item, i) => (
            <div key={i} className="border border-teal-800 rounded-lg p-4 bg-teal-900/30">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-mono text-gray-600">0{i + 1}</span>
                    <h4 className="text-white font-bold">{item.title}</h4>
                  </div>
                  <p className="text-gray-400 text-sm">{item.reason}</p>
                  <p className="text-gray-500 text-xs mt-1 italic">{item.what_to_do}</p>
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    <span className="text-[10px] bg-teal-800/60 text-teal-300 px-2 py-0.5 rounded-full">
                      {item.implementation_weeks}w implementation
                    </span>
                    <span className="text-[10px] bg-teal-800/60 text-teal-300 px-2 py-0.5 rounded-full">
                      Sustainability {item.sustainability_score}/10
                    </span>
                    {item.co_benefits?.map(b => (
                      <span key={b} className="text-[10px] bg-teal-800/60 text-teal-300 px-2 py-0.5 rounded-full">{b}</span>
                    ))}
                  </div>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-heat-green font-black text-2xl">-{item.temp_reduction_c}°C</p>
                  <p className="text-white font-semibold">₹{item.budget_inr.toLocaleString("en-IN")}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="border border-heat-green/30 rounded-xl p-5 bg-heat-green/5">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-gray-400 text-xs">Budget Used</p>
            <p className="text-white font-black text-xl">₹{totalCost.toLocaleString("en-IN")}</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Temp Drop</p>
            <p className="text-heat-green font-black text-xl">-{totalDrop.toFixed(1)}°C</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Interventions</p>
            <p className="text-white font-black text-xl">{applied.length}</p>
          </div>
        </div>
      </div>

      <div className="flex justify-center pb-12">
        <button onClick={onReset}
          className="flex items-center gap-2 border border-teal-600 text-teal-400 px-6 py-2.5 rounded font-semibold hover:bg-teal-800 transition-all">
          <ArrowLeft className="w-4 h-4" /> Start Over
        </button>
      </div>
    </motion.div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function PipelinePage() {
  const [activeStep, setActiveStep] = useState(1);
  const [file, setFile]             = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [segResult, setSegResult]   = useState<SegResult | null>(null);
  const [trellis, setTrellis]       = useState<TrellisResult | null>(null);
  const [interventions, setInterventions]   = useState<Intervention[] | null>(null);
  const [selectedIdxs, setSelectedIdxs]     = useState<number[]>([]);
  const [interventionImage, setInterventionImage]     = useState<string | null>(null);
  const [interventionTrellis, setInterventionTrellis] = useState<TrellisResult | null>(null);
  const [budget, setBudget]   = useState(500000);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setActiveStep(1); setFile(null); setPreviewUrl(null);
    setSegResult(null); setTrellis(null); setInterventions(null);
    setSelectedIdxs([]); setInterventionImage(null); setInterventionTrellis(null);
    setError(null);
  }, []);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setPreviewUrl(URL.createObjectURL(f)); setError(null);
  }

  function toggleIntervention(idx: number) {
    setSelectedIdxs(prev =>
      prev.includes(idx) ? prev.filter(i => i !== idx) : [...prev, idx]
    );
  }

  async function runSegment() {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/api/segment`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSegResult(data);
      // Stay on step 2 to show results — user clicks Continue to go to step 3
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function runTrellis() {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData(); form.append("file", file);
      const res = await fetch(`${API}/api/generate-3d`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setTrellis(await res.json()); setActiveStep(4);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function runInterventions() {
    if (!file || !segResult) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("features_json", JSON.stringify(segResult));
      form.append("budget_inr", String(budget));
      const res = await fetch(`${API}/api/interventions`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const suggestions: Intervention[] = data.suggestions;
      setInterventions(suggestions);
      // Select all by default
      setSelectedIdxs(suggestions.map((_, i) => i));
      setActiveStep(5);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function runVisualize() {
    if (!file || !interventions) return;
    setLoading(true); setError(null);
    try {
      // Build prompt from ONLY the selected interventions
      const selected = interventions.filter((_, i) => selectedIdxs.includes(i));
      const prompt   = selected.map(i => i.what_to_do).join(". ");

      // 1. Generate intervention image
      const form1 = new FormData();
      form1.append("file", file);
      form1.append("prompt", prompt);
      const res1 = await fetch(`${API}/api/generate-intervention-image`, { method: "POST", body: form1 });
      if (!res1.ok) throw new Error(await res1.text());
      const data1 = await res1.json();
      setInterventionImage(data1.image_url);

      // 2. Send that new image to TRELLIS for the "after" 3D model (non-fatal if HuggingFace is queued)
      try {
        const imgBlob  = await (await fetch(`${API}${data1.image_url}`)).blob();
        const newFile  = new File([imgBlob], "after.png", { type: "image/png" });
        const form2    = new FormData();
        form2.append("file", newFile);
        const res2 = await fetch(`${API}/api/generate-3d`, { method: "POST", body: form2 });
        if (res2.ok) {
          setInterventionTrellis(await res2.json());
        }
      } catch (trellisErr) {
        console.warn("TRELLIS 3D generation skipped or timed out:", trellisErr);
      }

      setActiveStep(6);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }

  // ── Step renderers ────────────────────────────────────────────────────────

  function renderStep() {
    switch (activeStep) {

      /* ── 1: Upload ────────────────────────────────────────────────────── */
      case 1:
        return (
          <div className="flex flex-col items-center gap-6">
            <div onClick={() => fileRef.current?.click()}
              className={`w-full max-w-lg h-64 border-2 border-dashed rounded-xl flex flex-col items-center
                justify-center gap-3 cursor-pointer transition-all
                ${previewUrl ? "border-heat-green" : "border-teal-700 hover:border-heat-green hover:bg-teal-800/20"}`}>
              {previewUrl
                ? <img src={previewUrl} alt="preview" className="h-full w-full object-contain rounded-xl" />
                : <><Upload className="w-12 h-12 text-teal-600" />
                    <p className="text-gray-400 text-sm">Click to upload · JPG, PNG, WEBP</p>
                    <p className="text-gray-600 text-xs">GPS-tagged photos match satellite thermal data</p>
                  </>}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
            {previewUrl && (
              <motion.button initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-2 bg-heat-green text-teal-900 px-8 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all">
                Analyse This Image <ChevronRight className="w-5 h-5" />
              </motion.button>
            )}
          </div>
        );

      /* ── 2: SAM2 Analysis — shows results inline ──────────────────────── */
      case 2:
        return (
          <div className="flex flex-col items-center gap-6 w-full max-w-2xl mx-auto">
            {!segResult ? (
              /* Pre-analysis state */
              <>
                {previewUrl && (
                  <img src={previewUrl} alt="uploaded" className="max-h-48 rounded-lg border border-teal-700" />
                )}
                <div className="bg-teal-900/40 border border-teal-700 rounded-xl p-5 w-full space-y-2">
                  <p className="text-white font-semibold text-sm mb-3">What this step detects:</p>
                  {[
                    ["🌿", "Vegetation / Trees", "green"],
                    ["🏗️", "Buildings & Facades", "orange"],
                    ["🛣️", "Roads & Pavement", "slate/grey"],
                    ["☁️", "Open Sky (used for SVF)", "light blue"],
                    ["🌑", "Shadows", "dark indigo"],
                  ].map(([icon, label, colour]) => (
                    <div key={label as string} className="flex items-center gap-2 text-sm text-gray-400">
                      <span>{icon}</span>
                      <span className="flex-1">{label}</span>
                      <span className="text-[10px] text-gray-600 italic">→ coloured {colour}</span>
                    </div>
                  ))}
                </div>
                <button onClick={runSegment} disabled={loading}
                  className="flex items-center gap-2 bg-heat-green text-teal-900 px-8 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all disabled:opacity-50">
                  {loading
                    ? <><Loader2 className="w-5 h-5 animate-spin" /> Analysing…</>
                    : <>Run Surface Analysis <ChevronRight className="w-5 h-5" /></>}
                </button>
              </>
            ) : (
              /* Post-analysis: show overlay + stats, then Continue button */
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="w-full space-y-5">
                <p className="text-heat-green font-semibold flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" /> Surface Analysis Complete
                </p>

                <SegStats seg={segResult} />

                <div className="border-t border-teal-800/50 pt-5 flex flex-col items-center gap-3">
                  <p className="text-gray-500 text-sm text-center">
                    Generate a rotatable 3D model from your photo using TRELLIS (~1–3 min on HuggingFace GPU).
                  </p>
                  <div className="flex gap-3 flex-wrap justify-center">
                    <button onClick={runTrellis} disabled={loading}
                      className="flex items-center gap-2 bg-heat-green text-teal-900 px-6 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all disabled:opacity-50">
                      {loading
                        ? <><Loader2 className="w-5 h-5 animate-spin" /> Generating 3D…</>
                        : "Generate 3D Model"}
                    </button>
                    <button onClick={() => setActiveStep(4)}
                      className="flex items-center gap-2 border border-teal-600 text-teal-400 px-6 py-3 rounded font-semibold hover:bg-teal-800 transition-all text-sm">
                      Skip → Interventions
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        );

      /* ── 3: 3D model (just the viewer + go to interventions) ─────────── */
      case 3:
        return (
          <div className="flex flex-col items-center gap-6 w-full max-w-2xl mx-auto">
            {trellis && (
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="w-full space-y-3">
                <p className="text-heat-green font-semibold flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" /> 3D Model Ready — Drag to rotate · Scroll to zoom
                </p>
                <GLBViewer url={`${API}${trellis.model_url}`} height="h-96" />
                <a href={`${API}${trellis.model_url}`} download
                  className="inline-flex items-center gap-2 border border-teal-600 text-teal-400 px-4 py-2 rounded text-sm font-semibold hover:bg-teal-800 transition-all">
                  <Download className="w-4 h-4" /> Download GLB
                </a>
              </motion.div>
            )}
            <button onClick={() => setActiveStep(4)}
              className="flex items-center gap-2 bg-heat-green text-teal-900 px-8 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all">
              Get AI Intervention Strategies <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        );

      /* ── 4: Budget + Gemini ──────────────────────────────────────────── */
      case 4:
        return (
          <div className="flex flex-col items-center gap-6 w-full max-w-xl mx-auto">
            <p className="text-gray-400 text-sm text-center">
              Gemini Vision analyses your actual photo and suggests strategies calibrated to your budget.
            </p>
            <div className="w-full bg-teal-900/40 border border-teal-700 rounded-xl p-6 space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-white font-bold">Total Budget</p>
                <p className="text-heat-green font-black text-2xl">₹{budget.toLocaleString("en-IN")}</p>
              </div>
              <input type="range" min={50000} max={2000000} step={50000} value={budget}
                onChange={e => setBudget(Number(e.target.value))}
                className="w-full accent-[#06D6A0] h-2 cursor-pointer" />
              <div className="flex justify-between text-xs text-gray-600">
                <span>₹50,000</span><span>₹20,00,000</span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {[
                  { label: "Small", amount: 100000 },
                  { label: "Medium", amount: 500000 },
                  { label: "Large", amount: 1000000 },
                  { label: "Flagship", amount: 2000000 },
                ].map(t => (
                  <button key={t.label} onClick={() => setBudget(t.amount)}
                    className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all
                      ${budget === t.amount
                        ? "bg-heat-green text-teal-900 border-heat-green"
                        : "border-teal-700 text-gray-400 hover:border-heat-green"}`}>
                    {t.label} (₹{(t.amount / 100000).toFixed(0)}L)
                  </button>
                ))}
              </div>
            </div>
            <button onClick={runInterventions} disabled={loading}
              className="flex items-center gap-2 bg-heat-green text-teal-900 px-8 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all disabled:opacity-50 w-full justify-center">
              {loading
                ? <><Loader2 className="w-5 h-5 animate-spin" /> Gemini is analysing your image…</>
                : <>Suggest Interventions <ChevronRight className="w-5 h-5" /></>}
            </button>
          </div>
        );

      /* ── 5: Selectable intervention cards + Visualize ───────────────── */
      case 5:
        return (
          <div className="flex flex-col items-center gap-6 w-full max-w-2xl mx-auto">
            <div className="text-center w-full">
              <h3 className="text-xl font-bold text-white mb-1">
                Select interventions to visualize
              </h3>
              <p className="text-gray-400 text-sm">
                Tick one or more strategies below. Only selected ones will be drawn into the AI image.
              </p>
            </div>

            {/* "Select all / none" toggle */}
            <div className="flex gap-3 self-start">
              <button onClick={() => setSelectedIdxs((interventions ?? []).map((_, i) => i))}
                className="text-xs text-teal-400 border border-teal-700 px-3 py-1 rounded hover:bg-teal-800">
                Select All
              </button>
              <button onClick={() => setSelectedIdxs([])}
                className="text-xs text-gray-500 border border-teal-800 px-3 py-1 rounded hover:bg-teal-900">
                Clear
              </button>
              <span className="text-xs text-gray-600 self-center">
                {selectedIdxs.length} / {interventions?.length ?? 0} selected
              </span>
            </div>

            {/* Intervention cards */}
            <div className="w-full space-y-3">
              {interventions?.map((item, i) => (
                <InterventionCard
                  key={i}
                  item={item}
                  index={i}
                  selected={selectedIdxs.includes(i)}
                  onToggle={() => toggleIntervention(i)}
                />
              ))}
            </div>

            {/* Selected summary */}
            {selectedIdxs.length > 0 && interventions && (
              <div className="w-full bg-teal-900/40 border border-teal-700 rounded-xl p-4">
                <div className="grid grid-cols-3 text-center gap-2">
                  <div>
                    <p className="text-[10px] text-gray-500">Est. Cost</p>
                    <p className="text-white font-black">
                      ₹{interventions.filter((_, i) => selectedIdxs.includes(i))
                          .reduce((s, x) => s + x.budget_inr, 0).toLocaleString("en-IN")}
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500">Temp Drop</p>
                    <p className="text-heat-green font-black">
                      -{interventions.filter((_, i) => selectedIdxs.includes(i))
                           .reduce((s, x) => s + x.temp_reduction_c, 0).toFixed(1)}°C
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500">Strategies</p>
                    <p className="text-white font-black">{selectedIdxs.length}</p>
                  </div>
                </div>
              </div>
            )}

            <button onClick={runVisualize} disabled={loading || selectedIdxs.length === 0}
              className="flex items-center gap-2 bg-heat-green text-teal-900 px-8 py-3 rounded font-bold uppercase hover:bg-heat-yellow transition-all disabled:opacity-50 w-full justify-center">
              {loading
                ? <><Loader2 className="w-5 h-5 animate-spin" /> Generating image & 3D model…</>
                : `Visualize ${selectedIdxs.length} Intervention${selectedIdxs.length !== 1 ? "s" : ""} with FLUX + Trellis`}
            </button>
            <button onClick={() => setActiveStep(6)} disabled={loading}
              className="text-teal-500 text-sm hover:underline">
              Skip — go straight to Report
            </button>
          </div>
        );

      /* ── 6: Report ───────────────────────────────────────────────────── */
      case 6:
        if (!segResult || !interventions) return null;
        return (
          <Report
            segResult={segResult}
            trellisResult={trellis}
            interventions={interventions}
            budget={budget}
            selectedIdxs={selectedIdxs}
            interventionImage={interventionImage}
            interventionTrellis={interventionTrellis}
            onReset={reset}
          />
        );
    }
  }

  return (
    <main className="min-h-screen bg-[var(--background)] text-white font-sans">
      {/* Nav */}
      <div className="border-b border-teal-800/50 px-6 py-4 flex items-center gap-3 sticky top-0 z-50 bg-[var(--background)]/90 backdrop-blur-md">
        <a href="/" className="text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </a>
        <span className="text-heat-green font-bold tracking-widest text-lg uppercase">HeatScape</span>
        <span className="text-gray-600 mx-2">/</span>
        <span className="text-gray-400 text-sm uppercase tracking-wider">Analysis Pipeline</span>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10">
        {/* Step bar */}
        <div className="flex items-start justify-between mb-12 relative">
          <div className="absolute top-4 left-0 right-0 h-px bg-teal-800 z-0" />
          {STEPS.map(s => <StepIndicator key={s.id} step={s} activeStep={activeStep} />)}
        </div>

        {/* Step title */}
        <div className="mb-8">
          <span className="text-[10px] font-mono tracking-widest text-heat-green">
            STEP {String(activeStep).padStart(2, "0")} / 0{STEPS.length}
          </span>
          <h1 className="text-3xl md:text-4xl font-black uppercase tracking-tight text-white mt-1">
            {STEPS[activeStep - 1]?.label}
          </h1>
          <p className="text-gray-400 mt-1 text-sm">{STEPS[activeStep - 1]?.desc}</p>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 bg-red-950/40 border border-red-700 rounded-lg p-4 text-sm text-red-300">
            <p className="font-semibold mb-1">Error</p>
            <p className="text-xs font-mono">{error}</p>
          </div>
        )}

        {/* Step content */}
        <AnimatePresence mode="wait">
          <motion.div key={activeStep}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -14 }}
            transition={{ duration: 0.3, ease: "easeOut" }}>
            {renderStep()}
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}
