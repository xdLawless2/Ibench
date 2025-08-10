from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import aiohttp
from asyncio_throttle import Throttler
from tqdm.asyncio import tqdm


def slug_to_name(slug: str) -> str:
    return slug.split("/", 1)[-1]


def read_models(path: Path) -> List[str]:
    slugs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            slugs.append(s)
    if not slugs:
        raise SystemExit(f"No model slugs found in {path}")
    return slugs


def read_manifest(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    if not items:
        raise SystemExit(f"Empty manifest at {path}")
    return items


def b64_data_url(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def compute_accuracy(truth: List[int], preds: List[int]) -> Tuple[float, int, int]:
    if len(truth) != len(preds):
        raise ValueError(f"Length mismatch: truth={len(truth)} vs preds={len(preds)}")
    correct = sum(1 for t, p in zip(truth, preds) if t == p)
    total = len(truth)
    return (correct / total if total else 0.0, correct, total)


def plot_bar(results: Dict[str, float], out_path: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    names = list(results.keys())
    vals = [results[k] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, vals, color="#4C78A8")
    for i, n in enumerate(names):
        if n.lower() == "human":
            bars[i].set_color("#72B7B2")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title or "IBench Accuracy")
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


async def get_model_info(session: aiohttp.ClientSession, api_key: str) -> Dict[str, int]:
    """Fetch model information from OpenRouter API to get max output tokens."""
    model_limits = {}
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with session.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                for model in data.get("data", []):
                    model_id = model.get("id")
                    # Get max_completion_tokens from top_provider if available
                    top_provider = model.get("top_provider", {})
                    max_tokens = top_provider.get("max_completion_tokens")
                    if model_id and max_tokens:
                        model_limits[model_id] = min(max_tokens, 16384)  # Cap at 16k for safety
    except:
        pass  # Fall back to defaults if API call fails
    return model_limits


async def process_single_image(
    session: aiohttp.ClientSession,
    throttler: Throttler,
    api_key: str,
    model_slug: str,
    item: Dict,
    prompt: str,
    debug_dir: Path,
    max_tokens: int = 8192
) -> Tuple[int, str]:
    """Process a single image with rate limiting and debug logging."""
    img_path = Path(item["image_path"])
    img_name = img_path.stem
    
    async with throttler:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ibench.ai",
                "X-Title": "IBench Visual Benchmark"
            }
            
            payload = {
                "model": model_slug,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": b64_data_url(img_path)}}
                        ]
                    }
                ],
                "max_tokens": max_tokens,  # Use model-specific limit
                "temperature": 0
            }
            
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout for reasoning models
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
                    
                    # Save raw response to debug file
                    model_name = slug_to_name(model_slug).replace("/", "_")
                    debug_file = debug_dir / f"{model_name}_{img_name}.json"
                    debug_data = {
                        "model": model_slug,
                        "image": str(img_path),
                        "truth": item["label"],
                        "raw_response": text,
                        "full_response": data
                    }
                    with open(debug_file, "w") as f:
                        json.dump(debug_data, f, indent=2)
                    
                    # Extract first integer from response (1-6 only)
                    digit = next((int(ch) for ch in text if ch in "123456"), 0)
                    return digit, text
                else:
                    error_text = await response.text()
                    return 0, f"HTTP {response.status}: {error_text}"
        except Exception as e:
            return 0, f"Exception: {str(e)}"


async def process_model(
    api_key: str,
    model_slug: str,
    items: List[Dict],
    prompt: str,
    debug_dir: Path,
    model_limits: Dict[str, int],
    max_concurrent: int = 5,
    requests_per_second: float = 10
) -> Tuple[List[int], int]:
    """Process all images for a single model with concurrency control."""
    # Create model-specific debug directory
    model_name = slug_to_name(model_slug).replace("/", "_")
    model_debug_dir = debug_dir / model_name
    model_debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Get max tokens for this model (default to 8192 if not found)
    max_tokens = model_limits.get(model_slug, 8192)
    
    # Create throttler for rate limiting
    throttler = Throttler(rate_limit=requests_per_second)
    
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all images
        tasks = [
            process_single_image(session, throttler, api_key, model_slug, item, prompt, model_debug_dir, max_tokens)
            for item in items
        ]
        
        # Process with progress bar
        results = await tqdm.gather(
            *tasks,
            desc=f"  {slug_to_name(model_slug):30s}",
            unit="img",
            leave=False,
            position=1
        )
        
        preds = [r[0] for r in results]
        responses = [r[1] for r in results]
        errors = sum(1 for p in preds if p == 0)
    
    # Save summary for this model
    summary_file = model_debug_dir / "_summary.json"
    summary_data = {
        "model": model_slug,
        "predictions": preds,
        "truth": [int(it["label"]) for it in items],
        "responses_preview": responses[:5],
        "error_count": errors
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    return preds, errors


async def run_benchmark_async(
    api_key: str,
    items: List[Dict],
    model_slugs: List[str],
    prompt: str,
    debug_dir: Path,
    max_models_concurrent: int = 3,
    max_images_concurrent: int = 5,
    requests_per_second: float = 10
) -> List[Tuple[str, List[int], int]]:
    """Run benchmark for all models concurrently."""
    print(f"\n📊 Running benchmark on {len(model_slugs)} models with {len(items)} images each")
    print(f"   Concurrency: {max_models_concurrent} models × {max_images_concurrent} images")
    print(f"   Rate limit: {requests_per_second} requests/second")
    print(f"   Debug output: {debug_dir}")
    
    # Fetch model limits from OpenRouter API
    print("📋 Fetching model token limits...")
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        model_limits = await get_model_info(session, api_key)
    
    if model_limits:
        print(f"   Found limits for {len(model_limits)} models")
    else:
        print("   Using default limits (8192 tokens)")
    print(f"{'='*60}\n")
    
    # Clear and recreate debug directory for fresh results
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Use asyncio.Semaphore to limit concurrent models
    semaphore = asyncio.Semaphore(max_models_concurrent)
    
    async def process_model_with_semaphore(slug):
        async with semaphore:
            preds, errors = await process_model(
                api_key, slug, items, prompt, debug_dir, model_limits,
                max_images_concurrent, requests_per_second
            )
            return (slug, preds, errors)
    
    # Create tasks for all models
    tasks = [process_model_with_semaphore(slug) for slug in model_slugs]
    
    # Process all models with progress bar
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Models",
        unit="model",
        position=0
    ):
        result = await coro
        results.append(result)
        
        # Print result for this model
        slug, preds, errors = result
        truth = [int(it["label"]) for it in items]
        acc, correct, total = compute_accuracy(truth, preds)
        status = f"✅" if acc >= 0.8 else "⚠️" if acc >= 0.5 else "❌"
        print(f"{status} {slug_to_name(slug):30s}: {acc*100:6.2f}% ({correct}/{total})", end="")
        if errors > 0:
            print(f" [{errors} errors]")
        else:
            print()
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run models via OpenRouter with debug output.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--models", type=Path, default=Path("src/models.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/accuracy.png"))
    parser.add_argument("--debug-dir", type=Path, default=Path("debug_output"))
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--max-models-concurrent", type=int, default=3, help="Max models to run concurrently")
    parser.add_argument("--max-images-concurrent", type=int, default=5, help="Max images per model to process concurrently")
    parser.add_argument("--rps", type=float, default=20, help="Requests per second rate limit")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY env var")

    items = read_manifest(args.manifest)
    model_slugs = read_models(args.models)

    prompt = (
        "How many total intersection points are here (points where two distinct line segments meet)? "
        "Reply with a single integer and that's it."
    )

    # Run async benchmark
    start_time = time.time()
    results = asyncio.run(run_benchmark_async(
        api_key,
        items,
        model_slugs,
        prompt,
        args.debug_dir,
        args.max_models_concurrent,
        args.max_images_concurrent,
        args.rps
    ))
    elapsed = time.time() - start_time
    
    # Process results
    truth = [int(it["label"]) for it in items]
    per_model_acc: List[Tuple[str, float]] = []
    
    for slug, preds, errors in results:
        acc, correct, total = compute_accuracy(truth, preds)
        per_model_acc.append((slug_to_name(slug), acc))

    # Sort by accuracy desc and build results including Human=100%
    per_model_acc.sort(key=lambda x: x[1], reverse=True)
    results_dict: Dict[str, float] = {"Human": 1.0}
    results_dict.update({name: acc for name, acc in per_model_acc})

    plot_bar(results_dict, args.out, args.title)

    print(f"\n{'='*60}")
    print("📈 Final Leaderboard (Accuracy):")
    print(f"{'='*60}")
    for rank, (name, acc) in enumerate(results_dict.items(), 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji} {rank:2d}. {name:30s}: {acc*100:6.2f}%")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📊 Chart saved to: {args.out}")
    print(f"📝 Debug output: {args.debug_dir}/")


if __name__ == "__main__":
    main()