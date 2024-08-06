import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
array = AsyncEngineArray.from_args([
  EngineArgs(
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2",
    # revision
    revision=None,
    # best to keep at 32
    batch_size=32,
    device="cpu",
    engine="optimum",
    )
])

async def embed_text(engine: AsyncEmbeddingEngine): 
    async with engine: 
        embeddings, usage = await engine.embed(sentences=sentences)
    # or handle the async start / stop yourself.
    await engine.astart()
    embeddings, usage = await engine.embed(sentences=sentences)
    await engine.astop()
asyncio.run(embed_text(array[0]))