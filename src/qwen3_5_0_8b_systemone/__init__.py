def main() -> None:
    import uvicorn

    from .engine import get_engine

    get_engine().load()
    uvicorn.run("qwen3_5_0_8b_systemone.app:app", host="127.0.0.1", port=8000)
