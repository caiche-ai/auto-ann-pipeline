from __future__ import annotations

import os

import uvicorn


def main() -> int:
    uvicorn.run(
        "annotation_service.inference.app:app",
        host=os.getenv("ANNOTATION_INFERENCE_BIND_ADDRESS", "0.0.0.0"),
        port=int(os.getenv("ANNOTATION_INFERENCE_PORT", "8010")),
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
