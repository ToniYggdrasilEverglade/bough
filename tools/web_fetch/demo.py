import asyncio
import json
from pathlib import Path
from web_fetch_wrapper import WebFetchConfig, dispatch_web_action

async def main():
    config = WebFetchConfig(
        executable=(Path(__file__).parent / "target/release/web_fetch").resolve(),
        enabled=True,
        allowed_hosts=("example.com",),
    )
    result = await dispatch_web_action(
        {"kind": "web_fetch", "url": "https://example.com/"}, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
