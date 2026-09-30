install:  ; pip install -r requirements.txt
run:      ; uvicorn atlas.api:app --reload --port 8000
cli:      ; python -m atlas.cli
eval:     ; python -m atlas.evaluate
test:     ; pytest -q
mcp:      ; python -m atlas.mcp_server
