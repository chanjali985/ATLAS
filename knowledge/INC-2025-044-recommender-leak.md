# INC-2025-044: Recommender memory leak and GC pressure
Summary: The recommender model cache did not evict old embeddings. Heap grew over days, garbage collection pauses increased CPU usage, and pods were OOMKilled.
Root cause: unbounded LRU cache after config change (max_entries unset).
Fix: set cache max_entries, add heap usage alert at 80%, roll pods.
