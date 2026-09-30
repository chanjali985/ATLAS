# Architecture: services
api-gateway routes to auth, catalog, search, checkout. checkout depends on payments and inventory. recommender and ml-inference serve personalization. batch-etl feeds analytics and reporting nightly. notifications is async via queue.
