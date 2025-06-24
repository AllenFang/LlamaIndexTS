# LlamaIndex.TS Redis Vector Store

This package contains the Redis vector store for LlamaIndex.TS. It uses [Redis Stack](https://redis.io/docs/stack/) to store and query vector embeddings.

## Installation

```bash
npm install @llamaindex/provider-storage-redis
```

## Usage

```typescript
import { Document, VectorStoreIndex } from "@llamaindex/core/indices";
import { RedisVectorStore } from "@llamaindex/provider-storage-redis";
import { storageContextFromDefaults } from "@llamaindex/core/storage";

async function main() {
  // Create a new RedisVectorStore
  const vectorStore = new RedisVectorStore({
    redisClientOptions: {
      url: "redis://localhost:6379",
    },
  });

  // Create a storage context
  const storageContext = await storageContextFromDefaults({
    vectorStore,
  });

  // Create a VectorStoreIndex
  const index = await VectorStoreIndex.fromDocuments(
    [new Document({ text: "Hello world" })],
    { storageContext },
  );

  // Query the index
  const queryEngine = index.asQueryEngine();
  const response = await queryEngine.query({ query: "What did I say?" });

  console.log(response.toString());
}

main();
```

## Configuration

The `RedisVectorStore` can be configured with the following options:

- `redisClient`: An existing `RedisClientType` instance.
- `redisClientOptions`: Options to create a new `RedisClientType` instance.
- `indexName`: The name of the RediSearch index to use (default: `llamaindex-redis-index`).
