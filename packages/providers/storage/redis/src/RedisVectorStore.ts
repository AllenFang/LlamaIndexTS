import type { BaseNode } from "@llamaindex/core/schema";
import { MetadataMode } from "@llamaindex/core/schema";
import {
  BaseVectorStore,
  metadataDictToNode,
  nodeToMetadata,
  type VectorStoreBaseParams,
  type VectorStoreQuery,
  type VectorStoreQueryResult,
} from "@llamaindex/core/vector-store";
import {
  SchemaFieldTypes,
  VectorAlgorithms,
  createClient,
  createCluster,
  type RedisClientOptions,
} from "redis";

export type RedisClientType =
  | ReturnType<typeof createClient>
  | ReturnType<typeof createCluster>;

export interface RedisVectorStoreOptions extends VectorStoreBaseParams {
  client?: RedisClientType;
  redisClientOptions?: RedisClientOptions;
  indexName?: string;
}

export class RedisVectorStore extends BaseVectorStore {
  storesText: boolean = true;

  private db: RedisClientType;
  private indexName: string;

  constructor(options: RedisVectorStoreOptions = {}) {
    super(options);
    if (options.client) {
      this.db = options.client;
    } else if (options.redisClientOptions) {
      this.db = createClient(options.redisClientOptions);
    } else {
      this.db = createClient(); // default localhost
    }
    this.indexName = options.indexName ?? "llamaindex-default-redis-index";
  }

  client(): RedisClientType {
    return this.db;
  }

  private async connect(): Promise<void> {
    if (!this.db.isOpen) {
      await this.db.connect();
    }
  }

  private async createIndexIfNotExists(dimension: number) {
    await this.connect();
    try {
      await this.db.ft.info(this.indexName);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      if (
        e instanceof Error &&
        e.message.toLowerCase().includes("unknown index name")
      ) {
        await this.db.ft.create(
          this.indexName,
          {
            doc_id: {
              type: SchemaFieldTypes.TAG,
            },
            text: {
              type: SchemaFieldTypes.TEXT,
            },
            metadata: {
              type: SchemaFieldTypes.TEXT,
            },
            vector: {
              type: SchemaFieldTypes.VECTOR,
              ALGORITHM: VectorAlgorithms.FLAT,
              TYPE: "FLOAT32",
              DIM: dimension,
              DISTANCE_METRIC: "COSINE",
            },
          },
          {
            ON: "HASH",
            PREFIX: `node:`,
          },
        );
      } else {
        throw e;
      }
    }
  }

  async add(nodes: BaseNode[]): Promise<string[]> {
    if (nodes.length === 0) {
      return [];
    }

    if (!nodes[0] || !nodes[0].getEmbedding().length) {
      throw new Error("No valid vectors provided");
    }
    await this.connect();

    const dimension = nodes[0].getEmbedding().length;
    await this.createIndexIfNotExists(dimension);

    const pipeline = this.db.multi();
    const ids: string[] = [];

    for (const node of nodes) {
      const nodeKey = `node:${node.id_}`;
      ids.push(node.id_);

      const metadata = nodeToMetadata(node);
      const vector = node.getEmbedding();
      const vectorBuffer = Buffer.from(new Float32Array(vector).buffer);

      pipeline.hSet(nodeKey, {
        text: node.getContent(MetadataMode.NONE),
        doc_id: node.id_,
        metadata: JSON.stringify(metadata),
        vector: vectorBuffer,
      });
    }

    await pipeline.exec();
    return ids;
  }

  async delete(refDocId: string): Promise<void> {
    await this.connect();
    const nodeKey = `node:${refDocId}`;
    await this.db.del(nodeKey);
  }

  async query(
    query: VectorStoreQuery,
    options?: object,
  ): Promise<VectorStoreQueryResult> {
    await this.connect();
    const vectorScoreField = "vector_score";

    const topK = query.similarityTopK ?? 10;
    const queryEmbedding = query.queryEmbedding;

    if (!queryEmbedding) {
      throw new Error("Query embedding is required.");
    }

    const queryVector = Buffer.from(new Float32Array(queryEmbedding).buffer);

    const ftQuery = `*=>[KNN ${topK} @vector $queryVector AS ${vectorScoreField}]`;

    const searchResult = await this.db.ft.search(this.indexName, ftQuery, {
      PARAMS: {
        queryVector: queryVector,
      },
      RETURN: ["doc_id", "text", "vector", "metadata", vectorScoreField],
      SORTBY: vectorScoreField,
      DIALECT: 2,
      LIMIT: {
        from: 0,
        size: topK,
      },
    });

    const nodes: BaseNode[] = [];
    const similarities: number[] = [];
    const ids: string[] = [];

    for (const doc of searchResult.documents) {
      const id = doc.value.doc_id as string;
      const text = doc.value.text as string;
      const metadata = JSON.parse(doc.value.metadata as string);
      const score = parseFloat(doc.value[vectorScoreField] as string);

      const node = metadataDictToNode(doc.value, {
        fallback: {
          id: id,
          text: text,
          metadata: metadata,
        },
      });
      node.embedding = doc.value.vector as unknown as number[];

      ids.push(id);
      similarities.push(1 - score); // Cosine distance to similarity
      nodes.push(node);
    }

    return {
      nodes: nodes,
      similarities: similarities,
      ids: ids,
    };
  }
}
