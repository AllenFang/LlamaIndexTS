import { TextNode } from "@llamaindex/core/schema";
import { VectorStoreQueryMode } from "@llamaindex/core/vector-store";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RedisVectorStore } from "../src";

import { Settings } from "@llamaindex/core/global";
import { OpenAIEmbedding } from "@llamaindex/openai";
Settings.embedModel = new OpenAIEmbedding();

// Mock the redis client
const mockFtInfo = vi.fn();
const mockFtCreate = vi.fn();
const mockFtSearch = vi.fn();
const mockHSet = vi.fn();
const mockDel = vi.fn();
const mockMulti = vi.fn(() => ({
  hSet: mockHSet,
  exec: mockExec,
}));
const mockExec = vi.fn();
const mockConnect = vi.fn();

vi.mock("redis", async (importOriginal) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const actual: any = await importOriginal();
  return {
    ...actual,
    createClient: vi.fn(() => ({
      ft: {
        info: mockFtInfo,
        create: mockFtCreate,
        search: mockFtSearch,
      },
      multi: mockMulti,
      del: mockDel,
      connect: mockConnect,
      isOpen: false,
    })),
  };
});

describe("RedisVectorStore", () => {
  let store: RedisVectorStore;

  beforeEach(() => {
    vi.clearAllMocks();
    store = new RedisVectorStore();
  });

  describe("add", () => {
    it("should add nodes to the vector store and create index if it does not exist", async () => {
      mockFtInfo.mockRejectedValue(new Error("Unknown Index name"));
      const nodes = [
        new TextNode({
          id_: "1",
          embedding: [0.1, 0.2, 0.3],
          text: "test text",
        }),
      ];

      const ids = await store.add(nodes);

      expect(mockConnect).toHaveBeenCalled();
      expect(mockFtInfo).toHaveBeenCalledWith("llamaindex-default-redis-index");
      expect(mockFtCreate).toHaveBeenCalled();
      expect(mockMulti).toHaveBeenCalled();
      expect(mockHSet).toHaveBeenCalled();
      expect(mockExec).toHaveBeenCalled();
      expect(ids).toEqual(["1"]);
    });
  });

  describe("delete", () => {
    it("should delete a node from the vector store", async () => {
      await store.delete("1");

      expect(mockConnect).toHaveBeenCalled();
      expect(mockDel).toHaveBeenCalledWith("node:1");
    });
  });

  describe("query", () => {
    it("should query the vector store", async () => {
      mockFtSearch.mockResolvedValue({
        total: 1,
        documents: [
          {
            id: "node:1",
            value: {
              doc_id: "1",
              text: "test text",
              metadata: JSON.stringify({ foo: "bar" }),
              vector_score: "0.1",
            },
          },
        ],
      });

      const result = await store.query({
        queryEmbedding: [0.1, 0.2, 0.3],
        similarityTopK: 1,
        mode: VectorStoreQueryMode.DEFAULT,
      });

      expect(mockConnect).toHaveBeenCalled();
      expect(mockFtSearch).toHaveBeenCalled();
      expect(result.nodes).toHaveLength(1);
      expect(result.ids).toEqual(["1"]);
      expect(result.similarities).toEqual([0.9]);
    });
  });
});
