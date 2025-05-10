import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

/**
 * Upsert 1 vector embedding vào Pinecone
 * @param {string} id - ID vector (thường dùng filename)
 * @param {number[]} vector - Mảng embedding
 * @param {object} metadata - Thông tin thêm
 */
export async function upsertImageEmbedding(id, vector, metadata = {}) {
  await index.upsert([
    {
      id,
      values: vector,
      metadata,
    },
  ]);
}
