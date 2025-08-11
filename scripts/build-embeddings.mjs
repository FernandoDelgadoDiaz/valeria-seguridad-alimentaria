import fs from "fs";
import path from "path";
import pdf from "pdf-parse/lib/pdf-parse.js"; // Import corregido
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const docsDir = "./docs";
const outputDir = "./data";
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
}

const files = fs.readdirSync(docsDir).filter(f => f.endsWith(".pdf"));

console.log(`Encontrados ${files.length} PDF(s) en ${docsDir}`);

let chunks = [];
let embeddings = [];

for (const file of files) {
  const filePath = path.join(docsDir, file);
  const dataBuffer = fs.readFileSync(filePath);
  const pdfData = await pdf(dataBuffer);

  const text = pdfData.text.replace(/\s+/g, " ").trim();
  const chunkSize = 800;
  for (let i = 0; i < text.length; i += chunkSize) {
    const chunk = text.substring(i, i + chunkSize);
    chunks.push({ file, chunk });
  }
}

console.log(`Generando embeddings para ${chunks.length} fragmentos...`);

for (const chunk of chunks) {
  const emb = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: chunk.chunk
  });
  embeddings.push({ file: chunk.file, embedding: emb.data[0].embedding });
}

fs.writeFileSync(path.join(outputDir, "chunks.jsonl"), JSON.stringify(chunks, null, 2));
fs.writeFileSync(path.join(outputDir, "embeddings.jsonl"), JSON.stringify(embeddings, null, 2));

console.log(`✅ Proceso completado. Archivos guardados en ${outputDir}`);