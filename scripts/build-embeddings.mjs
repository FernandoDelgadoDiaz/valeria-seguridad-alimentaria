import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";
pdfjsLib.GlobalWorkerOptions.workerSrc = undefined;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.resolve(__dirname, "..", "docs");
const DATA_DIR = path.resolve(__dirname, "..", "data");
const OUT_FILE = path.join(DATA_DIR, "embeddings.json");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Filtro para eliminar marcas corporativas y limpiar el texto
function cleanAndAnonymize(text) {
  return text
    .replace(/La Anónima/gi, "esta organización") // Cambia el nombre de la empresa
    .replace(/S\.A\./gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function chunkText(text, maxLen = 1200, overlap = 200) {
  const chunks = [];
  const clean = cleanAndAnonymize(text);
  for (let i = 0; i < clean.length; i += (maxLen - overlap)) {
    chunks.push(clean.slice(i, i + maxLen));
  }
  return chunks;
}

async function extractPdfText(pdfPath) {
  const u8 = new Uint8Array(await fs.readFile(pdfPath));
  const loadingTask = pdfjsLib.getDocument({ data: u8 });
  const pdf = await loadingTask.promise;
  let out = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    out += content.items.map((i) => i.str).join(" ") + "\n";
  }
  return out;
}

async function main() {
  if (!process.env.OPENAI_API_KEY) throw new Error("Falta OPENAI_API_KEY");
  await fs.mkdir(DATA_DIR, { recursive: true });

  const files = (await fs.readdir(DOCS_DIR))
    .filter((f) => f.toLowerCase().endsWith(".pdf")).sort();

  const finalChunks = []; // Aquí guardaremos TEXTO + VECTOR

  for (const file of files) {
    console.log(`Procesando: ${file}...`);
    const text = await extractPdfText(path.join(DOCS_DIR, file));
    const parts = chunkText(text);

    for (let i = 0; i < parts.length; i += 100) {
      const batch = parts.slice(i, i + 100);
      const resp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch,
      });

      resp.data.forEach((d, j) => {
        finalChunks.push({
          source: file,
          text: batch[j],     // ¡IMPORTANTÍSIMO!: Guardar el texto
          vec: d.embedding,   // El vector para buscar
        });
      });
      await new Promise((r) => setTimeout(r, 150));
    }
  }

  // Guardamos la estructura que Inocuo necesita para "despertar"
  await fs.writeFile(OUT_FILE, JSON.stringify({ 
    version: "inocuo-v1", 
    chunks: finalChunks 
  }));
  
  console.log(`✅ Cerebro cargado: ${finalChunks.length} fragmentos en ${OUT_FILE}`);
}

main().catch(console.error);
