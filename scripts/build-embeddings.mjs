import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";

// Configuración de PDF.js para entorno Node
pdfjsLib.GlobalWorkerOptions.workerSrc = undefined;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.resolve(__dirname, "..", "docs");
const DATA_DIR = path.resolve(__dirname, "..", "data");
const OUT_FILE = path.join(DATA_DIR, "embeddings.json");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Limpia y anonimiza el texto eliminando referencias corporativas
 */
function cleanAndAnonymize(text) {
  return text
    .replace(/La Anónima/gi, "esta organización profesional")
    .replace(/procedimientos de la empresa/gi, "protocolos de buenas prácticas")
    .replace(/estándares de la compañía/gi, "estándares de seguridad alimentaria")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Divide el texto en fragmentos manejables con solapamiento para no perder contexto
 */
function chunkText(text, maxLen = 1200, overlap = 200) {
  const chunks = [];
  const sanitized = cleanAndAnonymize(text);
  for (let i = 0; i < sanitized.length; i += (maxLen - overlap)) {
    chunks.push(sanitized.slice(i, i + maxLen));
  }
  return chunks;
}

/**
 * Extrae texto de un PDF página por página
 */
async function extractPdfText(pdfPath) {
  try {
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
  } catch (err) {
    console.error(`Error leyendo PDF ${pdfPath}:`, err);
    return "";
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("ERROR: OPENAI_API_KEY no detectada en las variables de entorno.");
  }

  await fs.mkdir(DATA_DIR, { recursive: true });
  
  const files = (await fs.readdir(DOCS_DIR))
    .filter((f) => f.toLowerCase().endsWith(".pdf"))
    .sort();

  const finalData = [];
  let totalChunks = 0;

  for (const file of files) {
    console.log(`📖 Procesando: ${file}...`);
    const fullPath = path.join(DOCS_DIR, file);
    const rawText = await extractPdfText(fullPath);
    const parts = chunkText(rawText);
    
    totalChunks += parts.length;

    // Procesamos en lotes de 100 para optimizar llamadas a la API
    for (let i = 0; i < parts.length; i += 100) {
      const batch = parts.slice(i, i + 100);
      const resp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch,
      });

      resp.data.forEach((d, j) => {
        finalData.push({
          source: file,
          text: batch[j], // Guardamos el texto limpio para que la IA lo lea
          embedding: d.embedding, // El vector numérico para la búsqueda semántica
        });
      });

      // Breve pausa para respetar límites de la API
      await new Promise((r) => setTimeout(r, 100));
    }
    console.log(`✅ ${file} completado (${parts.length} fragmentos)`);
  }

  const payload = {
    version: "inocuo-v1-2026",
    generatedAt: new Date().toISOString(),
    docsCount: files.length,
    chunks: finalData, // IMPORTANTE: chat.js buscará en este array
  };

  await fs.writeFile(OUT_FILE, JSON.stringify(payload));
  console.log(`\n🚀 ¡Cerebro activado! ${finalData.length} vectores guardados en ${OUT_FILE}`);
}

main().catch((e) => {
  console.error("Error fatal en el build:", e);
  process.exit(1);
});
