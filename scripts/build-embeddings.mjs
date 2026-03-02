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

    // Si el texto extraído es muy corto, puede ser un PDF escaneado
    if (out.trim().length < 100) {
      console.warn(`⚠️  El PDF ${pdfPath} tiene muy poco texto (${out.length} caracteres). ¿Será una imagen escaneada?`);
    }
    return out;
  } catch (err) {
    console.error(`❌ Error leyendo PDF ${pdfPath}:`, err.message);
    return null; // Indicar que falló
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("ERROR: OPENAI_API_KEY no detectada en las variables de entorno.");
  }

  await fs.mkdir(DATA_DIR, { recursive: true });

  // Leemos tanto PDFs como archivos de texto
  const files = (await fs.readdir(DOCS_DIR))
    .filter((f) => f.toLowerCase().endsWith(".pdf") || f.toLowerCase().endsWith(".txt"))
    .sort();

  console.log(`📁 Archivos encontrados: ${files.length}`);
  if (files.length === 0) {
    console.warn("⚠️  No hay archivos PDF o TXT en docs/");
    return;
  }

  const finalData = [];
  let totalChunks = 0;

  for (const file of files) {
    console.log(`\n📖 Procesando: ${file}...`);
    const fullPath = path.join(DOCS_DIR, file);
    let rawText;

    if (file.toLowerCase().endsWith(".pdf")) {
      rawText = await extractPdfText(fullPath);
    } else {
      // Archivo de texto: leer directamente
      try {
        rawText = await fs.readFile(fullPath, "utf-8");
      } catch (err) {
        console.error(`❌ Error leyendo archivo de texto ${fullPath}:`, err.message);
        rawText = null;
      }
    }

    if (!rawText) {
      console.log(`⏩ Saltando ${file} por errores de lectura.`);
      continue;
    }

    const parts = chunkText(rawText);
    console.log(`   → ${parts.length} fragmentos generados`);

    if (parts.length === 0) {
      console.log(`⏩ ${file} no generó fragmentos (texto vacío).`);
      continue;
    }

    totalChunks += parts.length;

    // Procesamos en lotes de 100 para optimizar llamadas a la API
    for (let i = 0; i < parts.length; i += 100) {
      const batch = parts.slice(i, i + 100);
      try {
        const resp = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: batch,
        });

        resp.data.forEach((d, j) => {
          finalData.push({
            source: file,
            text: batch[j],
            embedding: d.embedding,
          });
        });

        console.log(`   → Lote ${Math.floor(i / 100) + 1} procesado (${batch.length} embeddings)`);
      } catch (err) {
        console.error(`   ❌ Error en lote de embeddings:`, err.message);
        // Continuamos con el siguiente lote
      }

      // Breve pausa para respetar límites de la API
      await new Promise((r) => setTimeout(r, 100));
    }
    console.log(`✅ ${file} completado`);
  }

  const payload = {
    version: "inocuo-v1-2026",
    generatedAt: new Date().toISOString(),
    docsCount: files.length,
    chunks: finalData,
  };

  await fs.writeFile(OUT_FILE, JSON.stringify(payload, null, 2));
  console.log(`\n🚀 ¡Cerebro activado! ${finalData.length} vectores guardados en ${OUT_FILE}`);
}

main().catch((e) => {
  console.error("Error fatal en el build:", e);
  process.exit(1);
});