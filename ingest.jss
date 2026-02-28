// ingest.js
import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import pdf from 'pdf-parse'; // Necesitas instalar esto: npm install pdf-parse

const client = new OpenAI({ apiKey: 'TU_API_KEY_AQUI' });
const DOCS_DIR = './docs';
const OUTPUT_FILE = './data/embeddings.json';

async function ingest() {
  const files = fs.readdirSync(DOCS_DIR).filter(f => f.endsWith('.pdf'));
  const chunks = [];

  for (const file of files) {
    console.log(`Procesando: ${file}...`);
    const dataBuffer = fs.readFileSync(path.join(DOCS_DIR, file));
    const data = await pdf(dataBuffer);
    
    // Dividimos el texto en trozos de 1000 caracteres para que la IA sea precisa
    const textChunks = data.text.match(/[\s\S]{1,1000}/g) || [];

    for (const text of textChunks) {
      const response = await client.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
      });

      chunks.push({
        text: text,
        vec: response.data[0].embedding,
        source: file
      });
    }
  }

  if (!fs.existsSync('./data')) fs.mkdirSync('./data');
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify({ chunks }));
  console.log("✅ Cerebro generado en data/embeddings.json");
}

ingest();
