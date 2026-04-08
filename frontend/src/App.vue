<template>
  <div class="page">
    <header class="header">
      <div class="logo">
        <img src="/logo_v4.png" />
      </div>
      <div class="title">Image To Code</div>
    </header>

    <main class="content">
      <section class="left">
        <h2>Upload image (1 file only)</h2>

        <div class="upload-row">
          <label class="upload-btn" :class="{ 'disabled-label': loading }">
            <span class="icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 16V4M12 4L7 9M12 4L17 9M4 20H20" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </span>
            Attach File (.png,.jpg)
            <input ref="fileInput" type="file" accept="image/png,image/jpeg" hidden :disabled="loading"
              @change="onFileChange" />
          </label>

          <select v-model="selectedEngine" :disabled="loading" class="engine-dropdown">
            <option value="colab">Mimic</option>
            <option value="gemini">Gemini Flash 2.5</option>
          </select>

          <select v-model="selectedColabModel" :disabled="loading" class="engine-dropdown">
            <option value="header">Header</option>
            <option value="body">Body</option>
            <option value="footer">Footer</option>
            <option value="full">Full Page</option>
            <option value="login">Login</option>
            <option value="regist">Register</option>
            <option value="forget">Forgot Password</option>
          </select>

          <button class="generate-btn" @click="uploadImage" :disabled="loading || !selectedFile">
            <svg class="generate-icon" width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M12 2 L12 22 M2 12 L22 12" stroke="white" stroke-width="2" />
            </svg>
            {{ loading ? "Processing..." : "Generate" }}
          </button>
        </div>

        <div v-if="previewUrl" class="preview">
          <img :src="previewUrl" alt="Preview" />
          <p class="file-info">📄 {{ fileName }} - {{ fileSize }} KB</p>
          <button class="remove-btn" @click="removeFile" :disabled="loading">❌ Remove</button>
        </div>

        <div v-if="generatedPreview" class="rendered-preview">
          <h3>Rendered Preview</h3>
          <img :src="generatedPreview" alt="Rendered result" />
        </div>

        <p v-if="errorMsg" class="error">{{ errorMsg }}</p>
      </section>

      <section class="right">
        <div class="right-header">
          <h2>Code (HTML+CSS)</h2>
          <div v-if="combinedSimilarity !== null" class="score-badge">
            Similarity: {{ (combinedSimilarity * 100).toFixed(2) }}%
          </div>
        </div>

        <div class="code-box">
          <button v-if="formattedCode && !loading" class="copy-btn" @click="copyCode" :title="copyStatus">
            <span v-if="copyStatus === 'Copied!'" class="copy-text">Copied!</span>
            <img v-else src="/copy.png" alt="Copy" />
          </button>

          <div v-if="loading" class="loading-state">
            <div class="spinner"></div>
            <p>Processing image, please wait...</p>
          </div>

          <div v-else-if="formattedCode" class="code-content">
            <pre>{{ formattedCode }}</pre>
          </div>

          <p v-else class="placeholder-text">
            Generated code will appear here...
          </p>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import { html as beautifyHtml } from "js-beautify";

const previewUrl = ref(null);
const selectedFile = ref(null);
const fileName = ref("");
const fileSize = ref(0);
const sourceCode = ref("");
const loading = ref(false);
const errorMsg = ref("");
const copyStatus = ref("Copy code");
const selectedEngine = ref("colab");
const selectedColabModel = ref("header");
const fileInput = ref(null);
const combinedSimilarity = ref(null);
const generatedPreview = ref(null);

let abortController = null;
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

const formattedCode = computed(() => {
  if (!sourceCode.value) return "";

  try {
    return beautifyHtml(sourceCode.value.trim(), {
      indent_size: 2,
      indent_inner_html: true,
      wrap_line_length: 120,
      preserve_newlines: true,
      max_preserve_newlines: 2,
      end_with_newline: false,
      extra_liners: []
    });
  } catch (err) {
    console.error("Format failed", err);
    return sourceCode.value;
  }
});

const onFileChange = (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (file.size > MAX_FILE_SIZE) {
    errorMsg.value = "⚠️ File is too large (max 5MB)";
    removeFile();
    return;
  } else {
    errorMsg.value = "";
  }

  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);

  selectedFile.value = file;
  fileName.value = file.name;
  fileSize.value = Math.round(file.size / 1024);
  sourceCode.value = "";
  combinedSimilarity.value = null;
  generatedPreview.value = null;
};

const removeFile = () => {
  if (abortController) abortController.abort();
  loading.value = false;

  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = null;
  selectedFile.value = null;
  fileName.value = "";
  fileSize.value = 0;
  sourceCode.value = "";
  errorMsg.value = "";
  combinedSimilarity.value = null;
  generatedPreview.value = null;

  if (fileInput.value) fileInput.value.value = null;
};

const uploadImage = async () => {
  if (!selectedFile.value) return;

  loading.value = true;
  errorMsg.value = "";
  sourceCode.value = "";
  combinedSimilarity.value = null;
  generatedPreview.value = null;

  abortController = new AbortController();
  const formData = new FormData();
  formData.append("file", selectedFile.value);

  try {
    let apiUrl = "";

    if (selectedEngine.value === "gemini") {
      apiUrl = `${import.meta.env.VITE_API_URL}/api/convert/gemini/${selectedColabModel.value}`;
    } else {
      apiUrl = `${import.meta.env.VITE_API_URL}/api/convert/colab/${selectedColabModel.value}`;
    }

    const res = await fetch(apiUrl, {
      method: "POST",
      body: formData,
      signal: abortController.signal
    });

    if (!res.ok) throw new Error("Server error: " + res.status);

    const data = await res.json();

    sourceCode.value =
      data.code ||
      data.html ||
      data.raw_code ||
      data.raw_html ||
      "❌ No code returned from API";

    combinedSimilarity.value = data.combined_similarity ?? null;
    generatedPreview.value = data.generated_preview ?? null;
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("Fetch aborted");
    } else {
      errorMsg.value = "❌ Error: " + err.message;
    }
  } finally {
    loading.value = false;
  }
};

const copyCode = async () => {
  if (!formattedCode.value) return;

  try {
    await navigator.clipboard.writeText(formattedCode.value);
    copyStatus.value = "Copied!";
    setTimeout(() => {
      copyStatus.value = "Copy code";
    }, 2000);
  } catch (err) {
    console.error("Copy failed", err);
  }
};
</script>

<style scoped>
.page {
  min-height: 100vh;
  background-color: #fefefe;
  font-family: 'Segoe UI', sans-serif;
}

.header {
  height: 70px;
  background: linear-gradient(90deg, #8b5e3c, #d8a06f);
  display: flex;
  align-items: center;
  padding: 0 32px;
  color: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.logo img {
  width: 66px;
  height: auto;
}

.title {
  margin: 0 auto;
  font-size: 22px;
  font-style: italic;
}

.content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 80px;
  padding: 48px 64px;
  color: #213555;
  align-items: start;
}

.upload-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.upload-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background-color: #fff;
  border: 2px solid #d0d0d0;
  border-radius: 24px;
  cursor: pointer;
  font-size: 14px;
  color: #333;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.upload-btn:hover {
  border-color: #8b5e3c;
  background: #fffcf9;
}

.disabled-label {
  opacity: 0.5;
  cursor: not-allowed;
}

.icon {
  display: flex;
  align-items: center;
  color: #9e9e9e;
  transition: color 0.2s;
}

.upload-btn:hover .icon {
  color: #666;
}

.engine-dropdown {
  padding: 8px 16px;
  padding-right: 36px;
  border-radius: 24px;
  border: 1px solid #d0d0d0;
  font-size: 14px;
  cursor: pointer;
  background-color: #fff;
  color: #333;
  transition: all 0.2s ease;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%23333' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 16px center;
  background-size: 12px 12px;
}

.engine-dropdown:hover {
  border-color: #8b5e3c;
  background-color: #fffcf9;
}

.engine-dropdown:focus {
  outline: none;
  border-color: #8b5e3c;
  box-shadow: 0 0 5px rgba(139, 94, 60, 0.5);
}

.engine-dropdown:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.generate-btn {
  padding: 6px 12px;
  border: none;
  background: #d5944a;
  color: white;
  border-radius: 8px;
  cursor: pointer;
  font-weight: bold;
  transition: background 0.2s;
  flex-shrink: 0;
}

.generate-btn:hover {
  background: #c0803a;
}

.generate-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.preview {
  border: 2px dashed #ddd;
  padding: 12px;
  border-radius: 12px;
  text-align: center;
  margin-bottom: 16px;
}

.preview img {
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  border-radius: 8px;
  margin-bottom: 8px;
}

.rendered-preview {
  border: 2px dashed #ddd;
  padding: 12px;
  border-radius: 12px;
  text-align: center;
  margin-bottom: 16px;
  background: #fff;
}

.rendered-preview h3 {
  margin: 0 0 10px 0;
  color: #213555;
}

.rendered-preview img {
  width: 100%;
  height: auto;
  border-radius: 8px;
  display: block;
}

.file-info {
  margin-bottom: 8px;
}

.remove-btn {
  padding: 6px 12px;
  border: none;
  background: #ff4d4f;
  color: white;
  border-radius: 8px;
  cursor: pointer;
  font-weight: bold;
  transition: background 0.2s;
}

.remove-btn:hover {
  background: #d9363e;
}

.remove-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error {
  color: red;
  margin-top: 8px;
}

.right-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.score-badge {
  background: #f3e7d8;
  color: #8b5e3c;
  border: 1px solid #d8b38a;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
}

.code-box {
  position: relative;
  width: 100%;
  height: 470px;
  background-color: #1e1e1e;
  color: #dcdcdc;
  border-radius: 12px;
  padding: 16px;
  overflow: hidden;
  font-family: 'Courier New', monospace;
  font-size: 14px;
}

.code-content {
  height: 100%;
  overflow: auto;
  padding-top: 5px;
  box-sizing: border-box;
}

.code-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
  line-height: 1.6;
  tab-size: 2;
}

.copy-btn {
  position: absolute;
  top: 12px;
  right: 40px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 6px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
}

.copy-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.copy-btn img {
  width: 18px;
  height: 18px;
}

.copy-text {
  font-size: 12px;
  color: #4caf50;
  font-weight: bold;
  padding: 0 4px;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-left-color: #8b5e3c;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

.generate-btn svg path {
  stroke: #fff;
}
</style>