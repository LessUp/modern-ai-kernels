<script setup lang="ts">
import { ref } from 'vue'

interface Props {
  title: string
  authors: string
  venue?: string
  year?: string
  doi?: string
  url?: string
  arxiv?: string
  tags?: string
  bibtex?: string
}

const props = defineProps<Props>()
const copied = ref(false)

function copyBibTeX() {
  if (props.bibtex) {
    navigator.clipboard.writeText(props.bibtex)
    copied.value = true
    setTimeout(() => copied.value = false, 2000)
  }
}
</script>

<template>
  <div class="tc-citation-card">
    <div class="tc-citation-card-header">
      <h4 class="tc-citation-card-title">{{ title }}</h4>
      <button
        v-if="bibtex"
        class="tc-citation-copy-btn"
        :class="{ copied }"
        @click="copyBibTeX"
        title="Copy BibTeX"
      >
        <span v-if="copied">Copied!</span>
        <span v-else>BibTeX</span>
      </button>
    </div>
    <div class="tc-citation-card-meta">
      <span class="tc-citation-authors">{{ authors }}</span>
      <span v-if="venue" class="tc-citation-venue">{{ venue }}</span>
      <span v-if="year" class="tc-citation-year">{{ year }}</span>
    </div>
    <div class="tc-citation-links">
      <a v-if="doi" :href="`https://doi.org/${doi}`" target="_blank" rel="noopener" class="tc-citation-link">DOI</a>
      <a v-if="arxiv" :href="arxiv" target="_blank" rel="noopener" class="tc-citation-link">arXiv</a>
      <a v-if="url" :href="url" target="_blank" rel="noopener" class="tc-citation-link">URL</a>
    </div>
    <div v-if="tags" class="tc-citation-card-tags">
      <span v-for="tag in tags.split(',').map(t => t.trim()).filter(Boolean)" :key="tag" class="tc-citation-card-tag">{{ tag }}</span>
    </div>
  </div>
</template>

<style scoped>
.tc-citation-card {
  margin: 1.2rem 0;
  padding: 1.1rem 1.3rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--tc-radius-md);
  background: var(--vp-c-bg-alt);
  box-shadow: var(--tc-shadow-sm);
  transition: box-shadow 0.2s ease;
}

.tc-citation-card:hover {
  box-shadow: var(--tc-shadow-md);
}

.tc-citation-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.4rem;
  gap: 0.8rem;
}

.tc-citation-card-title {
  font-family: 'Fraunces', serif;
  font-weight: 700;
  font-size: 1rem;
  color: var(--vp-c-text-1);
  margin: 0;
  line-height: 1.35;
  flex: 1;
}

.tc-citation-copy-btn {
  padding: 0.25rem 0.6rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.7rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
  white-space: nowrap;
}

.tc-citation-copy-btn:hover {
  border-color: var(--tc-accent-primary);
  color: var(--tc-accent-primary);
}

.tc-citation-copy-btn.copied {
  border-color: var(--tc-accent-success);
  color: var(--tc-accent-success);
  background: var(--tc-emerald-soft);
}

.tc-citation-card-meta {
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
  margin-bottom: 0.5rem;
  line-height: 1.5;
}

.tc-citation-authors {
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.tc-citation-venue {
  font-style: italic;
}

.tc-citation-year {
  font-family: var(--vp-font-family-mono);
  font-weight: 600;
}

.tc-citation-links {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.tc-citation-link {
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  background: var(--vp-c-bg-mute);
  color: var(--tc-accent-primary);
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.15s ease;
}

.tc-citation-link:hover {
  background: var(--tc-blue-soft);
  text-decoration: none;
  border-bottom: none;
}

.tc-citation-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.tc-citation-card-tag {
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  font-family: var(--vp-font-family-mono);
  font-weight: 600;
}
</style>
