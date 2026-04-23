// ═══════════════════════════════════════════════════════════════
// TENSORCRAFT-HPC - Documentation Page JavaScript
// ═══════════════════════════════════════════════════════════════

(function() {
  'use strict';

  // ─── Mobile Menu Toggle ────────────────────────────────────────
  function initMobileMenu() {
    const toggle = document.getElementById('mobile-menu-toggle');
    const sidebar = document.getElementById('docs-sidebar');
    
    if (!toggle || !sidebar) return;
    
    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('active');
      toggle.classList.toggle('active');
    });
    
    // Close sidebar when clicking on a link
    sidebar.querySelectorAll('a').forEach(function(link) {
      link.addEventListener('click', function() {
        sidebar.classList.remove('active');
        toggle.classList.remove('active');
      });
    });
  }

  // ─── Generate Table of Contents ────────────────────────────────
  function initTableOfContents() {
    const content = document.querySelector('.content-body');
    const tocNav = document.getElementById('toc-nav');
    
    if (!content || !tocNav) return;
    
    const headings = content.querySelectorAll('h2, h3');
    if (headings.length < 3) {
      tocNav.parentElement.style.display = 'none';
      return;
    }
    
    const tocList = document.createElement('ul');
    
    headings.forEach(function(heading) {
      // Create ID if not exists
      if (!heading.id) {
        heading.id = heading.textContent.toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-');
      }
      
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + heading.id;
      a.textContent = heading.textContent;
      a.dataset.target = heading.id;
      
      if (heading.tagName === 'H3') {
        const subList = document.createElement('ul');
        const subLi = document.createElement('li');
        subLi.appendChild(a);
        subList.appendChild(subLi);
        
        // Find the last H2's li and append subList
        const lastH2Li = tocList.querySelector('li:last-child');
        if (lastH2Li) {
          lastH2Li.appendChild(subList);
        }
      } else {
        li.appendChild(a);
        tocList.appendChild(li);
      }
    });
    
    tocNav.appendChild(tocList);
    
    // Smooth scroll for TOC links
    tocList.querySelectorAll('a').forEach(function(link) {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.getElementById(this.dataset.target);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          history.pushState(null, null, '#' + this.dataset.target);
        }
      });
    });
    
    // Highlight active section
    highlightActiveSection(tocList);
  }

  // ─── Highlight Active Section in TOC ───────────────────────────
  function highlightActiveSection(tocList) {
    const headings = document.querySelectorAll('.content-body h2, .content-body h3');
    const tocLinks = tocList.querySelectorAll('a');
    
    if (!headings.length) return;
    
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          
          tocLinks.forEach(function(link) {
            link.classList.remove('active');
            if (link.dataset.target === id) {
              link.classList.add('active');
            }
          });
        }
      });
    }, {
      rootMargin: '-80px 0px -80% 0px',
      threshold: 0
    });
    
    headings.forEach(function(heading) {
      observer.observe(heading);
    });
  }

  // ─── Copy Code Blocks ──────────────────────────────────────────
  function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(function(pre) {
      const button = document.createElement('button');
      button.className = 'code-copy-btn';
      button.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
      button.setAttribute('aria-label', 'Copy code');
      
      // Position relative to parent pre
      pre.style.position = 'relative';
      
      button.addEventListener('click', function() {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        
        navigator.clipboard.writeText(text).then(function() {
          button.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';
          button.style.color = '#10B981';
          
          setTimeout(function() {
            button.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
            button.style.color = '';
          }, 2000);
        });
      });
      
      pre.appendChild(button);
    });
  }

  // ─── Initialize Everything ─────────────────────────────────────
  function init() {
    initMobileMenu();
    initTableOfContents();
    initCodeCopy();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
