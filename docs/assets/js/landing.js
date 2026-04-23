// ═══════════════════════════════════════════════════════════════
// TENSORCRAFT-HPC - Landing Page JavaScript
// ═══════════════════════════════════════════════════════════════

(function() {
  'use strict';

  // ─── Navigation Scroll Effect ──────────────────────────────────
  function initNavigation() {
    const nav = document.getElementById('landing-nav');
    if (!nav) return;

    function updateNav() {
      if (window.scrollY > 50) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }
    }

    window.addEventListener('scroll', updateNav, { passive: true });
    updateNav();
  }

  // ─── Mobile Navigation Toggle ──────────────────────────────────
  function initMobileNav() {
    const toggle = document.getElementById('nav-toggle');
    const links = document.getElementById('nav-links');
    
    if (!toggle || !links) return;
    
    toggle.addEventListener('click', function() {
      links.classList.toggle('active');
      toggle.classList.toggle('active');
    });
    
    // Close menu when clicking on a link
    links.querySelectorAll('a').forEach(function(link) {
      link.addEventListener('click', function() {
        links.classList.remove('active');
        toggle.classList.remove('active');
      });
    });
  }

  // ─── Code Tab Switching ────────────────────────────────────────
  function initCodeTabs() {
    const tabs = document.querySelectorAll('.code-tab');
    const panels = document.querySelectorAll('.code-panel');
    
    tabs.forEach(function(tab) {
      tab.addEventListener('click', function() {
        const lang = this.dataset.lang;
        
        // Update tabs
        tabs.forEach(function(t) { t.classList.remove('active'); });
        this.classList.add('active');
        
        // Update panels
        panels.forEach(function(p) {
          p.classList.remove('active');
          if (p.dataset.lang === lang) {
            p.classList.add('active');
          }
        });
      });
    });
  }

  // ─── Copy to Clipboard ─────────────────────────────────────────
  function initCopyButtons() {
    const buttons = document.querySelectorAll('.copy-btn');
    
    buttons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        const text = this.dataset.clipboard;
        
        navigator.clipboard.writeText(text).then(function() {
          // Visual feedback
          const originalHTML = btn.innerHTML;
          btn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';
          btn.style.color = '#10B981';
          
          setTimeout(function() {
            btn.innerHTML = originalHTML;
            btn.style.color = '';
          }, 2000);
        });
      });
    });
  }

  // ─── Performance Bar Animation ─────────────────────────────────
  function initPerformanceBars() {
    const bars = document.querySelectorAll('.perf-bar-fill');
    
    if (!bars.length) return;
    
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          const bar = entry.target;
          const width = bar.style.getPropertyValue('--target-width');
          bar.style.width = width;
          observer.unobserve(bar);
        }
      });
    }, { threshold: 0.5 });
    
    bars.forEach(function(bar) {
      bar.style.width = '0';
      observer.observe(bar);
    });
  }

  // ─── Scroll Animations ─────────────────────────────────────────
  function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.feature-card, .perf-card, .gpu-card, .step, .why-card, .community-card');
    
    if (!animatedElements.length) return;
    
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry, index) {
        if (entry.isIntersecting) {
          setTimeout(function() {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
          }, index * 50);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    
    animatedElements.forEach(function(el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      observer.observe(el);
    });
  }

  // ─── Smooth Scroll for Anchor Links ──────────────────────────
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
      anchor.addEventListener('click', function(e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });
  }

  // ─── Initialize Everything ─────────────────────────────────────
  function init() {
    initNavigation();
    initMobileNav();
    initCodeTabs();
    initCopyButtons();
    initPerformanceBars();
    initScrollAnimations();
    initSmoothScroll();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
