# ═══════════════════════════════════════════════════════════════
# TensorCraft-HPC GitHub Pages Makefile
# 本地开发、构建和验证工具
# ═══════════════════════════════════════════════════════════════

.PHONY: help install serve build clean validate deploy

# 默认目标
help: ## 显示帮助信息
	@echo "🚀 TensorCraft-HPC GitHub Pages 管理工具"
	@echo "=========================================="
	@echo ""
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装 Jekyll 依赖
	@echo "📦 安装依赖..."
	bundle install

serve: ## 本地预览站点 (http://localhost:4000)
	@echo "🌐 启动本地服务器..."
	@echo "访问: http://localhost:4000/modern-ai-kernels/"
	bundle exec jekyll serve --livereload --incremental

build: ## 构建站点到 _site 目录
	@echo "🔨 构建站点..."
	JEKYLL_ENV=production bundle exec jekyll build
	@echo "✅ 构建完成! 输出目录: _site/"

clean: ## 清理构建文件
	@echo "🧹 清理构建文件..."
	bundle exec jekyll clean
	rm -rf _site/ .jekyll-cache/ .jekyll-metadata

validate: build ## 验证构建结果
	@echo "🔍 验证构建结果..."
	@echo ""
	@echo "📊 站点统计:"
	@echo "  - HTML 文件数: $$(find _site -name '*.html' | wc -l)"
	@echo "  - CSS 文件数: $$(find _site -name '*.css' | wc -l)"
	@echo "  - JS 文件数: $$(find _site -name '*.js' | wc -l)"
	@echo "  - 图片数: $$(find _site -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.svg' \) | wc -l)"
	@echo ""
	@echo "🔍 检查关键文件:"
	@test -f _site/index.html && echo "  ✅ index.html" || echo "  ❌ index.html 缺失"
	@test -f _site/assets/css/custom.css && echo "  ✅ custom.css" || echo "  ❌ custom.css 缺失"
	@echo ""
	@echo "📝 生成的页面列表:"
	@find _site -name '*.html' | sed 's|_site/|  - |' | head -10

deploy-check: ## 检查部署前的状态
	@echo "🚀 部署前检查..."
	@echo ""
	@echo "📋 当前 Git 状态:"
	@git status --short
	@echo ""
	@echo "📝 最近提交:"
	@git log --oneline -3
	@echo ""
	@echo "🔧 Jekyll 配置检查:"
	@grep -E "^(title|description|url|baseurl):" _config.yml
	@echo ""
	@echo "⚠️  准备部署到 GitHub Pages"

.DEFAULT_GOAL := help
