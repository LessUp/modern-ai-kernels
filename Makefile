# ═══════════════════════════════════════════════════════════════
# TensorCraft-HPC Makefile
# 构建开发、测试和文档工具
# ═══════════════════════════════════════════════════════════════

.PHONY: help install serve build clean validate deploy \
        configure build-dev build-release test benchmark lint format \
        python-install python-test ccache-clean

# 默认目标
help: ## 显示帮助信息
	@echo "🚀 TensorCraft-HPC 管理工具"
	@echo "=========================================="
	@echo ""
	@echo "CUDA 开发命令:"
	@grep -E '^(configure|build-dev|build-release|test|benchmark|lint|format):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Python 开发命令:"
	@grep -E '^(python-):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "文档命令:"
	@grep -E '^(install|serve|build|clean|validate|deploy):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════
# CUDA 开发命令
# ═══════════════════════════════════════════════════════════════

configure: ## 配置开发构建 (使用 preset)
	@echo "⚙️  配置开发构建..."
	cmake --preset dev

build-dev: ## 构建开发版本
	@echo "🔨 构建开发版本..."
	cmake --build --preset dev --parallel $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

build-release: ## 构建发布版本
	@echo "🚀 构建发布版本..."
	cmake --preset release
	cmake --build --preset release --parallel $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

test: ## 运行单元测试
	@echo "🧪 运行单元测试..."
	ctest --preset dev --output-on-failure

benchmark: build-release ## 运行性能基准测试
	@echo "📊 运行性能基准测试..."
	./build/release/benchmarks/gemm_benchmark --benchmark_time_unit=ms
	./build/release/benchmarks/attention_benchmark --benchmark_time_unit=ms
	./build/release/benchmarks/conv_benchmark --benchmark_time_unit=ms

lint: ## 运行代码静态检查
	@echo "🔍 运行静态检查..."
	pre-commit run --all-files

format: ## 格式化代码
	@echo "✨ 格式化代码..."
	find include src tests examples -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
	black tests/*.py

# ═══════════════════════════════════════════════════════════════
# Python 开发命令
# ═══════════════════════════════════════════════════════════════

python-install: ## 安装 Python 绑定 (可编辑模式)
	@echo "📦 安装 Python 绑定..."
	pip install -e .

python-test: python-install ## 运行 Python 测试
	@echo "🧪 运行 Python 测试..."
	python -m pytest tests/test_python_bindings.py -v

# ═══════════════════════════════════════════════════════════════
# 文档命令
# ═══════════════════════════════════════════════════════════════

install: ## 安装 Jekyll 依赖
	@echo "📦 安装 Jekyll 依赖..."
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
	rm -rf _site/ .jekyll-cache/ .jekyll-metadata build/

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

ccache-clean: ## 清理 ccache 缓存
	@echo "🧹 清理 ccache 缓存..."
	ccache -C 2>/dev/null || true

.DEFAULT_GOAL := help
