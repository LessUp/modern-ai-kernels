# ═══════════════════════════════════════════════════════════════
# TensorCraft-HPC GitHub Pages Gemfile
# 用于本地开发和测试 Jekyll 站点
# ═══════════════════════════════════════════════════════════════

source "https://rubygems.org"

# 使用 GitHub Pages  gem
gem "github-pages", group: :jekyll_plugins

# 本地开发插件
group :jekyll_plugins do
  gem "jekyll-remote-theme"
  gem "jekyll-seo-tag"
  gem "jekyll-github-metadata"
  gem "jekyll-include-cache"
end

# Windows 和 JRuby 依赖排除
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Windows 性能改进
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# 监听文件变更 (JRuby 不支持)
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# 我们brick作为本地服务器 (Ruby 3.0+)
gem "webrick", "~> 1.8"
