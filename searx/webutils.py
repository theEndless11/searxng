import requests
import trafilatura
from urllib.parse import urlparse
from transformers import pipeline
import re
import warnings
import time

# Suppress transformer warnings
warnings.filterwarnings("ignore", message=".*max_length.*max_new_tokens.*")

# Load summarizer once
print("Loading summarization model...")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
print("Model loaded successfully!")

# --- CONFIG ---
SEARXNG_URL = "http://localhost:8888"
MIN_CONTENT_LENGTH = 400
MAX_CONTENT_LENGTH = 10000  # Limit very long articles for processing efficiency

SKIP_DOMAINS = [
    "youtube.com", "reddit.com", "twitter.com", "facebook.com", "instagram.com",
    "tiktok.com", "pinterest.com", "linkedin.com", "quora.com",
    "amazon.com", "ebay.com", "shopify.com", "etsy.com",
    "wikipedia.org",  # Often too long and encyclopedic
    "fandom.com", "wikia.com"
]

# Quality content domains (prioritize these)
QUALITY_DOMAINS = [
    "reuters.com", "bbc.com", "cnn.com", "npr.org", "apnews.com",
    "theguardian.com", "nytimes.com", "wsj.com", "bloomberg.com",
    "techcrunch.com", "arstechnica.com", "wired.com", "theverge.com",
    "nature.com", "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov"
]

def calculate_dynamic_params(word_count, target_compression=0.3):
    """Dynamic parameter calculation based on input length and desired compression"""
    target_words = max(15, int(word_count * target_compression))
    target_tokens = int(target_words * 1.3)
    
    min_tokens = max(10, target_tokens - 15)
    max_tokens = target_tokens + 25
    
    return {
        "max_new_tokens": max_tokens,
        "min_length": min_tokens,
        "target_words": target_words
    }

def generate_summary_flexible(text, target_compression=0.3, max_attempts=3):
    """Generate summary with flexible parameter adjustment"""
    word_count = len(text.split())
    
    for attempt in range(max_attempts):
        try:
            adjusted_compression = target_compression + (attempt * 0.1)
            params = calculate_dynamic_params(word_count, adjusted_compression)
            
            result = summarizer(
                text,
                max_new_tokens=params["max_new_tokens"],
                min_length=params["min_length"],
                do_sample=False,
                truncation=True,
                early_stopping=False,
                no_repeat_ngram_size=2
            )
            
            summary = result[0]['summary_text']
            
            if summary.strip().endswith(('.', '!', '?')):
                return summary, params["target_words"]
            
        except Exception as e:
            continue
    
    # Fallback
    try:
        fallback_params = {
            "max_new_tokens": max(30, word_count // 2),
            "min_length": 15
        }
        result = summarizer(
            text,
            max_new_tokens=fallback_params["max_new_tokens"],
            min_length=fallback_params["min_length"],
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text'], fallback_params["max_new_tokens"] // 1.3
    except:
        return "Unable to generate summary.", 0

def improve_summary_advanced(summary):
    """Advanced post-processing for web content"""
    if not summary or len(summary.strip()) == 0:
        return "Unable to generate summary."
    
    original_summary = summary
    
    # Clean up common web article patterns
    summary = re.sub(r'\s+', ' ', summary)
    
    # Remove meta-commentary and web-specific fluff
    web_patterns = [
        r'\bthis (article|story|report|piece)\b.*?(\.|,)',
        r'\b(according to|as reported by|sources say)\b.*?(\.|,)',
        r'\bin (this|the) (article|story|report)\b.*?(\.|,)',
        r'\bthe (author|reporter|writer) (says?|writes?|reports?)\b.*?(\.|,)',
        r'\b(click here|read more|continue reading|full story)\b.*?(\.|,)',
        r'\b(updated|published|posted|edited):\s*\d+.*?(\.|,)',
        r'\b(tags?|categories?):\s*.*?(\.|,)',
        r'\boverall\b,?\s*',
        r'\bin conclusion\b,?\s*',
        r'\bin summary\b,?\s*',
        r'\bto summarize\b,?\s*'
    ]
    
    for pattern in web_patterns:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
    
    # Fix incomplete sentences and trailing connectors
    summary = re.sub(r'\s+(and|or|but|with|in|of|to|for|by|while|as|that|which)\s*\.?$', '.', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\s+\w{1,2}\.?$', '.', summary)
    
    # Remove duplicate phrases common in web content
    summary = re.sub(r'\b(\w+(?:\s+\w+){0,4})\s+\1\b', r'\1', summary, flags=re.IGNORECASE)
    
    # Proper capitalization and punctuation
    summary = summary.strip()
    if summary:
        summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
    
    # Final cleanup
    summary = re.sub(r'\s+([.!?])', r'\1', summary)
    summary = re.sub(r'([.!?])\s+', r'\1 ', summary)
    summary = re.sub(r'\s+', ' ', summary)
    
    # Quality check
    if len(summary.split()) < 8 and len(original_summary.split()) > 15:
        return original_summary.strip()
    
    return summary.strip()

def assess_content_quality(text, url):
    """Assess the quality of extracted content"""
    score = 0
    notes = []
    
    # Length check
    word_count = len(text.split())
    if word_count > MIN_CONTENT_LENGTH:
        score += 2
    if word_count > 1000:
        score += 1
    
    # Domain quality
    domain = urlparse(url).netloc.lower()
    if any(quality in domain for quality in QUALITY_DOMAINS):
        score += 3
        notes.append("Quality source domain")
    
    # Content quality indicators
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences > 5:
        score += 1
    
    # Check for common low-quality indicators
    low_quality_indicators = [
        "subscribe to our newsletter",
        "click here for more",
        "advertisement",
        "sponsored content",
        "this page requires javascript"
    ]
    
    if any(indicator in text.lower() for indicator in low_quality_indicators):
        score -= 1
        notes.append("Contains promotional content")
    
    return score, notes

def fetch_search_results(query, num_results=10):
    """Fetch search results from SearXNG"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ContentSummarizer/1.0)"}
    search_url = f"{SEARXNG_URL}/search"
    
    params = {
        "q": query,
        "format": "json",
        "safesearch": "0",
        "time_range": "month"  # Focus on recent content
    }
    
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        # Filter and sort results
        valid_results = []
        for r in results[:num_results * 2]:  # Get extra to filter
            url = r.get("url", "")
            if url and not any(bad in url.lower() for bad in SKIP_DOMAINS):
                valid_results.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "content": r.get("content", "")
                })
        
        # Prioritize quality domains
        quality_results = [r for r in valid_results if any(q in r["url"].lower() for q in QUALITY_DOMAINS)]
        other_results = [r for r in valid_results if not any(q in r["url"].lower() for q in QUALITY_DOMAINS)]
        
        return (quality_results + other_results)[:num_results]
    
    except Exception as e:
        print(f"Error fetching search results: {e}")
        return []

def extract_content(url, timeout=10):
    """Extract clean content from URL"""
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            return None
        
        # Extract with optimized settings for summarization
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,  # Keep tables for data richness
            include_images=False,
            include_links=False,
            only_with_metadata=False,
            output_format="txt",
            target_language="en"
        )
        
        if text and len(text.strip()) > MIN_CONTENT_LENGTH:
            # Truncate if too long for efficiency
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "..."
            return text.strip()
        
        return None
    
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def search_and_summarize(query, target_compression=0.3, max_articles=5):
    """Main function to search, extract, and summarize content"""
    print(f"🔍 Searching for: '{query}'")
    print(f"📡 Using SearXNG at: {SEARXNG_URL}")
    print("-" * 60)
    
    # Fetch search results
    results = fetch_search_results(query, num_results=max_articles * 2)
    if not results:
        print("❌ No search results found")
        return []
    
    print(f"📋 Found {len(results)} potential sources")
    
    summaries = []
    processed = 0
    
    for i, result in enumerate(results):
        if processed >= max_articles:
            break
            
        url = result["url"]
        title = result["title"]
        
        print(f"\n📄 Processing [{i+1}/{len(results)}]: {title[:60]}...")
        print(f"🌐 URL: {url}")
        
        # Extract content
        content = extract_content(url)
        if not content:
            print("❌ Failed to extract content")
            continue
        
        # Assess content quality
        quality_score, quality_notes = assess_content_quality(content, url)
        word_count = len(content.split())
        
        print(f"📊 Content: {word_count} words, Quality score: {quality_score}/5")
        if quality_notes:
            print(f"📝 Notes: {', '.join(quality_notes)}")
        
        # Skip low-quality content
        if quality_score < 2:
            print("❌ Content quality too low, skipping")
            continue
        
        # Generate summary
        print("🤖 Generating summary...")
        try:
            raw_summary, target_words = generate_summary_flexible(content, target_compression)
            final_summary = improve_summary_advanced(raw_summary)
            
            summary_info = {
                "title": title,
                "url": url,
                "summary": final_summary,
                "word_count": word_count,
                "summary_words": len(final_summary.split()),
                "quality_score": quality_score,
                "compression_ratio": len(final_summary.split()) / word_count
            }
            
            summaries.append(summary_info)
            processed += 1
            
            print(f"✅ Summary generated ({len(final_summary.split())} words)")
            
        except Exception as e:
            print(f"❌ Summarization failed: {e}")
            continue
        
        # Brief pause to avoid overwhelming servers
        time.sleep(0.5)
    
    return summaries

def display_results(summaries, query):
    """Display formatted results"""
    if not summaries:
        print("\n❌ No summaries generated")
        return
    
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY RESULTS FOR: '{query.upper()}'")
    print(f"{'='*80}")
    
    for i, summary in enumerate(summaries, 1):
        print(f"\n📄 [{i}] {summary['title']}")
        print(f"🌐 {summary['url']}")
        print(f"📊 {summary['word_count']} words → {summary['summary_words']} words ({summary['compression_ratio']:.1%} compression)")
        print(f"⭐ Quality Score: {summary['quality_score']}/5")
        print(f"\n📝 SUMMARY:")
        print(f"{'-'*60}")
        print(summary['summary'])
        print(f"{'-'*60}")

if __name__ == "__main__":
    print("🚀 SearXNG Content Fetcher & Summarizer")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 Enter search query (or 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                print("👋 Goodbye!")
                break
            
            if not query:
                print("❌ Please enter a search query")
                continue
            
            # Optional parameters
            compression_input = input("🎚️ Target compression (0.2-0.5, default 0.3): ").strip()
            try:
                target_compression = float(compression_input) if compression_input else 0.3
                target_compression = max(0.2, min(0.5, target_compression))
            except:
                target_compression = 0.3
            
            articles_input = input("📚 Max articles to process (1-10, default 5): ").strip()
            try:
                max_articles = int(articles_input) if articles_input else 5
                max_articles = max(1, min(10, max_articles))
            except:
                max_articles = 5
            
            print(f"\n🎯 Configuration:")
            print(f"   • Query: {query}")
            print(f"   • Compression: {target_compression:.1%}")
            print(f"   • Max articles: {max_articles}")
            
            # Run the search and summarization
            summaries = search_and_summarize(query, target_compression, max_articles)
            display_results(summaries, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue