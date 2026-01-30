import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Literal, Optional
from pydantic import BaseModel
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv()

client = Groq()
MODEL = "moonshotai/kimi-k2-instruct-0905"

# === Rate Limit Configuration ===
RATE_LIMITS = {
    "rpm": 60,           # Requests per minute
    "rpd": 1000,         # Requests per day
    "tpm": 10_000,       # Tokens per minute
    "tpd": 300_000,      # Tokens per day
}

# Safety margins (use 90% of limits to avoid edge cases)
SAFE_RPM = int(RATE_LIMITS["rpm"] * 0.9)  # 54 requests/min
SAFE_RPD = int(RATE_LIMITS["rpd"] * 0.9)  # 900 requests/day
MIN_DELAY_BETWEEN_REQUESTS = 60 / SAFE_RPM  # ~1.1 seconds


# === Pydantic Schema for Structured Output ===

class TweetAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    sentiment_score: float  # -1 to +1
    crypto_tokens: list[str]  # e.g., ["BTC", "DOGE", "ETH"]
    event_type: Literal[
        "payment_adoption",
        "partnership",
        "meme_joke",
        "market_commentary",
        "regulatory",
        "airdrop",
        "product_launch",
        "endorsement",
        "criticism",
        "none"
    ]
    has_market_signal: bool
    market_signal: Literal["buy", "sell", "hold", "none"]
    stance_on_original: Optional[Literal["agrees", "disagrees", "mocking", "amplifying", "neutral"]]


# === Rate Limiter Class ===

class RateLimiter:
    """
    Handles rate limiting with awareness of daily resets at midnight.
    Optimized for running at ~11pm to maximize quota usage across 2 days.
    """

    def __init__(self, state_file: str = "rate_limit_state.json"):
        self.state_file = state_file
        self.requests_this_minute = 0
        self.requests_today = 0
        self.minute_start = time.time()
        self.day_start = self._get_day_start()
        self._load_state()

    def _get_day_start(self) -> datetime:
        """Get the start of the current day (midnight)."""
        now = datetime.now()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _load_state(self):
        """Load rate limit state from file (for resuming runs)."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                saved_day = datetime.fromisoformat(state.get('day_start', ''))
                current_day = self._get_day_start()

                # If same day, restore count; otherwise reset
                if saved_day.date() == current_day.date():
                    self.requests_today = state.get('requests_today', 0)
                    print(f"Resumed: {self.requests_today} requests already made today")
                else:
                    self.requests_today = 0
                    print("New day detected, starting fresh")
            except (json.JSONDecodeError, KeyError, ValueError):
                self.requests_today = 0

    def _save_state(self):
        """Save current state for resumption."""
        state = {
            'day_start': self._get_day_start().isoformat(),
            'requests_today': self.requests_today,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def _check_day_reset(self):
        """Check if we've crossed midnight and reset daily counter."""
        current_day_start = self._get_day_start()
        if current_day_start > self.day_start:
            old_count = self.requests_today
            self.requests_today = 0
            self.day_start = current_day_start
            print(f"\n🌙 MIDNIGHT RESET! Daily quota refreshed (was {old_count} requests)")
            return True
        return False

    def _check_minute_reset(self):
        """Reset minute counter if a minute has passed."""
        elapsed = time.time() - self.minute_start
        if elapsed >= 60:
            self.requests_this_minute = 0
            self.minute_start = time.time()

    def wait_if_needed(self) -> dict:
        """
        Wait if rate limits would be exceeded.
        Returns status info about the wait.
        """
        self._check_day_reset()
        self._check_minute_reset()

        status = {
            'waited': False,
            'wait_seconds': 0,
            'reason': None,
            'requests_today': self.requests_today,
            'requests_remaining_today': SAFE_RPD - self.requests_today
        }

        # Check daily limit
        if self.requests_today >= SAFE_RPD:
            # Calculate time until midnight
            now = datetime.now()
            midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            wait_seconds = (midnight - now).total_seconds()

            print(f"\n⏸️  Daily limit reached ({self.requests_today} requests)")
            print(f"   Waiting until midnight ({wait_seconds/3600:.1f} hours)...")
            print(f"   Resume time: {midnight.strftime('%Y-%m-%d %H:%M:%S')}")

            # Wait until midnight + small buffer
            time.sleep(wait_seconds + 5)
            self._check_day_reset()

            status['waited'] = True
            status['wait_seconds'] = wait_seconds
            status['reason'] = 'daily_limit'

        # Check per-minute limit
        if self.requests_this_minute >= SAFE_RPM:
            wait_seconds = 60 - (time.time() - self.minute_start) + 1
            if wait_seconds > 0:
                print(f"\n⏳ Minute limit reached, waiting {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)
                self._check_minute_reset()
                status['waited'] = True
                status['wait_seconds'] = wait_seconds
                status['reason'] = 'minute_limit'

        # Standard delay between requests
        time.sleep(MIN_DELAY_BETWEEN_REQUESTS)

        return status

    def record_request(self):
        """Record that a request was made."""
        self.requests_this_minute += 1
        self.requests_today += 1
        self._save_state()

    def get_status(self) -> str:
        """Get human-readable status."""
        remaining_today = SAFE_RPD - self.requests_today
        return (
            f"Today: {self.requests_today}/{SAFE_RPD} "
            f"(~{remaining_today} remaining) | "
            f"This minute: {self.requests_this_minute}/{SAFE_RPM}"
        )


# === System Prompt ===
SYSTEM_PROMPT = """You are analyzing Elon Musk's tweets for cryptocurrency market signals.

Extract the following as JSON:
- sentiment: Overall sentiment toward crypto (positive/negative/neutral)
- sentiment_score: Intensity from -1 (very negative) to +1 (very positive)
- crypto_tokens: List of cryptocurrency tickers mentioned or implied (e.g., BTC, DOGE, ETH). Use standard tickers.
- event_type: Categorize the tweet:
  - payment_adoption: Accepting crypto as payment
  - partnership: Business collaboration related to crypto
  - meme_joke: Humorous/meme content about crypto
  - market_commentary: Opinion on prices/market
  - regulatory: Government/regulation related
  - airdrop: Token giveaway
  - product_launch: New crypto product/feature
  - endorsement: Explicit support for a token
  - criticism: Negative view of a token/project
  - none: Doesn't fit other categories
- has_market_signal: Does the tweet suggest any trading action? (true/false)
- market_signal: If has_market_signal is true, what action? (buy/sell/hold/none)
- stance_on_original: ONLY for quote tweets (text starting with "[Quoting"). How does Musk's comment relate to the quoted tweet? (agrees/disagrees/mocking/amplifying/neutral). Set to null for non-quote tweets.

Respond ONLY with valid JSON matching this exact schema."""


# === LLM Extraction Function ===

def extract_features(
    text: str,
    is_quote: bool,
    rate_limiter: RateLimiter,
    max_retries: int = 5
) -> dict:
    """
    Extract features from a single tweet using LLM.

    Args:
        text: The combined_text of the tweet
        is_quote: Whether this is a quote tweet
        rate_limiter: RateLimiter instance
        max_retries: Number of retries on failure

    Returns:
        Dictionary of extracted features
    """
    for attempt in range(max_retries):
        try:
            # Wait for rate limits
            rate_limiter.wait_if_needed()

            # Build user message with explicit JSON instruction
            user_message = f"""Analyze this tweet and respond with JSON only:

Tweet: {text}

Is quote tweet: {is_quote}

Respond with this JSON structure:
{{
    "sentiment": "positive" or "negative" or "neutral",
    "sentiment_score": number between -1 and 1,
    "crypto_tokens": ["TOKEN1", "TOKEN2"],
    "event_type": "one of the categories",
    "has_market_signal": true or false,
    "market_signal": "buy" or "sell" or "hold" or "none",
    "stance_on_original": "agrees/disagrees/mocking/amplifying/neutral" or null
}}"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500    # Limit response size
            )

            # Record successful request
            rate_limiter.record_request()

            # Parse response
            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result_dict = json.loads(content)

            # Validate with Pydantic
            result = TweetAnalysis.model_validate(result_dict)
            result_dict = result.model_dump()

            # Set stance_on_original to None if not a quote tweet
            if not is_quote:
                result_dict['stance_on_original'] = None

            return result_dict

        except RateLimitError as e:
            print(f"\n⚠️  Rate limit error from API: {e}")
            # Wait longer on rate limit errors
            wait_time = 60 * (attempt + 1)
            print(f"   Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue

        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

        except Exception as e:
            print(f"\n⚠️  Error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

    # Return None values after all retries failed
    print(f"❌ Failed after {max_retries} attempts")
    return {
        'sentiment': None,
        'sentiment_score': None,
        'crypto_tokens': None,
        'event_type': None,
        'has_market_signal': None,
        'market_signal': None,
        'stance_on_original': None
    }


def estimate_runtime(num_tweets: int, requests_remaining: int) -> str:
    """Estimate how long processing will take."""
    # Time per request (with rate limiting)
    time_per_request = MIN_DELAY_BETWEEN_REQUESTS + 0.5  # Add buffer for API response

    if num_tweets <= requests_remaining:
        # Can complete in current quota
        total_seconds = num_tweets * time_per_request
        hours = total_seconds / 3600
        return f"~{hours:.1f} hours (fits in current quota)"
    else:
        # Will need to wait for midnight
        first_batch_time = requests_remaining * time_per_request
        remaining_tweets = num_tweets - requests_remaining
        days_needed = (remaining_tweets // SAFE_RPD) + 1
        return f"~{days_needed} day(s) - will pause at midnight for quota reset"


def process_crypto_tweets(
    df: pd.DataFrame,
    output_path: str = "musk_tweets_llm_features.csv",
    batch_save_interval: int = 50,
    resume: bool = True
) -> pd.DataFrame:
    """
    Process all crypto-related tweets and extract LLM features.

    Args:
        df: DataFrame with is_crypto_related column
        output_path: Path to save results
        batch_save_interval: Save progress every N tweets
        resume: Whether to resume from previous run

    Returns:
        DataFrame with LLM features added
    """
    # Initialize rate limiter
    rate_limiter = RateLimiter()

    # Filter to crypto-related tweets only
    crypto_df = df[df['is_crypto_related'] == True].copy()

    # Initialize new columns
    new_columns = [
        'sentiment', 'sentiment_score', 'crypto_tokens',
        'event_type', 'has_market_signal', 'market_signal', 'stance_on_original'
    ]
    for col in new_columns:
        if col not in crypto_df.columns:
            crypto_df[col] = None

    # Check for existing progress
    start_idx = 0
    if resume and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # Find how many are already processed
        processed_count = existing_df['sentiment'].notna().sum()
        if processed_count > 0:
            print(f"📂 Found existing progress: {processed_count} tweets already processed")
            crypto_df = existing_df
            start_idx = processed_count

    total_tweets = len(crypto_df)
    remaining_tweets = total_tweets - start_idx

    print(f"\n{'='*60}")
    print(f"CRYPTO TWEET PROCESSING - Kimi K2 Model")
    print(f"{'='*60}")
    print(f"Total crypto tweets: {total_tweets:,}")
    print(f"Already processed: {start_idx:,}")
    print(f"Remaining: {remaining_tweets:,}")
    print(f"\nRate limits: {SAFE_RPM} req/min, {SAFE_RPD} req/day")
    print(f"Current status: {rate_limiter.get_status()}")
    print(f"\nEstimated time: {estimate_runtime(remaining_tweets, SAFE_RPD - rate_limiter.requests_today)}")
    print(f"{'='*60}\n")

    # Process each tweet
    processed_in_session = 0
    failed_count = 0

    for idx, (row_idx, row) in enumerate(tqdm(
        list(crypto_df.iterrows())[start_idx:],
        total=remaining_tweets,
        desc="Processing tweets"
    )):
        # Skip if already processed
        if pd.notna(row.get('sentiment')):
            continue

        features = extract_features(
            text=row['combined_text'],
            is_quote=row['isQuote'],
            rate_limiter=rate_limiter
        )

        for col, value in features.items():
            crypto_df.at[row_idx, col] = value if not isinstance(value, list) else json.dumps(value)

        processed_in_session += 1
        if features['sentiment'] is None:
            failed_count += 1

        # Batch save progress
        if processed_in_session % batch_save_interval == 0:
            crypto_df.to_csv(output_path, index=False)
            print(f"\n💾 Progress saved: {start_idx + processed_in_session}/{total_tweets}")
            print(f"   {rate_limiter.get_status()}")

    # Final save
    crypto_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("LLM EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed this session: {processed_in_session}")
    print(f"Failed extractions: {failed_count}")

    print(f"\nSentiment distribution:")
    print(crypto_df['sentiment'].value_counts().to_string())

    print(f"\nEvent type distribution:")
    print(crypto_df['event_type'].value_counts().to_string())

    print(f"\nMarket signals:")
    has_signal = crypto_df['has_market_signal'].apply(
        lambda x: x == True or x == 'True' or x == 'true'
    ).sum()
    print(f"  Has signal: {has_signal}")
    print(crypto_df['market_signal'].value_counts().to_string())

    # Top tokens
    def parse_tokens(x):
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                return []
        return x if isinstance(x, list) else []

    all_tokens = [t for tokens in crypto_df['crypto_tokens'].apply(parse_tokens) for t in tokens]
    if all_tokens:
        print(f"\nTop crypto tokens mentioned:")
        token_counts = pd.Series(all_tokens).value_counts().head(10)
        print(token_counts.to_string())

    return crypto_df