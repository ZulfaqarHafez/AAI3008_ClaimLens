"""Example usage script for ClaimLens pipeline."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Run a sample verification."""
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with your API keys")
        return
    
    if not os.getenv("TAVILY_API_KEY") and not os.getenv("SERPAPI_KEY"):
        print("Error: No search API key set (TAVILY_API_KEY or SERPAPI_KEY)")
        print("Please create a .env file with your API keys")
        return
    
    # Import after environment is loaded
    from claimlens.graph import run_verification
    
    # Sample text to verify
    sample_text = """
    The Eiffel Tower is 330 meters tall and was completed in 1889. 
    It was designed by Gustave Eiffel and is located in Paris, France.
    The tower attracts approximately 7 million visitors per year.
    The Great Wall of China is visible from space with the naked eye.
    """
    
    print("=" * 60)
    print("ClaimLens - Fact Verification Pipeline")
    print("=" * 60)
    print(f"\nInput Text:\n{sample_text.strip()}\n")
    print("-" * 60)
    print("Running verification...\n")
    
    try:
        # Run verification
        report = run_verification(sample_text)
        
        # Display results
        print("\n" + "=" * 60)
        print("VERIFICATION REPORT")
        print("=" * 60)
        
        print(f"\nOverall Trust Score: {report.overall_trust_score:.1%}")
        print(f"Processing Time: {report.processing_time_seconds:.2f}s")
        print(f"\nSummary: {report.summary}")
        
        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)
        
        for i, result in enumerate(report.verification_results, 1):
            print(f"\n[Claim {i}]")
            print(f"  Text: {result.claim.text}")
            print(f"  Verdict: {result.verdict.value}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Reasoning: {result.reasoning}")
            
            if result.evidence_list:
                print(f"  Evidence sources:")
                for j, evidence in enumerate(result.evidence_list[:3], 1):
                    print(f"    {j}. {evidence.title} ({evidence.url[:50]}...)")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
