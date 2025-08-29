"""
Load Testing Script for Model API
=================================

This script performs load testing on the model API to measure performance
under various loads and configurations.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import json
import argparse

class LoadTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, prompt: str, max_length: int = 50) -> Dict:
        """Make a single generation request to the API"""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": 0.7,
                    "do_sample": True
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "generated_text": data.get("generated_text", ""),
                        "tokens_per_second": data.get("tokens_per_second", 0)
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
        except asyncio.TimeoutError:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "status_code": 0,
                "error": "Timeout"
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "status_code": 0,
                "error": str(e)
            }
    
    async def run_concurrent_requests(self, prompts: List[str], concurrency: int) -> List[Dict]:
        """Run multiple requests concurrently"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(prompt):
            async with semaphore:
                return await self.make_request(prompt)
        
        tasks = [bounded_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Exception occurred: {result}")
                continue
            valid_results.append(result)
        
        return valid_results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate performance statistics from results"""
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if not successful_requests:
            return {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "median_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
                "avg_tokens_per_second": 0.0
            }
        
        response_times = [r["response_time"] for r in successful_requests]
        tokens_per_second = [r["tokens_per_second"] for r in successful_requests if r["tokens_per_second"] > 0]
        
        stats = {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(results) * 100,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_tokens_per_second": statistics.mean(tokens_per_second) if tokens_per_second else 0.0
        }
        
        return stats
    
    async def run_load_test(self, test_config: Dict) -> Dict:
        """Run a load test with the given configuration"""
        print(f"Running load test: {test_config['name']}")
        print(f"Concurrency: {test_config['concurrency']}, Requests: {test_config['requests']}")
        
        # Generate test prompts
        prompts = [
            f"Explain the concept of {test_config['topic']} in detail."
            for i in range(test_config['requests'])
        ]
        
        start_time = time.time()
        results = await self.run_concurrent_requests(prompts, test_config['concurrency'])
        end_time = time.time()
        
        total_time = end_time - start_time
        stats = self.calculate_statistics(results)
        
        # Add overall statistics
        stats["total_test_time"] = total_time
        stats["requests_per_second"] = len(results) / total_time if total_time > 0 else 0
        stats["test_name"] = test_config['name']
        
        print(f"Test completed in {total_time:.2f} seconds")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        print(f"Average response time: {stats['avg_response_time']:.3f}s")
        print(f"Requests per second: {stats['requests_per_second']:.2f}")
        print("---")
        
        return stats

async def main():
    parser = argparse.ArgumentParser(description='Load test the model API')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the API')
    parser.add_argument('--test-suite', choices=['quick', 'standard', 'comprehensive'], 
                       default='standard', help='Test suite to run')
    
    args = parser.parse_args()
    
    test_suites = {
        "quick": [
            {"name": "Low Concurrency Test", "concurrency": 2, "requests": 10, "topic": "machine learning"},
            {"name": "Medium Concurrency Test", "concurrency": 5, "requests": 20, "topic": "artificial intelligence"}
        ],
        "standard": [
            {"name": "Low Concurrency Test", "concurrency": 2, "requests": 20, "topic": "machine learning"},
            {"name": "Medium Concurrency Test", "concurrency": 5, "requests": 50, "topic": "artificial intelligence"},
            {"name": "High Concurrency Test", "concurrency": 10, "requests": 100, "topic": "deep learning"}
        ],
        "comprehensive": [
            {"name": "Low Concurrency Test", "concurrency": 2, "requests": 50, "topic": "machine learning"},
            {"name": "Medium Concurrency Test", "concurrency": 5, "requests": 100, "topic": "artificial intelligence"},
            {"name": "High Concurrency Test", "concurrency": 10, "requests": 200, "topic": "deep learning"},
            {"name": "Very High Concurrency Test", "concurrency": 20, "requests": 100, "topic": "neural networks"}
        ]
    }
    
    async with LoadTester(args.url) as tester:
        test_configs = test_suites[args.test_suite]
        all_stats = []
        
        print(f"Starting {args.test_suite} load test suite...")
        print(f"Target URL: {args.url}")
        print("=" * 50)
        
        for config in test_configs:
            stats = await tester.run_load_test(config)
            all_stats.append(stats)
        
        # Print summary
        print("\n" + "=" * 50)
        print("LOAD TEST SUMMARY")
        print("=" * 50)
        
        for stats in all_stats:
            print(f"\n{stats['test_name']}:")
            print(f"  Requests: {stats['total_requests']} ({stats['successful_requests']} successful)")
            print(f"  Success Rate: {stats['success_rate']:.2f}%")
            print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
            print(f"  Requests/Second: {stats['requests_per_second']:.2f}")
            if stats['avg_tokens_per_second'] > 0:
                print(f"  Avg Tokens/Second: {stats['avg_tokens_per_second']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())