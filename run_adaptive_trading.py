#!/usr/bin/env python3

# ULTRA-LOW LATENCY HFT ORCHESTRATOR - MAXIMUM STARTUP SPEED
# Minimal imports for sub-5ms startup time

import asyncio
import atexit
import signal
import sys
from polygon_client import RealTimeDataFeed

# Global cleanup function for CUDA contexts
def cleanup_cuda_contexts():
    """Cleanup CUDA contexts to prevent PyCUDA errors"""
    try:
        # Try to import and cleanup CUDA contexts
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Check if there's a current context
        try:
            current_ctx = cuda.Context.get_current()
            if current_ctx:
                # Synchronize before cleanup
                cuda.Context.synchronize()
                # Pop the context safely
                current_ctx.pop()
        except cuda.LogicError:
            # Context already popped or invalid
            pass
        except Exception:
            # Other CUDA errors, ignore
            pass
            
        # Clean up any remaining contexts in the stack
        try:
            while True:
                try:
                    ctx = cuda.Context.get_current()
                    if ctx:
                        ctx.pop()
                    else:
                        break
                except cuda.LogicError:
                    break
                except Exception:
                    break
        except Exception:
            pass
            
    except ImportError:
        pass  # PyCUDA not available
    except Exception:
        pass  # Ignore cleanup errors

# Register cleanup function
atexit.register(cleanup_cuda_contexts)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    cleanup_cuda_contexts()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Hardcoded API key for maximum speed (no import overhead)
POLYGON_API_KEY = "Tsw3D3MzKZaO1irgwJRYJBfyprCrqB57"

async def main():
    """Ultra-fast startup - WebSocket handles all component initialization"""
    
    try:
        # Single component initialization - WebSocket becomes master controller
        feed = RealTimeDataFeed(
            api_key=POLYGON_API_KEY,
            symbols=None,  # Auto-fetch all 11,500 symbols
            enable_filtering=True,
            memory_pools=None  # WebSocket will create its own optimized pools
        )
        
        # Start WebSocket - it will auto-initialize all other components
        await feed.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure cleanup happens
        cleanup_cuda_contexts()

if __name__ == "__main__":
    try:
        # Direct execution for maximum speed
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    finally:
        # Final cleanup
        cleanup_cuda_contexts()