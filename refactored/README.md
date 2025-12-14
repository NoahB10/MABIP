# MABIP Async Refactoring

This folder contains the modernized async version of the MABIP system.

## Architecture Overview

### Core Files Created

1. **config.py** - Centralized configuration constants
   - `HARDWARE` - Serial/Bluetooth/timing settings
   - `UI` - Window sizes, colors, layouts
   - `SENSOR` - Gain values, calibrations
   - `FILES` - File paths, formats
   - `ASYNC` - Timeouts, queue sizes

2. **app_state.py** - Thread-safe application state
   - Replaces all global variables
   - Uses asyncio.Lock for thread safety
   - Centralized well selections, connection, timing

3. **async_utils.py** - Async helper functions
   - `AsyncTaskManager` - Track and cleanup tasks
   - `interruptible_sleep()` - Cancelable sleep
   - `AsyncRateLimiter` - Rate limiting for commands

## Migration Plan

### Phase 1: Foundation (Current)
- ✅ Created config constants
- ✅ Created AppState for globals
- ✅ Added async utilities
- ✅ Fixed missing checkProgress() in AMUZA_Master

### Phase 2: Async AMUZA Communication
- Create `AMUZA_Master_async.py` with:
  - Command priority queue
  - Async Bluetooth using asyncio.to_thread()
  - Proper stop command handling
  - State tracking with async events

### Phase 3: Async Serial Communication
- Create `sensor_reader_async.py` with:
  - serial-asyncio for non-blocking reads
  - asyncio.Queue for data passing
  - Proper resource cleanup

### Phase 4: Async GUI
- Refactor `Sampling_Collector.py` to use:
  - qasync for PyQt5 async integration
  - Async slot decorators
  - AsyncTaskManager for background work
  - Incremental file reading

### Phase 5: Testing & Polish
- Add unit tests with pytest-asyncio
- Performance monitoring
- Error handling improvements
- Documentation

## Installation

```bash
# Install new async dependencies
pip install -r requirements.txt
```

## Key Improvements

1. **No More Globals** - All state in AppState with proper locking
2. **Responsive Stop** - Async sleep checks every 0.5s instead of blocking
3. **Better Performance** - Incremental file reading, cached data
4. **Resource Safety** - AsyncTaskManager ensures cleanup
5. **Command Queue** - Priority-based Bluetooth commands
6. **Type Safety** - Dataclasses for configuration
7. **Logging** - Proper logging infrastructure

## Next Steps

1. Review the created files (config.py, app_state.py, async_utils.py)
2. Install dependencies: `pip install -r requirements.txt`
3. I'll create the async versions of AMUZA_Master and sensor reader
4. Then refactor the main GUI with qasync integration

## Testing Mock Mode

The refactored version will have better mock mode support for testing without hardware.
