# Dynamic Runtime Updates Feature

## 🎯 Overview

The sampling system now supports **dynamic runtime updates** - you can change settings, wells, or stop/eject during an active sampling sequence, and the changes will apply **immediately** to remaining wells without restarting.

## ✨ Key Features

### 1. **Dynamic Timing Updates**
- Change buffer time or sampling time via Settings dialog **during sampling**
- New timing values apply to all **remaining wells** in the sequence
- Current well completes with old timing, next well uses new timing
- Visual confirmation shown in display screen

**How it works:**
```python
# Instance variables track current values
self.t_buffer = 60
self.t_sampling = 90
self.settings_lock = threading.Lock()

# Control_Move reads current values before each well
with self.settings_lock:
    current_buffer = self.t_buffer
    current_sampling = self.t_sampling
```

### 2. **Dynamic Well List Updates**
- Add or remove wells from selection **during sampling**
- System detects changes and rebuilds sequence for remaining wells
- If all remaining wells removed, sampling stops gracefully
- Visual confirmation of updated well list

**How it works:**
```python
# Check for updates before each well
updated_wells = self._check_well_list_updates()
if updated_wells is not None:
    # Rebuild method for remaining wells
    remaining_wells = updated_wells[current_index:]
    # Update sequence and continue
```

### 3. **Interruptible Stop**
- Stop button interrupts within **0.5 seconds**
- Checks stop flag every 0.5s during buffer and sampling waits
- Clean exit without leaving robot in undefined state
- "Process stopped by user" message displayed

### 4. **State Tracking**
- `is_sampling` flag indicates if sequence is active
- `current_well_index` tracks progress through sequence
- Settings dialog shows real-time update notification during sampling

## 🚀 Usage Examples

### Example 1: Adjusting Timing Mid-Sequence

**Scenario:** You started sampling with 60s buffer / 90s sampling, but realize you need longer sampling time.

**Steps:**
1. Sequence is running, currently on well A3
2. Click **Settings** button
3. Change Sampling Time to `120` seconds
4. Click **OK**
5. Display shows: `⚙️ Settings updated: Buffer=60s, Sampling=120s (applied to remaining wells)`
6. Well A3 completes with 90s (old value)
7. Well A4 onwards use 120s (new value)

### Example 2: Adding Wells During Sampling

**Scenario:** Forgot to include well H12, sequence already running.

**Steps:**
1. Sequence is running on wells A1-A6
2. Currently sampling A3
3. **Ctrl+Click** on well H12 to add it
4. Before next well starts, system detects change
5. Display shows: `🔄 Well list updated: A4, A5, A6, H12 remaining`
6. Sequence continues with updated list

### Example 3: Removing Wells During Sampling

**Scenario:** Well D4 is contaminated, remove it from sequence.

**Steps:**
1. Sequence running on A1-D6
2. Currently on B2
3. **Ctrl+Click** on D4 to deselect (turns from green to white)
4. System detects change before next well
5. Display shows updated well list without D4
6. Sequence skips D4 automatically

### Example 4: Emergency Stop

**Scenario:** Notice robot misbehaving, need to stop immediately.

**Steps:**
1. Click **STOP** button
2. Within 0.5 seconds, current sleep is interrupted
3. Display shows: `Process stopped by the user.`
4. Robot remains at current position (safe state)
5. Can eject, adjust, and restart manually

## 🔧 Technical Implementation

### Architecture Changes

**Before (Global Variables):**
```python
# Global timing (couldn't update during sampling)
t_buffer = 60
t_sampling = 90

def Control_Move(method, duration):
    for step in method:
        time.sleep(t_buffer)  # Always uses initial value
        time.sleep(duration)  # Always uses initial value
```

**After (Instance Variables + Lock):**
```python
# Instance variables with thread-safe access
self.t_buffer = 60
self.t_sampling = 90
self.settings_lock = threading.Lock()

def Control_Move(method):
    for step in method:
        # Read current values each iteration
        with self.settings_lock:
            current_buffer = self.t_buffer
            current_sampling = self.t_sampling
        
        # Check for well list updates
        updated_wells = self._check_well_list_updates()
        if updated_wells:
            # Rebuild sequence
            
        # Use current values
        self._interruptible_sleep(current_buffer)
        self._interruptible_sleep(current_sampling)
```

### Key Methods

#### `SettingsDialog.accept_settings()`
Updates both global and instance timing variables:
```python
def accept_settings(self):
    global t_sampling, t_buffer
    # Update globals (for new sequences)
    t_sampling = self.sampling_time_spinbox.value()
    t_buffer = self.buffer_time_spinbox.value()
    
    # Update instance (for running sequence)
    if hasattr(self.parent(), 'settings_lock'):
        gui = self.parent()
        with gui.settings_lock:
            gui.t_buffer = t_buffer
            gui.t_sampling = t_sampling
        if gui.is_sampling:
            gui.add_to_display("⚙️ Settings updated...")
```

#### `Control_Move()`
Main sampling loop with dynamic updates:
```python
def Control_Move(self, method):
    try:
        for i, step in enumerate(method):
            self.current_well_index = i
            
            # 1. Check stop
            if self.stop_flag:
                return
            
            # 2. Check well list updates
            updated_wells = self._check_well_list_updates()
            if updated_wells:
                # Rebuild for remaining wells
                
            # 3. Get current timing
            with self.settings_lock:
                current_buffer = self.t_buffer
                current_sampling = self.t_sampling
            
            # 4. Execute with current values
            self._interruptible_sleep(current_buffer)
            connection.Move(step)
            self._interruptible_sleep(current_sampling)
    finally:
        self.is_sampling = False
```

#### `_check_well_list_updates()`
Detects well selection changes:
```python
def _check_well_list_updates(self):
    global ctrl_selected_wells
    current_selection = self.order(list(ctrl_selected_wells))
    
    if current_selection != self.well_list:
        return current_selection
    
    return None
```

## ⚡ Performance

- **Stop Response:** < 0.5 seconds (was 60+ seconds)
- **Setting Update:** Immediate (next well)
- **Well Update:** Before next well starts
- **Thread Safety:** Lock-based synchronization (minimal overhead)

## 🎨 User Experience

### Visual Feedback

- ⚙️ **Settings Update:** `Settings updated: Buffer=60s, Sampling=90s (applied to remaining wells)`
- 🔄 **Well List Update:** `Well list updated: A4, A5, H12 remaining`
- ⚠️ **All Wells Removed:** `All remaining wells removed. Stopping.`
- 🛑 **Stop:** `Process stopped by the user.`

### Smooth Workflow

1. **Start sequence** - No interruption needed
2. **Adjust on-the-fly** - Settings/wells update live
3. **Continue seamlessly** - No restart required
4. **Stop anytime** - Instant response

## 🔒 Thread Safety

All shared state uses proper synchronization:

- `settings_lock` - Protects timing variables
- `stop_flag` - Simple boolean check (no lock needed)
- `ctrl_selected_wells` - Global set (read-only in thread)
- `is_sampling` - Status flag set in finally block

## 📊 State Diagram

```
[Idle] 
   │
   ├─→ Click MOVE/RUNPLATE
   │         │
   │         ↓
   │    [Sampling]
   │         │
   │         ├─→ Settings Changed → Update timing for next well
   │         ├─→ Wells Changed → Rebuild sequence
   │         ├─→ Stop Pressed → Exit within 0.5s
   │         │
   │         ↓
   │    [Complete]
   │         │
   └─────────┘
   
[Back to Idle]
```

## 🧪 Testing

### Test 1: Dynamic Timing
1. Start sequence with 60s buffer, 90s sampling
2. After 2 wells, change to 30s buffer, 60s sampling
3. **Expected:** Well 3 onwards uses 30s/60s

### Test 2: Add Wells
1. Start with A1, A2, A3
2. After A1 completes, add A4, A5
3. **Expected:** Sequence continues A2, A3, A4, A5

### Test 3: Remove Wells
1. Start with A1-A6
2. After A2, remove A4, A5
3. **Expected:** Sequence continues A3, A6 (skips A4, A5)

### Test 4: Rapid Stop
1. Start long sequence (30+ wells)
2. Press STOP during buffer wait
3. **Expected:** Stops within 0.5 seconds

## 💡 Tips

- **Timing Changes:** Applied to remaining wells immediately
- **Well Changes:** Detected before each new well starts
- **Stop:** Always safe, robot stays at current position
- **Eject:** Can eject during sampling (emergency only!)
- **Visual Feedback:** Watch display screen for confirmations

## 🎓 How It Works Under the Hood

The magic happens in the `Control_Move` loop:

```python
for i, step in enumerate(method):
    # ↓ Check 1: Did user press stop?
    if self.stop_flag:
        return
    
    # ↓ Check 2: Did well list change?
    if wells_changed():
        rebuild_sequence()
    
    # ↓ Check 3: Get latest timing
    timing = get_current_timing()
    
    # ↓ Execute with current values
    wait(timing.buffer)
    move_to_well()
    wait(timing.sampling)
```

Every iteration checks for updates before proceeding!

---

**Enjoy seamless, interruptible, dynamically-updatable sampling!** 🎉
