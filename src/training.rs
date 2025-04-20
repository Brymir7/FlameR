use crate::lazybuffer::{Backend, BufferHandle};
use std::collections::HashMap;
use std::sync::Mutex;

// Global buffer cache to reuse buffers across iterations in training loops
lazy_static::lazy_static! {
    static ref BUFFER_CACHE: Mutex<HashMap<String, Vec<BufferHandle>>> = Mutex::new(HashMap::new());
    static ref TRAINING_IN_PROGRESS: Mutex<HashMap<String, bool>> = Mutex::new(HashMap::new());
}

/// Begin a training loop which enables buffer caching
/// This allows the system to reuse buffers between training iterations
pub fn begin_training_loop(backend: &dyn Backend) {
    let backend_name = backend.name().to_string();
    let mut training_map = TRAINING_IN_PROGRESS.lock().unwrap();
    
    // Mark that training is in progress for this backend
    training_map.insert(backend_name, true);
    
    // Initialize an empty buffer cache for this backend if it doesn't exist
    let mut buffer_cache = BUFFER_CACHE.lock().unwrap();
    if !buffer_cache.contains_key(backend.name()) {
        buffer_cache.insert(backend.name().to_string(), Vec::new());
    }
    
    println!("Started training loop with buffer caching on {} backend", backend.name());
}

/// End a training loop and release cached buffers
pub fn end_training_loop(backend_name: &str) {
    {
        let mut training_map = TRAINING_IN_PROGRESS.lock().unwrap();
        training_map.insert(backend_name.to_string(), false);
    }
    
    // Optionally free all cached buffers
    // Note: in a real-world scenario, you might want to keep some buffers around
    // for future training sessions depending on memory constraints
    
    println!("Ended training loop on {} backend", backend_name);
}

/// Request a buffer from the cache or allocate a new one if none available
pub fn get_cached_buffer(backend: &dyn Backend, size: usize) -> BufferHandle {
    let mut buffer_cache = BUFFER_CACHE.lock().unwrap();
    let backend_name = backend.name();
    
    // Check if we're in a training loop
    let training_map = TRAINING_IN_PROGRESS.lock().unwrap();
    let training_active = training_map.get(backend_name).unwrap_or(&false);
    
    if *training_active {
        // Try to find a cached buffer of the right size
        if let Some(cache) = buffer_cache.get_mut(backend_name) {
            // Find a buffer with matching size
            if let Some(pos) = cache.iter().position(|b| b.size == size) {
                return cache.swap_remove(pos);
            }
        }
    }
    
    // If no suitable buffer found or training not active, allocate a new one
    backend.allocate_buffer(size)
}

/// Return a buffer to the cache for potential reuse
pub fn return_buffer_to_cache(backend_name: &str, handle: BufferHandle) {
    let mut buffer_cache = BUFFER_CACHE.lock().unwrap();
    
    // Check if we're in a training loop
    let training_map = TRAINING_IN_PROGRESS.lock().unwrap();
    let training_active = training_map.get(backend_name).unwrap_or(&false);
    
    if *training_active {
        // Add buffer to cache
        if let Some(cache) = buffer_cache.get_mut(backend_name) {
            cache.push(handle);
        }
    }
}

/// Free all cached buffers for a specific backend
pub fn free_all_cached_buffers(backend: &dyn Backend) {
    let mut buffer_cache = BUFFER_CACHE.lock().unwrap();
    let backend_name = backend.name();
    
    if let Some(cache) = buffer_cache.get_mut(backend_name) {
        for handle in cache.drain(..) {
            backend.free_buffer(&handle);
        }
        println!("Freed all cached buffers for {} backend", backend_name);
    }
}