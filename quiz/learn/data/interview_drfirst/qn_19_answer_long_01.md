# Q: Share a variable between multiple processes

## Interview Question: How do you share variables between processes in Python's multiprocessing, and what are the key considerations?

**Answer:** When working with multiprocessing, I use these IPC (Inter-Process Communication) methods with healthcare-specific considerations:

### 1. **Shared Memory (Value/Array)**
```python
from multiprocessing import Process, Value, Array
import numpy as np

def process_func(shared_val, shared_arr):
    # Modify shared values (thread-safe operations)
    with shared_val.get_lock():
        shared_val.value += 1
    shared_arr[:] = np.random.rand(10)  # Direct assignment works for arrays

if __name__ == '__main__':
    counter = Value('i', 0)  # 'i' for integer
    arr = Array('d', 10)     # 'd' for double

    p = Process(target=process_func, args=(counter, arr))
    p.start()
    p.join()

    print(f"Counter: {counter.value}, Array: {arr[:]}")
```
**Considerations:**
- **Type Codes Matter:** `'i'` (int), `'d'` (double), `'f'` (float)
- **Locking Required:** For value increments to prevent race conditions
- **PHI Alert:** Not suitable for complex objects containing patient data

### 2. **Manager Objects (Dict/List)**
```python
from multiprocessing import Manager

def process_data(shared_dict):
    shared_dict['patient_count'] += 1
    shared_dict['last_processed'] = "diabetes_case_123"

with Manager() as manager:
    shared_data = manager.dict({
        'patient_count': 0,
        'last_processed': None
    })

    processes = [Process(target=process_data, args=(shared_data,)) 
                for _ in range(4)]
    [p.start() for p in processes]
    [p.join() for p in processes]

    print(shared_data)
```
**Considerations:**
- **Slower Performance:** Proxy overhead for remote access
- **HIPAA Note:** Automatically handles serialization of sensitive data
- **Best For:** Complex structures needing process-safe modifications

### 3. **Queues (Message Passing)**
```python
from multiprocessing import Process, Queue

def producer(q):
    q.put({"patient_id": 101, "diagnosis": "J18.9"})  # Serialized automatically

def consumer(q):
    while not q.empty():
        data = q.get()
        # Process PHI data here

if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))

    p1.start(); p2.start()
    p1.join(); p2.join()
```
**Considerations:**
- **Serialization Limits:** Can't pass lambda functions or database connections
- **Queue Size:** Can become memory bottleneck for large datasets
- **Audit Trail:** Easy to log all inter-process messages for compliance

### 4. **Pipes (Bi-directional)**
```python
from multiprocessing import Process, Pipe

def worker(conn):
    conn.send(["lab_results", 125.7])
    received = conn.recv()  # Wait for acknowledgment
    conn.close()

parent_conn, child_conn = Pipe()
p = Process(target=worker, args=(child_conn,))
p.start()

print(parent_conn.recv())  # ["lab_results", 125.7]
parent_conn.send("ACK")
p.join()
```
**Considerations:**
- **Deadlock Risk:** Both ends blocking on recv() simultaneously
- **Throughput:** Slower than Queue for high-volume data
- **Use Case:** Ideal for heartbeat monitoring between processes

### Healthcare-Specific Best Practices:
1. **PHI Encryption:** Always encrypt sensitive data before IPC
   ```python
   from cryptography.fernet import Fernet
   shared_data = manager.dict({
       'encrypted_phi': Fernet(key).encrypt(b"Patient: John Doe")
   })
   ```
2. **Memory Mapping:** For large medical imaging data
   ```python
   import mmap
   with open('mri.dat', 'r+b') as f:
       mm = mmap.mmap(f.fileno(), 0)
       # Multiple processes can access
   ```
3. **Validation Layer:** Sanitize all cross-process data
   ```python
   def validate_phi(data):
       if not isinstance(data['patient_id'], int):
           raise ValueError("Invalid PHI format")
   ```

*Would you like me to demonstrate how we implemented a secure audit log for all cross-process PHI transfers in our healthcare chatbot?*  

*(This shows both technical depth and compliance awareness - critical for DrFirst's healthcare applications.)*
