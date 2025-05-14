# Q: multiprocessing vs multithreading

## Interview Question: Can you explain the difference between multiprocessing and multithreading, and when to use each approach?

**Answer:** Understanding this distinction is crucial for optimizing performance in data science pipelines. Here's my practical breakdown:

### Key Differences
| Characteristic       | Multiprocessing                      | Multithreading                      |
|-|--|-|
| **Memory**           | Separate memory space per process    | Shared memory space                 |
| **GIL Impact**       | Bypasses GIL (true parallelism)      | Bound by GIL (concurrent but not parallel) |
| **Overhead**         | High (new Python interpreter)        | Low (lightweight threads)           |
| **CPU Utilization**  | Uses multiple cores                  | Limited to single core              |
| **IPC**              | Requires serialization (pickle)      | Direct variable access              |

### When to Use Each (With Healthcare Examples)

**1. Use Multiprocessing When:**
```python
# CPU-bound medical image processing
from multiprocessing import Pool

def process_mri_slice(slice):
    # Heavy computation (CNN inference)
    return model.predict(slice)

with Pool(processes=8) as pool:
    results = pool.map(process_mri_slice, mri_slices)
```
- *My Project Use Case:* Processing 10,000+ pathology slides in parallel for cancer detection
- **Considerations:**
  - Watch for memory bloat (each process loads full Python interpreter)
  - Avoid for small tasks (overhead > benefit)
  - Required for NumPy/SciPy heavy computations

**2. Use Multithreading When:**
```python
# I/O-bound EHR API calls
from concurrent.futures import ThreadPoolExecutor

def fetch_patient_records(patient_id):
    # Network-bound operation
    return requests.get(f'ehr-api/{patient_id}').json()

with ThreadPoolExecutor(max_workers=20) as executor:
    records = list(executor.map(fetch_patient_records, patient_ids))
```
- *My Project Use Case:* Simultaneously querying 50+ external pharmacy databases
- **Considerations:**
  - Useless for CPU-bound tasks (GIL bottleneck)
  - Risk of race conditions (need `threading.Lock()`)
  - Ideal for DB/network operations with high latency

### Hybrid Approach (Advanced Pattern)
```python
# Combining both for ETL pipeline
def process_chunk(chunk):
    # CPU-bound
    with ThreadPoolExecutor(4) as thread_pool:
        # I/O-bound within process
        results = list(thread_pool.map(transform_data, chunk))
    return results

with multiprocessing.Pool(4) as process_pool:
    final = process_pool.map(process_chunk, data_chunks)
```

**Critical Healthcare-Specific Factors:**
1. **Data Sensitivity:** Multiprocessing requires careful PHI serialization
2. **Cloud Costs:** Threading often cheaper (less memory than multiprocessing)
3. **API Limits:** ThreadPoolExecutor's `max_workers` must respect rate limits
4. **Debugging:** Threading issues are harder to reproduce (race conditions)

*Would you like me to share how I implemented a thread-safe cache for patient data aggregation?*  

*(This demonstrates both deep technical understanding and practical healthcare data experience - valuable for DrFirst's high-performance systems.)*
