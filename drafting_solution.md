1. The Storage and State Layer (The Queue)
The foundation of any robust batch pipeline is state management. We do not extract data directly from an operational table and hold it in memory, because if the pipeline fails midway, we lose our place. Instead, we implement a Staging/Queue architecture.
The Airflow extraction task pulls raw records (e.g., thousands of user reviews) from the source and inserts them into a processing_queue table with a status of PENDING. This isolates the active workload. If the worker crashes, the queue retains the state, ensuring no row is processed twice and no row is permanently dropped. Once a row is successfully processed and validated, its state transitions to COMPLETED and the payload is pushed to the final analytical table. If it consistently fails validation, it is routed to a Dead Letter Queue (DLQ) for manual inspection.

2. The Extraction and Chunking Layer (The Generator)
You cannot send 10,000 rows to an LLM at once (due to context limits), nor should you send them 1 by 1 (due to network overhead). The first Python worker task queries the PENDING queue and applies Context Packing. It groups the raw data into optimized JSON arrays—for example, 50 rows per chunk.
This chunking strategy is vital: it reduces 50 individual HTTP network hops down to a single call, and it ensures we only pay the token cost for our "System Instructions" once per chunk rather than 50 times. This task then pushes this list of chunks (e.g., 200 chunks of 50 rows) into Airflow's XCom storage, acting as the generator for our downstream mapping.

3. Dynamic Task Mapping (The Fan-Out)
This is the core parallelization engine. Rather than writing a for loop that iterates through the 200 chunks sequentially, we use Airflow's dynamic task mapping (.expand()). Airflow evaluates the XCom list and dynamically spawns 200 independent, parallel worker tasks—one for each chunk.
Crucially, we enforce a strict concurrency limit using Airflow Pools (e.g., a pool limited to 10 active slots). This ensures that while Airflow wants to run all 200 tasks immediately, it only allows 10 concurrent HTTP requests to hit the LLM API at any given millisecond. This throttle mathematically prevents the pipeline from triggering an HTTP 429 "Too Many Requests" error from the provider.

4. The Routing Waterfall (Cost & Latency Optimization)
Inside each mapped task, we implement an intelligent routing waterfall to optimize cost. We do not send every chunk to an expensive model like GPT-4o. Instead, the Python worker first sends the 50-row chunk to a cheap, fast model (like gpt-4o-mini).
The worker uses asyncio to handle this network call asynchronously, freeing up the worker's thread while waiting for the HTTP response. The system instructions are structured to utilize Prompt Caching, ensuring the massive schema definitions are cached by the provider, dropping our input token costs significantly.

5. Schema Enforcement (The Decision Gate)
When the cheap model returns its payload, we do not blindly trust it. The payload is immediately passed through a strict Pydantic validation schema. Pydantic enforces two things: first, that the output is perfect JSON matching our database columns; and second, that the array length and the unique IDs perfectly match the input chunk (preventing the "lost in the middle" hallucination).
If Pydantic successfully parses the payload, the worker returns the data. If the cheap model hallucinates a bad schema, Pydantic throws a ValidationError. The Python except block catches this failure and immediately routes that specific chunk to the expensive, highly capable model for a second attempt. This guarantees we only pay premium API costs for the 5-10% of data that actually requires complex reasoning.

6. The Load Layer (Fan-In and Reduce)
Once all 200 dynamically mapped tasks succeed, the pipeline fans back in. A final "Reduce" task collects the validated outputs from all parallel workers via XCom. It flattens the nested lists into a single continuous dataset, executes a bulk UPDATE on the processing_queue table to mark the rows as COMPLETED, and performs a bulk INSERT of the enriched, structured data into the final analytical database.

This architecture achieves the exact same goals as Databricks' native ai_classify functions—high throughput, robust fault tolerance, and cost efficiency—but operates entirely under your control as a custom ETL pipeline.