# How the AI Functions Turn Business Data into Results

## What this product does

This prototype helps an organization use AI to classify or summarize business records in a controlled way.

For example, it can:

- label a customer note as `follow_up`, `resolved`, or `escalate`; or
- shorten a long note into a summary with a specified word limit.

The system does more than send text to an AI model. It controls which data can be used, limits cost and data exposure, checks the AI response, and clearly reports when part of the work cannot be completed.

The two available **AI functions** are:

- **`ai_classify`** — assigns one allowed label to each record.
- **`ai_summarize`** — creates a summary within the requested size limit.

An AI function is simply a predefined job that AI performs. The caller chooses the function and its options, such as the allowed labels or maximum summary length. The caller does not write or change the hidden AI instructions.

## A simple example

Imagine a support team with customer records that contain a customer ID and a note.

A support agent opens one customer and asks the system to classify the note. The system safely reads the record, sends the permitted text to an AI model, checks that the answer uses one of the allowed labels, and returns the result while the agent is waiting.

Later, an operations team asks the system to summarize thousands of customer notes. The system processes each customer separately, saves progress as it works, and reports which summaries succeeded or failed. If separately authorized, it can save approved summaries to a destination system.

In both cases, a raw AI response is **not** automatically treated as a valid result. The response must pass checks before it is returned or saved.

## Key terms in plain language

| Term | Meaning in this guide |
|---|---|
| **Source data** | The business records given to the AI function, such as customer IDs and notes. |
| **Interactive request** | Work performed while a person or application waits for an immediate response. |
| **Batch job** | Work performed over a list of records, with progress tracked separately for each one. The code often calls this “bulk” work. |
| **AI provider** | The service or model that creates the proposed classification or summary. The prototype uses a fake provider and makes no external AI calls. |
| **Validation** | Automatic checks that decide whether an AI response has the required fields, IDs, labels, and size limits. |
| **Configuration** | The approved set of rules used for one execution, including function definitions, security rules, cost limits, and validation rules. |
| **Write-back** | Saving an accepted batch result to a destination data system. The code may also call this a mutation or effect. |
| **Checkpoint** | A saved batch milestone that allows processing to continue without repeating every completed step. |

## The process at a glance

```text
Business records + chosen AI function
    -> check the request and rules
    -> prepare only the permitted data
    -> confirm cost and security approval
    -> ask the AI provider for a proposed answer
    -> check the answer against the function rules
    -> return or save the accepted result
```

| Step | Question answered |
|---|---|
| 1. Accept the request | Which processing path did the caller request, and is the request unambiguous? |
| 2. Lock the rules | Which exact versions of the business, security, cost, and validation rules govern this work? |
| 3. Obtain the records | Where does the source data come from, and is the system permitted to use it? |
| 4. Prepare the AI work | Is the selected function valid, and how will its records be packaged safely? |
| 5. Apply cost and security controls | Which model may process the data, and is this particular call affordable and permitted? |
| 6. Generate a proposed answer | How is the controlled request executed, and what does the provider return? |
| 7. Check the answer | Can the proposed answer be safely connected to the requested business records? |
| 8. Deliver the outcome | Which accepted results and failures should the caller receive or the system store? |

## The eight steps in detail

### Step 1: Accept the request

The first step decides whether the caller is asking for an **immediate request** or a **batch job**. This is not an AI judgment, and the system does not choose based on how many records it sees. The caller chooses by submitting one of two request types. An immediate request is intended for a person or application waiting for a response. A batch job is intended for independently tracked items that may take longer and may need to resume after an interruption.

The request arrives in an `AdmissionEnvelope`, which is a container holding a configuration reference and the immediate or batch content. The admission router applies a small set of deterministic rules:

1. If only immediate content is present, the immediate path is selected.
2. If only batch content is present, the batch path is selected.
3. If neither is present, the request is rejected because there is no work to perform.
4. If both are present and no explicit path is supplied, the request is rejected as ambiguous.
5. If both are present with an explicit path, that path is selected and only its content is dispatched.
6. If the explicit path says “batch” but only immediate content exists, or says “immediate” but only batch content exists, the request is rejected as inconsistent.

These rules run before configuration loading, database access, budget reservation, or provider use. A malformed or ambiguous request therefore cannot accidentally read data or call a model. Once a path is selected, the router sends the request to exactly one processing component; it never starts both paths and chooses a winner later.

An **accepted** request has passed only this entry decision and configuration binding. Acceptance does not promise a successful AI result. Source-data, budget, security, provider, or answer-checking failures can still occur later and will be reported as execution outcomes rather than admission failures.

**Output of this step:** one selected path and its request content, or a typed rejection explaining why processing did not start.

### Step 2: Lock the rules

Every request refers to a configuration version. The configuration is the approved rulebook for the execution: it identifies the function definitions, query plans, model-routing policy, security policy, validation rules, cost limits, credentials, and write-back setting that the system must use.

Before a configuration can become active, the configuration registry checks that required sections and values are present and that referenced policy, validation, model, credential, and adapter-evidence versions exist. It also checks basic value rules, such as whether required numbers are positive and whether expected flags are actually true or false. Invalid configurations cannot be activated for normal use.

The registry then converts the configuration into a standard representation and derives a version from its content. This means materially different rule content receives a different version. When a request starts, the system binds the requested version into a frozen execution snapshot. Activating a newer configuration later does not change work that is already running.

The system also creates an execution context containing unique execution and correlation IDs. These IDs connect results, saved progress, and audit events to the same request. Immediate requests receive one absolute deadline used throughout the workflow. Batch jobs do not use that immediate-response deadline because they are designed to progress through saved stages.

This locking step is important because an answer is meaningful only in relation to the rules that produced it. For example, the same text might receive a different label if the allowed labels or function instructions changed. Recording the exact configuration makes that difference traceable.

The admission-only CLI demonstration uses a lighter binder that freezes supplied values without performing the registry's full activation process. The executed prototype uses its own validated `prototype` configuration.

**Output of this step:** an immutable execution context containing the exact rule versions, identity, cancellation information, and—when applicable—the response deadline.

### Step 3: Obtain the records

The two processing paths obtain source data differently.

For an immediate request, the caller names an approved **query plan** and supplies its parameters. A query plan is a predefined description of allowed database work; it is not free-form SQL written by the caller or generated by the model. The data-access component resolves the exact plan version from the locked configuration and confirms that the operation is a read rather than an insert, update, or delete.

Before submitting the read, the system checks that the database adapter supports the capabilities the plan requires, the result has deterministic ordering where needed, parameter types are valid, the configured credential is authorized for read-only access, and required transport and storage protections are recorded. If any check fails, the database call does not proceed. A successful adapter response is converted into a standard table-shaped result so later processing does not depend on one database vendor's response format.

The configured row mapping then identifies which column is the business ID and which columns contain the source values for the AI function. IDs must be nonblank and unique, each row must have the expected number of columns, and values must be representable in the standard payload format. This mapping is the bridge between a database row and an identified AI input.

For a batch job, the current prototype does not query a database. The submitted job contains item IDs and may contain source data for each item. The dispatcher creates one independently tracked work item for each ID. If demonstration source data is omitted, the prototype uses `{"text": item_id}` as placeholder input; this fallback is a demo convenience, not a production retrieval strategy.

**Output of this step:** ordered, identified source records in a standard representation, or a read/input failure that prevents dependent AI work from starting.

### Step 4: Prepare the AI work

Preparation turns business records and caller options into a controlled AI-function request. The caller may select `ai_classify` or `ai_summarize`, but cannot supply an unrestricted hidden prompt. The system looks up the selected function's exact definition version in the frozen configuration. If the caller mentions a version, it acts only as a compatibility check; it cannot override the configured version.

The function definition determines its instructions, permitted inputs, limits, and expected output. Classification requires exactly one `labels` option containing distinct, nonblank allowed labels. Summarization requires exactly one positive `max_words` option within the definition's limit, and each source record must contain one valid `text` value. Unsupported functions, extra options, missing options, invalid values, or an unavailable definition cause preparation to fail before provider use.

After validation, the packer creates a standard payload containing the function name, definition version, controlled instructions, normalized options, identified rows, and expected output structure. Objects are serialized consistently so the same logical input produces the same bytes and content hash. These hashes allow saved batch data to be checked when work resumes.

Large inputs are divided into contiguous packages while preserving input order. Every package must remain within four configured limits: number of records, estimated input tokens, estimated output tokens, and final byte size. The packer measures the complete payload, including instructions and schema—not only the business text. If a single record cannot fit by itself, it is rejected as oversized. All packages are successfully prepared before any package is treated as ready, which avoids leaving a partial prepared request after a packing failure.

**Output of this step:** one or more canonical, content-addressed AI request packages with usage estimates, or a preparation failure produced before any source data reaches a provider.

### Step 5: Apply cost and security controls

For every prepared package, the model router considers the configured provider candidates. It removes any candidate that fails a mandatory condition. The routing policy must exist, be valid, and allow processing. The provider and data category must be permitted. The candidate must have the capabilities required by the function and meet the minimum quality score. Its estimated usage must appear affordable, and an immediate request must be predicted to finish before the shared deadline and response reserve.

Among the candidates that remain, the router sorts first by lowest estimated cost, then by higher configured priority, and finally by model ID for a stable tie-break. It asks the capacity controller for the first candidate with available request and token capacity. This is why the system is described as selecting the least-cost **eligible** model rather than simply the cheapest model. If no model qualifies or capacity is unavailable, the package is not sent.

After model selection, the budget controller performs the authoritative reservation. It attempts to reserve the estimated cost and tokens across every applicable budget scope as one operation. Either all required scopes accept the reservation or none do. A preliminary routing estimate is not the final budget decision. If reservation fails, the provider is not called.

The security gateway then fails closed: missing or invalid information causes rejection rather than best-effort disclosure. It resolves the security policy from the locked configuration and checks that the policy allows processing, the provider is approved, the data category is approved, and transport and storage protection requirements are present. It authenticates the calling service against the selected provider. Finally, it replaces fields or nested paths marked as sensitive with the configured masking value.

Only the masked payload produced after all of these checks can be passed to the provider gateway. A security denial produces an audit event and no provider call. In the current prototype, budget reservation happens before security evaluation, so a security rejection may be settled using estimated usage even though the provider call count remains zero.

**Output of this step:** a selected provider, capacity allocation, budget reservation, and secured payload, or a typed routing, capacity, budget, authentication, or policy failure.

### Step 6: Generate a proposed answer

The provider gateway receives the secured payload and asks the selected model to perform the predefined function. The provider is responsible only for proposing classifications or summaries; it does not decide whether its own response is acceptable.

#### How immediate requests handle multiple packages at once

An immediate request processes data in a specific order with specific concurrency rules:

1. **The database read runs first and alone.** The source records must be available before any AI package can be created, so no AI work begins until the read finishes. This is a strict dependency: the function's input comes from that read.

2. **After the read completes, all AI packages are launched concurrently.** If the source data produces three packages, all three start at approximately the same time. They do not wait for each other. Each package independently goes through model selection, budget reservation, security checks, and provider invocation.

3. **All launched packages share one remaining time budget.** The coordinator calculates the remaining time by subtracting a small response-assembly reserve from the original deadline. It then waits for all packages to finish within that window. Packages that finish in time have their results collected. Packages still running when time expires are marked as incomplete.

4. **Before launching each package, the coordinator checks whether there is still time.** If the deadline minus the response reserve has already passed, the package is immediately marked as deadline-exceeded without being started.

This means an immediate request with multiple packages can return partial results. Two packages might succeed while a third exceeds the deadline. The caller receives the two accepted records and a machine-readable explanation that the third was not completed in time.

#### How cancellation works

When the deadline is reached, pending packages are cancelled cooperatively. The coordinator:

- marks remaining work as deadline-exceeded in its response;
- sends a cancellation signal to the database and model executors; and
- watches for late completions that arrive after the deadline.

Late completions are **never** included in the caller's response. They are counted in the execution metrics and may generate telemetry, but they cannot retroactively change a delivered result. The system cannot force a remote provider to physically stop computing at that instant; it can only stop waiting and ignore the answer.

#### How batch items process packages

Batch processing handles packages differently from immediate requests:

1. **Each batch item is processed independently.** Items do not share a deadline or wait for each other within one worker run.

2. **Within one item, packages are processed one at a time in sequence.** The bulk pipeline iterates through the item's packages in order using a simple loop. It does not launch them concurrently. Each package's budget reservation, provider call, and response happen before the next package starts.

3. **All packages for an item must finish before the result is saved.** The "AI response received" checkpoint is written only after every package in the item has returned. This means if processing stops during the third of four packages, the first two provider calls may run again when the item resumes. The system does not save individual package responses between packages within one stage.

4. **Different items can potentially be processed by different workers.** The repository, outbox, and notification design allows multiple workers to claim different items from the same job. The current prototype runs items sequentially in one process, but the interfaces support independent item processing.

#### Why the two approaches differ

Immediate requests optimize for responsiveness within a fixed time limit. If one package takes longer than expected, the others can still finish and the caller receives partial value immediately.

Batch items optimize for correctability and recovery. Sequential package processing within an item makes budget accounting simpler and avoids partial checkpoint states within one stage. The trade-off is that a single slow package delays that item's completion, but since batch callers are not waiting for a real-time response, this is acceptable.

#### Budget settlement after execution

When a provider call finishes, the system compares the actual cost and token usage with the reservation made in Step 5. The difference is reconciled: unused reserved capacity is released. If execution fails after reservation but before a provider call, the prototype settles using the estimate because no actual usage is available.

The executable prototype uses `FakeModelProvider`, which creates deterministic local responses for tests and demonstrations. It does not send data to an external AI service. Regardless of provider, the returned value is considered an untrusted **proposed answer**, not a completed business result.

**Output of this step:** an untrusted provider response plus usage information, or an execution failure associated with the relevant package or batch item.

### Step 7: Check the answer

The answer-checking step converts an untrusted provider response into either accepted records or a controlled failure. It first asks whether the response can be read safely. The parser accepts supported in-memory values, JSON text or bytes, or a specifically configured provider envelope. It enforces the response-size limit and rejects invalid text encoding, malformed JSON, duplicate object keys, non-finite numbers, unsupported representations, the wrong top-level structure, non-object records, and missing or extra fields.

If parsing succeeds, deterministic validation compares the proposed records with the original request. It requires unique IDs, exactly one result for each input ID, no unknown IDs, the correct number of records, and expected field types and values. Classification adds the rule that every label must belong to the caller's approved label list. Summarization adds the requested word limit and the definition's character limit.

A failed package does not leak a partly trusted subset as successful output. Its proposed records are withheld and stable reason codes identify the failed rule without placing raw provider content in failure details. For a multi-package batch item, a failure in any package withholds the combined accepted result for that item. When all checks pass, records are restored to original input order by their IDs and package positions.

These checks establish structural and contractual correctness: the answer can be read, belongs to the expected records, and obeys the selected function's measurable limits. They cannot establish that a label is a good business decision or that a summary is complete, truthful, fair, or unbiased. Those questions require task-specific evaluation or human review.

**Output of this step:** ordered, accepted function records, or bounded parsing and validation failures that prevent proposed answers from being treated as results.

### Step 8: Deliver the outcome

Immediate and batch processing use different delivery contracts because their callers have different needs.

An immediate request returns one immutable response before its deadline. The response can contain the normalized database result, accepted AI records, and explicit entries for operations that did not complete. Each missing result has a machine-readable reason such as budget exhaustion, security rejection, validation failure, dependency failure, cancellation, or deadline expiration. The response also includes an overall status, configuration version, timestamps, and usage measurements. If the read succeeds but AI processing fails, the safe read can still be returned with an explanation of why the AI result is absent.

When the deadline approaches, pending work is marked incomplete, cancellation is requested, and late results are excluded from the caller's response. Cancellation is cooperative: the system can stop waiting and suppress a late answer, but it cannot guarantee that a remote provider physically stopped at that instant. Immediate processing remains read-only under every outcome because no write-back component exists in that path.

A batch item saves progress through preparation, provider response, parsing, validation, and completion. Each item can succeed or fail independently, and those item states are combined into a job status such as succeeded, partially succeeded, budget exhausted, or failed. With write-back disabled, accepted records are stored and no destination data is changed.

Write-back is a separate, batch-only decision after validation. Enabling it is not enough by itself. The configuration, policy, approval, job, item, destination, fields, type of change, validation version, and approval validity must all match. Recovery evidence is checked to avoid blindly repeating an uncertain change. The prototype sends approved changes only to a fake destination adapter.

Delivery never turns a failure into a fabricated answer. It exposes the accepted records, preserves safe partial information where the contract permits it, and reports the final state of every missing or failed operation.

**Output of this step:** a caller-visible immediate response or stored batch item outcomes, with optional separately approved write-back of validated batch results.

## Two ways to run an AI function

### Immediate processing

Use immediate processing when a person or application is waiting for an answer.

The caller supplies:

- an approved data query;
- the AI function to run; and
- the function options, such as allowed labels or a summary word limit.

The system first reads the data. If the read succeeds, it converts the selected columns into identified AI input records. It then runs the AI function within a fixed time limit.

The response can contain both successful information and clearly identified gaps. For example, the system may successfully read a customer record but be unable to return a classification because the cost limit was reached or the AI answer failed its checks. In that case, it returns the safe database result and explains why the AI result is missing.

Immediate processing is read-only. It cannot update the source records or save AI results back to a database, even if a caller asks it to do so.

### Batch processing

Use batch processing when many records must be handled independently and the work may need to continue after an interruption.

A batch job contains item IDs and, for AI-function work, the source data for each item. In the current prototype, batch processing does not retrieve those records from a database. If demonstration data is omitted, the prototype uses the item ID as placeholder text.

Each item receives a stable tracking key. The system saves milestones as the item moves through preparation, AI processing, response reading, answer checking, and completion. One failed item does not automatically fail the others.

At the end, the job reports an overall status such as:

- all items succeeded;
- some items succeeded;
- the available budget was exhausted; or
- the job failed.

The current prototype stores batch progress only in memory. It demonstrates how recovery should work, but it does not survive a process or machine restart.

## How the system prepares business data

### Immediate source data

For an immediate request, the caller chooses an approved query plan rather than providing free-form SQL. The data-access component checks that the plan is read-only, that required permissions exist, and that the results have predictable ordering. It then converts the database response into a standard table format.

The system maps configured columns from that table into AI input. For example:

```text
customer_id -> result identifier
note        -> text processed by the AI function
```

The identifier is important because it allows the system to prove that every accepted answer belongs to an input record.

### Batch source data

For a batch job, the source mapping is submitted with each item. For example:

```json
{
  "customer-1": {"text": "Customer requested a callback"},
  "customer-2": {"text": "Issue was resolved yesterday"}
}
```

The system turns each item into a separately tracked AI-function request. This makes it possible for `customer-1` to succeed even if `customer-2` fails.

## How an AI function is defined

The caller selects a named function instead of supplying an unrestricted prompt.

### Classification

`ai_classify` requires a list of allowed labels. The list must contain valid, distinct labels and remain within configured limits.

Example input option:

```json
{"labels": ["follow_up", "resolved", "escalate"]}
```

An accepted result must contain exactly a record ID and one of those labels:

```json
{"id": "customer-1", "label": "follow_up"}
```

### Summarization

`ai_summarize` requires a positive maximum word count. Each input record must contain one text field within the configured input-size limit.

Example input option:

```json
{"max_words": 20}
```

An accepted result must contain exactly a record ID and a summary that meets the requested word limit and the system's character limit:

```json
{"id": "customer-1", "summary": "Customer requested a callback about the unresolved delivery issue."}
```

Each execution uses a fixed version of the function definition. This keeps the instructions and result format consistent throughout the request or job. Asking for a missing or incompatible version causes the work to stop before any data is sent to the AI provider.

## How requests are kept within safe limits

The system creates a standard AI request containing:

- the selected function;
- the fixed definition version;
- the controlled instructions;
- the caller's valid options;
- the identified source records; and
- the expected result format.

Large groups of records are split into smaller packages. Each package must satisfy four limits:

1. number of records;
2. estimated input size;
3. estimated output size; and
4. final request size in bytes.

If even one record cannot fit safely, that record is rejected before the provider is called. The current size estimator is intentionally simple and is not a guarantee of provider billing or exact token usage.

## How cost and data exposure are controlled

Before an AI request is sent, the system chooses the least expensive provider option that still meets all required rules. A cheaper model is not selected if it lacks a required capability, does not meet the quality setting, violates a data policy, has no available capacity, or cannot finish within an immediate request's time limit.

The system then reserves the estimated cost and usage. If the budget is unavailable, the provider is not called. An immediate request reports a missing AI result; a batch item is marked as stopped because of the budget limit.

The security check then confirms that:

- the approved security policy is available;
- the selected provider and type of data are permitted;
- the calling service is authenticated;
- required protections are present; and
- configured sensitive fields are hidden or replaced before sending.

Only after these checks does the provider receive the prepared data.

In the prototype, a denied security check can still appear as estimated usage in the budget report because the budget is reserved first and later settled using the estimate when no provider usage is available. The provider call count remains zero. This is a prototype accounting behavior, not evidence that data was sent.

## How proposed AI answers become accepted results

The provider response is treated as untrusted input. It must pass two levels of checks.

### 1. Can the response be read safely?

The system rejects responses that are too large, malformed, contain duplicate fields, contain unsupported number values, use the wrong top-level structure, or contain missing or extra result fields.

### 2. Does the answer match the requested work?

The system checks that:

- every input ID has exactly one output;
- no unknown or duplicate IDs appear;
- each field has the expected type;
- a classification uses one of the allowed labels; and
- a summary stays within its word and character limits.

If a group fails, its proposed answers are withheld. Successful records are put back into input order before delivery.

These checks prove that an answer follows the requested structure and can be linked to the right record. They do **not** prove that the classification is wise, the summary is factually complete, or the AI is unbiased. Business-critical use may still require human review or additional quality checks.
## What the caller receives

### Immediate response

An immediate response includes:

- the successful database read, when available;
- accepted AI-function results;
- a reason for each missing result;
- an overall completion status;
- the time limit and response time;
- the configuration version; and
- basic usage measurements.

Common reasons for a missing AI result include an unavailable budget, a security rejection, invalid provider output, or an expired time limit. The system does not invent a replacement result when processing fails.

The time limit is cooperative. The system requests cancellation and ignores late answers, but it cannot guarantee that work in a remote system physically stops immediately.

### Batch result

Each batch item ends independently. Accepted output is stored in a standard format. Failed items record a clear final status and failure reason. The job combines those item outcomes into an overall status.

The command-line display sorts batch item results by item ID. Internally, items remain separately tracked by their stable job-and-item keys.

## How batch progress and recovery work

A batch item passes through five saved milestones:

1. **Prepared** — the function and source data were checked, and safe-sized AI requests were created.
2. **AI response received** — all proposed answers for the item were collected.
3. **Response read** — the provider output was converted into standard records.
4. **Answer accepted** — all records passed the required checks.
5. **Completed** — accepted output was stored or separately approved for write-back.

The source code names these milestones `ACCEPTED`, `MODEL_COMPLETED`, `RESULT_STORED`, `VALIDATED`, and `COMPLETED`. In code, the next milestone name describes the work the system is about to perform and save.

When work resumes, the system verifies the saved function definition and request data before continuing. It does not recreate milestones that are already complete.

There is one important limitation: progress is saved after all AI request packages for a milestone finish, not after every individual provider call. If processing stops halfway through a multi-package item, earlier provider calls in that milestone may run again. This prototype does not promise that each provider call happens exactly once.

## Saving batch results to another system

By default, an accepted batch result is stored without changing another data system.

**Write-back** is the optional step that saves an accepted result—such as a label or summary—to an approved destination. It is available only for batch jobs and only when all of the following match:

- write-back is enabled in the fixed configuration;
- the result passed the required validation version;
- an approved policy exists;
- a valid approval exists;
- the approval matches the job, item, destination, fields, and type of change; and
- recovery checks show that repeating the save will not create an uncertain duplicate change.

If write-back is disabled, storing the result is still considered successful. No destination record is changed.

Immediate requests can never reach the write-back component. This is enforced by how the application is assembled, not merely by a user setting.

The prototype writes only to a fake destination adapter. It does not update a real database.

## What happens when something goes wrong

The system follows a conservative rule: return only safe, accepted information and describe everything else as incomplete or failed.

| Situation | Outcome |
|---|---|
| The request does not clearly select immediate or batch processing | Rejected before data is read or an AI provider is called. |
| The fixed configuration cannot be loaded | Rejected before execution. |
| The database read fails | The immediate response reports that the source data is unavailable; AI work depending on it does not run. |
| Function options or source records are invalid | The item stops before provider disclosure. |
| The budget is unavailable | The provider is not called and the missing result is reported. |
| Security approval fails | The provider is not called and the rejection is recorded. |
| The provider response has the wrong shape or IDs | The proposed answer is withheld. |
| The time limit expires | Safe completed work may be returned; pending results are marked missing. |
| One batch item fails | Other eligible items continue. |
| Write-back is not approved | The destination is not changed; the result or failure remains recorded. |

A request can be **accepted for processing** and still fail later. Acceptance means only that the request was clear, its configuration was found, and it was sent down the selected path. It does not guarantee that the database read, AI function, validation, or write-back will succeed.

## Three example journeys

### Classify one customer note

1. A support application requests the approved `customers` read.
2. It selects `ai_classify` with `follow_up`, `resolved`, and `escalate` as the only allowed labels.
3. The system safely reads and standardizes the customer record.
4. The customer ID and note become the function input.
5. Cost and security checks pass.
6. The provider proposes a label.
7. The system confirms that the ID matches and the label is allowed.
8. The caller receives the accepted record and the original read result.

```powershell
python main.py interactive --request-id request-1 --query-plan customers `
  --config-version prototype --deadline-seconds 2 --execute `
  --task ai_classify --label follow_up --label resolved --label escalate
```

### Reject a bad summary

1. A caller requests `ai_summarize` with a 20-word limit.
2. The data read and provider call succeed.
3. The provider returns an answer that fails a required check.
4. The system returns the safe database result but withholds the summary.
5. The JSON output explains the failed check, and the task command exits with code `2`.

```powershell
python main.py interactive --request-id request-2 --query-plan customers `
  --config-version prototype --execute --task ai_summarize `
  --max-words 20 --fallback
```

### Summarize several records and recover progress

1. A job supplies several item IDs and their text.
2. Each item is prepared and processed separately.
3. Saved milestones allow completed stages to be reused after a simulated interruption.
4. Invalid items fail without discarding valid items.
5. With write-back disabled, accepted summaries are stored but no destination is changed.

```powershell
python main.py bulk --job-id summary-job --item customer-1 --item customer-2 `
  --config-version prototype --execute --task ai_summarize --max-words 20 `
  --write-back disabled --resume
```

The approved demonstration mode additionally checks authorization and duplicate-save recovery before sending a change to the fake destination:

```powershell
python main.py bulk --job-id bulk-job --item item-1 `
  --config-version prototype --execute --task ai_summarize --max-words 10 `
  --write-back approved
```

Task-aware commands exit with code `0` only when an immediate response is complete or every batch item succeeds. Otherwise, they exit with code `2`.

## What this prototype does and does not prove

### Demonstrated behavior

- One clear processing path is selected for each request.
- The governing rules remain fixed during an execution.
- Immediate database access is allowlisted and read-only.
- Function inputs and options are checked before provider use.
- Data is packaged within configured size limits.
- Budget and security controls run before provider disclosure.
- Proposed answers must pass strict structural checks.
- Immediate failures are visible rather than hidden.
- Batch items are isolated and can resume from saved in-process milestones.
- Saving results to another system requires separate batch-only approval.

### Not demonstrated or guaranteed

- Production database, AI-provider, message-broker, or monitoring integrations.
- Persistence across a machine or process restart.
- That every database vendor is supported without a tested adapter.
- That an accepted AI answer is factually correct, useful, fair, or unbiased.
- That every provider call or message delivery happens exactly once.
- That local limits prevent all provider rate-limit errors.
- That estimated usage equals the final provider bill.
- A measured production response-time target.
- Real database write-back; the prototype uses a fake destination.

## Business troubleshooting guide

Start with the last known business step rather than the internal class name.

1. **The request was rejected immediately:** confirm that it selected exactly one processing type and referenced an available configuration.
2. **No records were available:** check the approved query, supplied parameters, read permissions, and expected source columns. For batch jobs, check the source mapping submitted with each item.
3. **The AI work never started:** check function options, source shape, record size, budget availability, model eligibility, and security decisions.
4. **The provider answered but no result was returned:** check the response format, record IDs, allowed classification labels, and summary limits.
5. **An immediate request was incomplete:** inspect its missing-result reasons, time limit, and cancellation details.
6. **A batch item did not resume:** inspect its last saved milestone, item ownership, and saved data references.
7. **A result was not written back:** confirm that the result passed validation and that the configuration, policy, approval, item, destination, and fields all match.
8. **The command behaved unexpectedly:** confirm whether both `--execute` and `--task` were supplied. Without `--execute`, the prototype workflow does not run.
## Developer reference

Business readers can stop here. The following map connects the plain-language stages to the implementation.

| Business stage | Main implementation |
|---|---|
| Choose immediate or batch processing | `admission/router.py`, `control_plane/context.py` |
| Lock the execution rules | `control_plane/configuration.py`, `domain/configuration.py` |
| Read immediate source data | `relational/data_access.py`, `relational/plans.py`, `relational/normalization.py` |
| Map source rows into function inputs | `interactive/tasking.py` |
| Resolve and prepare the selected function | `tasks/registry.py`, `tasks/runtime.py`, `tasks/packing.py` |
| Select an eligible provider option | `model_routing/router.py`, `model_routing/capacity.py`, `interactive/integration.py` |
| Apply budget and security controls | `control_plane/budget.py`, `security/gateway.py` |
| Read and check proposed answers | `tasks/parser.py`, `tasks/validation.py`, `validation/validator.py` |
| Build the immediate response | `interactive/coordinator.py`, `interactive/aggregator.py` |
| Track and resume batch items | `bulk/coordinator.py`, `bulk/worker.py`, `bulk/execution.py`, `bulk/memory.py` |
| Authorize and recover write-back | `bulk/effects.py`, `write_back/executor.py`, `write_back/authorization.py` |
| Record decisions and outcomes | `observability/telemetry.py`, `prototype/adapters.py` |
| Assemble the executable prototype | `cli/app.py`, `composition/prototype.py` |

### Compatibility note

The repository also retains an older, generic model-work path. It accepts prebuilt model work and uses the shared routing, budget, security, and basic validation components, but it does not automatically gain the fixed classify/summarize definitions, strict function options, standard payload construction, or function-specific output parser.

A request cannot mix this generic work with the newer AI-function work. The generic batch demonstration also advances through milestones without making a real provider call. These compatibility details should not be used to describe the business behavior of `ai_classify` or `ai_summarize`.