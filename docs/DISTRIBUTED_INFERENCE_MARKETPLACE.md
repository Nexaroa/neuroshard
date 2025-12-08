# Distributed Inference Marketplace (REVISED)

## Critical Issues with Original Design

### Issue 1: Privacy Violation ❌
**Problem:** Prompts stored in plain text in marketplace → All nodes can read them

**Solution:** Driver-centric model where prompts are sent directly to driver node

### Issue 2: Ignores Distributed Architecture ❌
**Problem:** Marketplace assumes single-node inference

**Solution:** Integrate with NeuroShard's pipeline parallelism (Driver → Workers → Validator)

---

## Revised Architecture

### Overview

NeuroShard inference is **inherently distributed**:
- **Driver nodes** (Layer 0): Hold embedding, see the prompt
- **Worker nodes** (Layers 1-N): Process activations (never see prompt)
- **Validator nodes** (LM Head): Generate final output

The marketplace must **orchestrate this pipeline** while maintaining privacy and fair pricing.

---

## Data Structures (REVISED)

```python
@dataclass
class InferenceRequest:
    """
    Marketplace request (PUBLIC metadata).
    
    Privacy: Prompt is NOT stored here!
    """
    request_id: str              # UUID
    user_id: str                 # User submitting
    driver_node_id: str          # Which driver will process
    
    # Pricing
    tokens_requested: int        # Max tokens to generate
    max_price: float             # Max NEURO per 1M user will pay
    locked_price: float          # Locked at submission
    
    # Pipeline state
    status: RequestStatus        # PENDING, DRIVER_CLAIMED, PROCESSING, COMPLETED
    pipeline_session_id: str     # Links distributed pipeline
    
    # Timestamps
    submitted_at: float
    claimed_at: Optional[float]
    completed_at: Optional[float]
    
    # Security
    user_signature: str          # ECDSA signature


@dataclass  
class PipelineSession:
    """
    Tracks distributed inference through the pipeline.
    
    Privacy: Only driver knows the prompt!
    """
    session_id: str              # UUID
    request_id: str              # Links to InferenceRequest
    driver_node_id: str          # Layer 0
    worker_node_ids: List[str]   # Layers 1-N
    validator_node_id: str       # LM Head
    
    # State
    current_layer: int           # Which layer is processing
    activations_hash: str        # Verify integrity
    
    # Proofs submitted
    driver_proof: Optional[str]
    worker_proofs: Dict[str, str]  # node_id -> proof signature
    validator_proof: Optional[str]
```

---

## Request Flow (PRIVACY-PRESERVING)

### Phase 1: User Submission

```python
# User-side
async def submit_inference_request(prompt: str, max_tokens: int):
    # 1. Choose a driver node (round-robin or by preference)
    driver_node = await discover_driver_node()
    
    # 2. Submit metadata to marketplace (NO PROMPT!)
    request_id, locked_price = await marketplace.submit_request(
        user_id=my_user_id,
        driver_node_id=driver_node.node_id,
        tokens_requested=max_tokens,
        max_price=1.0,
        user_signature=sign(request_id)
    )
    
    # 3. Send ENCRYPTED prompt directly to driver
    encrypted_prompt = encrypt_with_driver_pubkey(prompt, driver_node.public_key)
    
    await driver_node.submit_prompt(
        request_id=request_id,
        encrypted_prompt=encrypted_prompt
    )
    
    # 4. Wait for result
    result = await marketplace.wait_for_completion(request_id)
    return result
```

**Privacy:** Prompt never touches the marketplace! Only sent to driver node.

### Phase 2: Driver Claims and Starts Pipeline

```python
# Driver node
async def process_inference():
    # 1. Claim request from marketplace
    request = marketplace.claim_request(my_node_id)
    
    # 2. Retrieve encrypted prompt from local queue
    encrypted_prompt = self.pending_prompts[request.request_id]
    prompt = decrypt_with_my_privkey(encrypted_prompt)
    
    # 3. Run embedding layer
    input_ids = tokenize(prompt)
    activations = self.embedding_layer(input_ids)
    
    # 4. Find next worker in pipeline
    next_worker = dht.find_node_with_layer(layer=1)
    
    # 5. Create pipeline session
    session = PipelineSession(
        session_id=uuid.uuid4(),
        request_id=request.request_id,
        driver_node_id=my_node_id,
        current_layer=0
    )
    
    # 6. Forward activations to next layer
    await next_worker.forward(
        session_id=session.session_id,
        activations=activations
    )
    
    # 7. Submit driver proof
    proof = create_ponw_proof(
        request_id=request.request_id,
        tokens_processed=len(input_ids),
        has_embedding=True
    )
    await ledger.process_proof(proof)
```

### Phase 3: Workers Process Activations

```python
# Worker node (Layers 1-N)
async def on_receive_activations(session_id: str, activations: Tensor):
    # 1. Look up session
    session = pipeline_sessions[session_id]
    
    # 2. Process through my layers
    output = self.my_layers(activations)
    
    # 3. Find next node in pipeline
    next_node = dht.find_node_with_layer(layer=session.current_layer + 1)
    
    # 4. Forward to next layer
    await next_node.forward(
        session_id=session_id,
        activations=output
    )
    
    # 5. Submit worker proof
    proof = create_ponw_proof(
        request_id=session.request_id,
        tokens_processed=activations.numel(),
        layers_held=len(self.my_layers)
    )
    await ledger.process_proof(proof)
```

### Phase 4: Validator Completes

```python
# Validator node (LM Head)
async def on_receive_final_activations(session_id: str, activations: Tensor):
    # 1. Look up session
    session = pipeline_sessions[session_id]
    
    # 2. Generate output token
    logits = self.lm_head(activations)
    next_token = sample(logits)
    
    # 3. Check if generation complete
    if next_token == EOS or tokens_generated >= max_tokens:
        # 4. Return result to user
        await marketplace.complete_request(
            request_id=session.request_id,
            result=detokenize(output_tokens)
        )
    else:
        # Loop back to driver for next token (autoregressive)
        pass
    
    # 5. Submit validator proof
    proof = create_ponw_proof(
        request_id=session.request_id,
        tokens_processed=activations.numel(),
        has_lm_head=True
    )
    await ledger.process_proof(proof)
```

---

## Reward Distribution (ALREADY IMPLEMENTED!)

The good news: **Your existing reward calculation already handles this!**

```python
# ledger.py _calculate_reward()
if proof.has_embedding:  # Driver
    inference_reward += inference_pool * DRIVER_SHARE  # 15%

if proof.layers_held > 0:  # Workers
    inference_reward += inference_pool * WORKER_SHARE  # 70%

if proof.has_lm_head:  # Validator
    inference_reward += inference_pool * VALIDATOR_SHARE  # 15%
```

**Key insight:** Multiple proofs with the same `request_id`!
- Driver proof: `request_id=X, has_embedding=True` → Gets 15%
- Worker proofs: `request_id=X, layers_held=25` → Get 70% (divided)
- Validator proof: `request_id=X, has_lm_head=True` → Gets 15%

---

## Privacy Guarantees

| Data | Who Sees It | Why |
|------|-------------|-----|
| **Prompt** | Driver only | Encrypted, sent directly to driver |
| **Activations** | Workers | No semantic meaning, just vectors |
| **Output** | Validator → User | Encrypted channel |
| **Request metadata** | All nodes (marketplace) | Public: price, tokens, driver ID |

**Result:** Workers process activations without ever knowing what the prompt was! ✅

---

## Security Considerations

### 1. Malicious Driver
**Attack:** Driver sees prompt, could log it

**Mitigation:**
- User chooses driver (reputation system)
- Drivers stake NEURO (slashed if privacy violation proven)
- End-to-end encryption (user decrypts final output)

### 2. Pipeline Hijacking
**Attack:** Malicious node inserts itself in pipeline

**Mitigation:**
- DHT verifies layer ownership
- Activations signed by previous node
- Pipeline session ID prevents injection

### 3. Proof Replay
**Attack:** Node submits proof for request they didn't process

**Mitigation:**
- Proof includes `request_id` in signature
- Ledger tracks which nodes processed which requests
- Pipeline session validates participant list

---

## Implementation Changes Needed

### 1. Marketplace (`inference_market.py`)
```python
class InferenceRequest:
    # REMOVE: prompt field
    # ADD: driver_node_id, pipeline_session_id, status
    
def submit_request(
    user_id: str,
    driver_node_id: str,  # NEW: User chooses driver
    tokens_requested: int,
    max_price: float
) -> Tuple[bool, str, float]:
    # Create request WITHOUT prompt
    # Prompt sent directly to driver (out of band)
```

### 2. Driver Node (`dynamic_model.py`)
```python
class DynamicNeuroNode:
    def __init__(self):
        self.pending_prompts: Dict[str, bytes] = {}  # request_id -> encrypted_prompt
    
    async def receive_encrypted_prompt(self, request_id: str, encrypted_prompt: bytes):
        """User sends prompt directly to driver (not via marketplace)."""
        self.pending_prompts[request_id] = encrypted_prompt
    
    async def claim_and_process_request(self):
        """Driver claims request and starts pipeline."""
        request = marketplace.claim_request(self.node_id)
        encrypted_prompt = self.pending_prompts[request.request_id]
        prompt = decrypt(encrypted_prompt)
        
        # Start distributed inference pipeline
        await self.start_pipeline(request, prompt)
```

### 3. Pipeline Management (NEW)
```python
class PipelineManager:
    """Manages distributed inference sessions."""
    
    def start_session(self, request_id: str, driver_id: str):
        """Initialize pipeline for a request."""
        
    def forward_to_next_layer(self, session_id: str, activations: Tensor):
        """Route activations to next node in pipeline."""
        
    def on_completion(self, session_id: str, output: str):
        """Pipeline completed, return result to user."""
```

---

## Comparison: Before vs After

### BEFORE (Current Implementation) ❌
```
User → Marketplace (prompt visible!) → Single node → Response
```
**Problems:**
- Privacy leak (all nodes see prompt)
- Single-node inference (ignores distributed architecture)
- Doesn't use pipeline parallelism

### AFTER (Revised Design) ✅
```
User → Marketplace (metadata only)
     ↓
     → Driver (encrypted prompt) → Workers (activations) → Validator → User
                    ↓                     ↓                      ↓
              Proof (15%)           Proofs (70%)          Proof (15%)
```
**Benefits:**
- Privacy preserved (only driver sees prompt)
- Uses distributed pipeline (as designed!)
- Fair reward distribution (already implemented!)

---

## Next Steps

1. **Remove `prompt` from `InferenceRequest`**
2. **Add driver selection** to request submission
3. **Implement direct prompt channel** (user → driver, encrypted)
4. **Create `PipelineManager`** for session tracking
5. **Update tests** for distributed inference flow

**Estimated effort:** 4-6 hours (significant refactor)

---

## Key Insight

The marketplace should **orchestrate the pipeline**, not replace it!

- Marketplace: Handles pricing, claiming, completion
- Pipeline: Handles actual distributed computation
- Privacy: Achieved through driver-centric prompt handling

This aligns perfectly with NeuroShard's original vision of **decentralized, privacy-preserving AI**! 🚀

