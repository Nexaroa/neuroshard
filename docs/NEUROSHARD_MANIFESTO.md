# NeuroShard: The Decentralized "Living" AI

> **Vision:** A global, unstoppable, self-evolving neural network where computing power, training, and ownership are distributed across millions of consumer devices.

---

## 1. The Concept

**NeuroShard** fundamentally reimagines how Large Language Models (LLMs) are run and trained. 

Currently, AI is centralized. To run GPT-4, you need a massive data center with thousands of interconnected H100 GPUs. This creates a monopoly on intelligence.

**NeuroShard breaks the model apart.** 
Instead of one server holding the entire model, we shard the model's layers across a peer-to-peer (P2P) network.
*   **Alice (Node A)** holds Layers 1-4.
*   **Bob (Node B)** holds Layers 5-8.
*   **Charlie (Node C)** holds Layers 9-12.

When a user sends a prompt, the tokens physically travel through the internet from Alice → Bob → Charlie. The "intelligence" emerges from the collective relay of these nodes.

---

## 2. Architecture Implementation (v1.0)

We have successfully built a working prototype of this architecture using **Pipeline Parallelism** over HTTP.

### 🏛️ The Node (`p2p_node.py`)
Every participant runs a "Shard Node". The node is:
1.  **Autonomous:** It discovers peers via a Gossip Protocol (no central server).
2.  **Specialized:** It loads only a slice of the model (e.g., GPT-2 Layers 0-4) to save RAM.
3.  **Intelligent:** It uses real Transformer blocks (PyTorch/HuggingFace) to process hidden states.

### 🔗 The Protocol
1.  **Forward Pass (Inference):**
    *   Input tokens arrive at an *Entry Node*.
    *   The node processes them through its transformer blocks.
    *   It looks up its `KNOWN_PEERS` table to find a node responsible for the *next* set of layers.
    *   It serializes the activation tensor (Base64) and sends it via HTTP.
2.  **Consensus (Trust):**
    *   In a trustless network, nodes might lie (return random noise to save power).
    *   **Redundant Execution:** High-value requests are sent to *multiple* nodes (e.g., 3 random nodes for Layer 5-8).
    *   **Majority Vote:** The next node waits for 3 results, hashes them, and picks the majority winner. Malicious nodes are "slashed" (reputation lowered).

### 📉 Distributed Training (The "Living" Aspect)
*   We implemented a **backward pass relay**.
*   Gradients flow in reverse: `Node C -> Node B -> Node A`.
*   Each node updates its own weights using local optimizers.
*   This allows the model to learn continuously from user interactions, evolving over time.

### 🌱 The "Genesis Block" and Network Bootstrap
In a decentralized network, there is a "Chicken and Egg" problem: how do you start a network with zero nodes?

NeuroShard solves this with a **Genesis Node** (The Observer):
1.  **Bootstrap Phase:** The project hosting infrastructure runs a permanent "Observer Node" (Node #1).
2.  **Genesis Block:** When this node starts, it initializes the ledger and begins the first epoch.
3.  **Mining:** It acts as a standard participant, processing its own "heartbeat" transactions (Proof of Uptime) to secure the chain and mint the initial supply of NEURO tokens.
4.  **Expansion:** As new users download the software and join, they connect to this Genesis Node (or any other active peer via DHT) to sync the ledger and begin their own contribution. This is why the ledger shows activity even before external users join - the heartbeat of the network has already begun.

---

## 3. Technical Specifications

| Component | Implementation Details |
| :--- | :--- |
| **Model** | GPT-2 (124M) sharded into variable-sized chunks. |
| **Transport** | HTTP/1.1 JSON (Base64 encoded Tensors). |
| **Discovery** | P2P Gossip Protocol (Random peer exchange). |
| **Consensus** | Majority Voting on Output Hashes. |
| **Trust** | Local Reputation Table (`local_trust_scores`). |

---

## 4. The Vision: Where We Go Next

### Phase 1: The "Swarm" (Current)
*   Nodes can run parts of GPT-2.
*   Basic reputation system.
*   Manual bootstrapping.

### Phase 2: The "Global Brain" (Next Steps)
*   **DHT (Kademlia):** Replace gossip with a structured Distributed Hash Table so we can find "Layer 98" instantly among 1,000,000 nodes.
*   **Quantization:** Compress 16-bit floats to 4-bit integers to reduce bandwidth usage by 4x.
*   **KV-Cache Passing:** Optimize autoregressive generation by passing the Key/Value cache between nodes (complex but necessary for speed).

### Phase 3: The "Economic Layer"
*   **Crypto Incentives:** Integrate a ledger (Solana/Ethereum L2).
*   **Proof of Inference:** Cryptographic receipts proving a node actually ran the matrix multiplications (ZK-SNARKs or Optimistic Fraud Proofs).
*   **DAO Governance:** Token holders vote on the model's "Constitution" (alignment, safety filters).

---

## 5. Project Status

*   ✅ **Core Pipeline:** Working (GPT-2 running across 3 processes).
*   ✅ **Decentralization:** Working (Gossip, Peer Discovery).
*   ✅ **Consensus:** Working (Malicious node detection).
*   🚧 **Production Scale:** Needs DHT and NAT Traversal.
*   🚧 **UI:** Needs a Chat Interface.

---

## 6. How to Join the Network

### Developers
Clone the repo and run a node:
```bash
# Install
pip install -r requirements.txt

# Run a Layer 1 Node
python p2p_node.py --port 8000 --start 0 --end 4 --entry

# Run a Layer 2 Node (connect to Layer 1)
python p2p_node.py --port 8001 --start 4 --end 8 --peers http://localhost:8000
```

### Users
Run the client to chat with the swarm:
```bash
python client.py --prompt "The future of AI is"
```

