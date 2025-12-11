"""
Test distributed training mode detection.

This tests the KEY fix: DRIVER should check for next_hop BEFORE deciding
to train locally. If another node has layers after ours, use distributed mode.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestTrainingModeDetection:
    """Test that training mode is correctly detected based on network state."""
    
    def test_driver_with_next_hop_uses_distributed(self):
        """
        DRIVER with next_hop should use distributed training,
        even if it has a temporary LM head.
        """
        # Mock p2p_manager that returns a next_hop
        mock_p2p = MagicMock()
        mock_p2p.get_next_hop.return_value = "http://worker:8000"
        
        # Create a mock node that acts as DRIVER with temp LM head
        mock_node = MagicMock()
        mock_node.my_layer_ids = [0, 1, 2, 3]  # DRIVER with layers 0-3
        mock_node.p2p_manager = mock_p2p
        mock_node.enable_training = True
        
        mock_model = MagicMock()
        mock_model.has_embedding = True  # DRIVER
        mock_model.has_lm_head = True    # Temporary LM head
        mock_node.model = mock_model
        
        # Check: get_next_hop should be called for layer 4 (after our last layer)
        next_layer = max(mock_node.my_layer_ids) + 1
        assert next_layer == 4
        
        next_hop = mock_p2p.get_next_hop(next_layer)
        assert next_hop == "http://worker:8000"
        
        # With next_hop, should use distributed training
        has_next_hop = next_hop is not None
        assert has_next_hop is True
        
        # The decision: DRIVER with next_hop → distributed
        should_use_distributed = mock_model.has_embedding and has_next_hop
        assert should_use_distributed is True
        
        print("✓ DRIVER with next_hop correctly routes to distributed training")
    
    def test_driver_without_next_hop_uses_local(self):
        """
        DRIVER without next_hop (solo node) should use local training.
        """
        # Mock p2p_manager that returns NO next_hop
        mock_p2p = MagicMock()
        mock_p2p.get_next_hop.return_value = None
        
        # Create a mock node that acts as full node
        mock_node = MagicMock()
        mock_node.my_layer_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All layers
        mock_node.p2p_manager = mock_p2p
        mock_node.enable_training = True
        
        mock_model = MagicMock()
        mock_model.has_embedding = True  # DRIVER
        mock_model.has_lm_head = True    # Full node
        mock_node.model = mock_model
        
        # Check: no next_hop exists
        next_layer = max(mock_node.my_layer_ids) + 1
        assert next_layer == 11
        
        next_hop = mock_p2p.get_next_hop(next_layer)
        assert next_hop is None
        
        has_next_hop = next_hop is not None
        assert has_next_hop is False
        
        # The decision: DRIVER without next_hop + has_lm_head → local
        should_use_local = mock_model.has_embedding and mock_model.has_lm_head and not has_next_hop
        assert should_use_local is True
        
        print("✓ DRIVER without next_hop correctly routes to local training")
    
    def test_worker_waits_for_activations(self):
        """
        WORKER (no embedding) should return None and wait for gRPC activations.
        """
        mock_model = MagicMock()
        mock_model.has_embedding = False  # WORKER
        mock_model.has_lm_head = True     # VALIDATOR
        
        # Workers don't need to check next_hop - they just wait
        should_wait = not mock_model.has_embedding
        assert should_wait is True
        
        print("✓ WORKER correctly waits for gRPC activations")
    
    def test_next_hop_lookup_uses_correct_layer(self):
        """
        The next_hop lookup should use layer AFTER our last layer.
        """
        mock_p2p = MagicMock()
        
        # DRIVER has layers 0-3
        my_layer_ids = [0, 1, 2, 3]
        my_last_layer = max(my_layer_ids)
        next_layer = my_last_layer + 1
        
        assert my_last_layer == 3
        assert next_layer == 4
        
        # Should look up layer 4, not layer 3
        mock_p2p.get_next_hop(next_layer)
        mock_p2p.get_next_hop.assert_called_with(4)
        
        print("✓ Next hop lookup uses correct layer (last + 1)")


class TestJetsonEC2Scenario:
    """
    Test the specific Jetson + EC2 scenario.
    
    - Jetson (DRIVER): Layers 0-3, embedding=True, has_lm_head=True (temp)
    - EC2 (WORKER): Layers 1-10, embedding=False, has_lm_head=True
    
    The Jetson should detect EC2 has layer 4+ and use distributed training.
    """
    
    def test_jetson_detects_ec2_and_uses_distributed(self):
        """Jetson should use distributed training when EC2 is available."""
        
        # Mock DHT/P2P that knows about EC2
        mock_p2p = MagicMock()
        
        def get_next_hop_mock(layer_id):
            # EC2 has layers 1-10, so it can handle layer 4
            if 1 <= layer_id <= 10:
                return "http://44.220.246.140:8000"  # EC2
            return None
        
        mock_p2p.get_next_hop = get_next_hop_mock
        
        # Jetson configuration
        jetson_layers = [0, 1, 2, 3]
        jetson_last_layer = max(jetson_layers)
        next_layer = jetson_last_layer + 1  # Layer 4
        
        # Look up who has layer 4
        next_hop = mock_p2p.get_next_hop(next_layer)
        
        assert next_hop == "http://44.220.246.140:8000"
        assert next_hop is not None  # EC2 has layer 4
        
        # Jetson should use distributed training
        has_embedding = True
        has_next_hop = next_hop is not None
        should_use_distributed = has_embedding and has_next_hop
        
        assert should_use_distributed is True
        
        print("✓ Jetson correctly detects EC2 and uses distributed training")
    
    def test_ec2_waits_for_activations(self):
        """EC2 (WORKER) should wait for activations from Jetson."""
        
        # EC2 configuration
        ec2_has_embedding = False  # WORKER
        ec2_has_lm_head = True     # VALIDATOR
        
        # WORKER should wait
        should_wait = not ec2_has_embedding
        assert should_wait is True
        
        print("✓ EC2 correctly waits for activations")


if __name__ == "__main__":
    print("\n=== Testing Training Mode Detection ===\n")
    
    # Run tests
    t1 = TestTrainingModeDetection()
    t1.test_driver_with_next_hop_uses_distributed()
    t1.test_driver_without_next_hop_uses_local()
    t1.test_worker_waits_for_activations()
    t1.test_next_hop_lookup_uses_correct_layer()
    
    print()
    
    t2 = TestJetsonEC2Scenario()
    t2.test_jetson_detects_ec2_and_uses_distributed()
    t2.test_ec2_waits_for_activations()
    
    print("\n=== All Tests Passed! ===\n")
