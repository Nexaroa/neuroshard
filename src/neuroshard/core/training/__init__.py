# neuroshard/core/training/__init__.py
"""
Training coordination components for NeuroShard.

- distributed: TrainingCoordinator, FederatedDataManager, GenesisDataLoader
- production: GradientCompressor (for DiLoCo gradient exchange)
- global_tracker: GlobalTrainingTracker (for training verification)
"""

__all__ = [
    'TrainingCoordinator',
    'FederatedDataManager',
    'GenesisDataLoader',
    'GradientCompressor',
    'GlobalTrainingTracker',
]

def __getattr__(name):
    """Lazy loading of submodules."""
    if name in ('TrainingCoordinator', 'FederatedDataManager', 'GenesisDataLoader', 'GradientContribution'):
        from neuroshard.core.training import distributed
        return getattr(distributed, name)
    elif name in ('GradientCompressor', 'CompressionConfig', 'CompressionMethod'):
        from neuroshard.core.training import production
        return getattr(production, name)
    elif name in ('GlobalTrainingTracker', 'TrainingSnapshot', 'GlobalTrainingStats'):
        from neuroshard.core.training import global_tracker
        return getattr(global_tracker, name)
    raise AttributeError(f"module 'neuroshard.core.training' has no attribute '{name}'")
