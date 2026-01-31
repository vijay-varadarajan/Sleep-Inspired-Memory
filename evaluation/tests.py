"""
Evaluation and Testing

Basic tests to validate the memory consolidation system.
"""

import unittest
from datetime import datetime, timedelta
import os
import tempfile
import shutil

from memory.episodic import EpisodicMemoryStore, Episode
from memory.consolidated import ConsolidatedMemoryStore
from memory.schema import SchemaStore
from sleep.replay import (
    calculate_replay_priority,
    select_episodes_for_replay,
    calculate_batch_diversity
)
from sleep.compression import estimate_novelty


class TestEpisodicMemory(unittest.TestCase):
    """Test episodic memory storage."""
    
    def setUp(self):
        self.store = EpisodicMemoryStore()
    
    def test_add_episode(self):
        """Test adding episodes."""
        episode = self.store.add_episode(
            content="Test content",
            importance=0.8,
            novelty=0.6,
            tags=["test"]
        )
        
        self.assertIsNotNone(episode.id)
        self.assertEqual(episode.content, "Test content")
        self.assertEqual(episode.importance, 0.8)
        self.assertEqual(len(self.store.episodes), 1)
    
    def test_get_unconsolidated(self):
        """Test retrieving unconsolidated episodes."""
        ep1 = self.store.add_episode("Episode 1")
        ep2 = self.store.add_episode("Episode 2")
        
        # Both should be unconsolidated
        unconsolidated = self.store.get_unconsolidated()
        self.assertEqual(len(unconsolidated), 2)
        
        # Mark one as consolidated
        self.store.mark_consolidated(ep1.id)
        unconsolidated = self.store.get_unconsolidated()
        self.assertEqual(len(unconsolidated), 1)
        self.assertEqual(unconsolidated[0].id, ep2.id)
    
    def test_mark_accessed(self):
        """Test access tracking."""
        episode = self.store.add_episode("Test")
        
        self.assertEqual(episode.access_count, 0)
        self.assertIsNone(episode.last_access)
        
        self.store.mark_accessed(episode.id)
        self.assertEqual(episode.access_count, 1)
        self.assertIsNotNone(episode.last_access)
    
    def test_decay(self):
        """Test memory decay."""
        # Add old episode
        old_episode = self.store.add_episode(
            "Old episode",
            importance=0.2  # Low importance
        )
        # Manually set old timestamp
        old_episode.timestamp = datetime.now() - timedelta(days=40)
        
        # Add recent episode
        recent_episode = self.store.add_episode("Recent episode")
        
        # Run decay
        forgotten = self.store.decay_episodes(decay_threshold_days=30)
        
        # Old, low-importance episode should be forgotten
        self.assertIn(old_episode.id, forgotten)
        self.assertNotIn(recent_episode.id, self.store.episodes)
        self.assertIn(recent_episode.id, self.store.episodes)


class TestConsolidatedMemory(unittest.TestCase):
    """Test consolidated memory storage."""
    
    def setUp(self):
        self.store = ConsolidatedMemoryStore()
    
    def test_add_memory(self):
        """Test adding consolidated memories."""
        memory = self.store.add_memory(
            summary="Test summary",
            source_episode_ids=["ep1", "ep2"],
            key_concepts=["test", "memory"],
            importance=0.7
        )
        
        self.assertIsNotNone(memory.id)
        self.assertEqual(memory.summary, "Test summary")
        self.assertEqual(len(memory.source_episode_ids), 2)
        self.assertEqual(len(self.store.memories), 1)
    
    def test_search_by_concepts(self):
        """Test concept-based search."""
        self.store.add_memory(
            summary="About ML",
            source_episode_ids=["ep1"],
            key_concepts=["machine-learning", "AI"]
        )
        self.store.add_memory(
            summary="About cooking",
            source_episode_ids=["ep2"],
            key_concepts=["cooking", "food"]
        )
        
        # Search for ML concepts
        results = self.store.search_by_concepts(["machine-learning"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary, "About ML")


class TestSchemaStore(unittest.TestCase):
    """Test schema storage."""
    
    def setUp(self):
        self.store = SchemaStore()
    
    def test_add_schema(self):
        """Test adding schemas."""
        schema = self.store.add_schema(
            name="Test Schema",
            description="A test schema",
            core_concepts=["concept1", "concept2"],
            related_memory_ids=["mem1", "mem2"],
            confidence=0.7
        )
        
        self.assertIsNotNone(schema.id)
        self.assertEqual(schema.name, "Test Schema")
        self.assertEqual(len(self.store.schemas), 1)
    
    def test_find_by_concepts(self):
        """Test concept-based schema search."""
        self.store.add_schema(
            name="ML Schema",
            description="ML patterns",
            core_concepts=["ml", "ai", "learning"],
            related_memory_ids=["m1"]
        )
        self.store.add_schema(
            name="Cooking Schema",
            description="Cooking patterns",
            core_concepts=["cooking", "food"],
            related_memory_ids=["m2"]
        )
        
        results = self.store.find_by_concepts(["ml", "ai"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "ML Schema")
    
    def test_update_schema(self):
        """Test schema updates."""
        schema = self.store.add_schema(
            name="Test",
            description="Test",
            core_concepts=["test"],
            related_memory_ids=["m1"],
            confidence=0.5
        )
        
        success = self.store.update_schema(
            schema.id,
            new_memory_ids=["m2"],
            confidence_boost=0.2
        )
        
        self.assertTrue(success)
        updated = self.store.get_schema(schema.id)
        self.assertEqual(len(updated.related_memory_ids), 2)
        self.assertEqual(updated.confidence, 0.7)


class TestReplayMechanism(unittest.TestCase):
    """Test prioritized replay."""
    
    def test_priority_calculation(self):
        """Test priority calculation for episodes."""
        now = datetime.now()
        
        # Recent, important, novel episode
        ep1 = Episode(
            content="Important recent episode",
            timestamp=now - timedelta(hours=1),
            importance=0.9,
            novelty=0.8
        )
        
        # Old, unimportant episode
        ep2 = Episode(
            content="Old episode",
            timestamp=now - timedelta(days=30),
            importance=0.2,
            novelty=0.3
        )
        
        priority1 = calculate_replay_priority(ep1, now)
        priority2 = calculate_replay_priority(ep2, now)
        
        # Recent important episode should have higher priority
        self.assertGreater(priority1, priority2)
    
    def test_episode_selection(self):
        """Test episode selection for replay."""
        now = datetime.now()
        
        episodes = [
            Episode(content=f"Episode {i}", importance=0.5 + i*0.1, novelty=0.5)
            for i in range(10)
        ]
        
        selected = select_episodes_for_replay(
            episodes=episodes,
            n_replay=5,
            current_time=now
        )
        
        self.assertEqual(len(selected), 5)
        
        # Should be sorted by priority (descending)
        priorities = [priority for _, priority in selected]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
    
    def test_batch_diversity(self):
        """Test batch diversity calculation."""
        episodes = [
            Episode(content="Ep1", tags=["ml", "python"]),
            Episode(content="Ep2", tags=["cooking", "food"]),
            Episode(content="Ep3", tags=["ml", "ai"]),
        ]
        
        diversity = calculate_batch_diversity(episodes)
        self.assertGreater(diversity, 0)
        self.assertLessEqual(diversity, 1)


class TestNoveltyEstimation(unittest.TestCase):
    """Test novelty estimation."""
    
    def test_novelty_calculation(self):
        """Test novelty estimation from concept overlap."""
        existing = ["python", "programming", "function"]
        
        # Completely new concepts
        new_concepts = ["cooking", "recipe"]
        novelty1 = estimate_novelty(new_concepts, existing)
        self.assertGreater(novelty1, 0.7)  # Should be high novelty
        
        # Overlapping concepts
        overlap_concepts = ["python", "programming"]
        novelty2 = estimate_novelty(overlap_concepts, existing)
        self.assertLess(novelty2, 0.5)  # Should be low novelty


class TestPersistence(unittest.TestCase):
    """Test saving and loading memory stores."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_episodic_persistence(self):
        """Test saving and loading episodic memory."""
        store1 = EpisodicMemoryStore()
        ep1 = store1.add_episode("Test episode", importance=0.8)
        
        filepath = os.path.join(self.temp_dir, "episodic.json")
        store1.save_to_file(filepath)
        
        store2 = EpisodicMemoryStore()
        store2.load_from_file(filepath)
        
        self.assertEqual(len(store2.episodes), 1)
        loaded_ep = list(store2.episodes.values())[0]
        self.assertEqual(loaded_ep.content, "Test episode")
        self.assertEqual(loaded_ep.importance, 0.8)
    
    def test_consolidated_persistence(self):
        """Test saving and loading consolidated memory."""
        store1 = ConsolidatedMemoryStore()
        store1.add_memory(
            summary="Test summary",
            source_episode_ids=["ep1"],
            key_concepts=["test"]
        )
        
        filepath = os.path.join(self.temp_dir, "consolidated.json")
        store1.save_to_file(filepath)
        
        store2 = ConsolidatedMemoryStore()
        store2.load_from_file(filepath)
        
        self.assertEqual(len(store2.memories), 1)
        loaded_mem = list(store2.memories.values())[0]
        self.assertEqual(loaded_mem.summary, "Test summary")


def run_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  Running Sleep-Inspired Memory System Tests")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEpisodicMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestConsolidatedMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaStore))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayMechanism))
    suite.addTests(loader.loadTestsFromTestCase(TestNoveltyEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestPersistence))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
