from unittest import TestCase
import os
from databases.dataset import ImageDatabase



class TestDatabase(TestCase):
    dataset_folder = "../datasets"
    def test_load_all_images(self):
        dataset_folder = os.path.join(self.dataset_folder, "dice")
        database = ImageDatabase(dataset_folder)
        assert len(database.images) > 0
