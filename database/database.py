import sys
sys.path.append('../')
import time
import numpy as np
import os
import Levenshtein
from typing import Optional, List

class Database:
    
    def __init__(self, path):
        self.path = path
        self.registers = []
        if os.path.exists(path):
            self.registers = np.load(path, allow_pickle=True)['registers']
            
    def remove(self, name:str):
        """
        Remove register from database
        """
        self.registers = [register for register in self.registers if register['name'] != name]  
        np.savez(self.path, **{'registers': self.registers})       
        
    def register(self, name:str, plates:List[str], embedding):
        """
        Add new register to database
        """
        names = [register['name'] for register in self.registers]
        assert name not in names, "Name already registered."
        
        self.registers = np.append(self.registers, {
            "name": name,
            "plates": plates,
            "embedding": embedding
        })
        
        np.savez(self.path, **{'registers': self.registers})  
        
    def _similarities(self, embedding:np.ndarray, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute the cosine similarities between the input embedding and the database vectors.
        """
        sims = (
            embedding @ (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).T
        ).T
        return sims
    
    def query_face(self, query_embedding:np.ndarray, top_k:Optional[int]=1):
        """
        Query the database for the k most similar faces.
        """
        embeddings = np.array([register['embedding'] for register in self.registers])
        sims = self._similarities(query_embedding, embeddings)
        indices = np.argsort(sims)[::-1][:top_k]
        results = list(zip(self.registers[indices], sims[indices]))
        return results
    
    def query_plate(self, query_plate:str, top_k:Optional[int]=1):
        """
        Query the database for the k most similar plates.
        """
        distances = []
        plates = []
        for register in self.registers:
            distances_ = [Levenshtein.distance(query_plate, plate) for plate in register['plates']]
            min_idx = np.argmin(distances_)
            distances.append(distances_[min_idx])
            plates.append(register['plates'][min_idx])
            
        indices = np.argsort(distances)[:top_k]
        results = list(zip(np.array(plates)[indices], np.array(distances)[indices]))
        return results
    

if __name__ == "__main__":
    
    database = Database('./database.npz')
    
    # Register
    database.register(
        name="joao",
        plates=["A3C5f34", "ABC1234"],
        image="../data/test_single_face.jpg"
    )
    database.register(
        name="lucas",
        plates=["EFG5678"],
        image="../data/test_lucas.png"
    )
    
    # Query face
    # face_recognition_pipeline = FaceRecognizer(use_colors=True)
    # embedding = face_recognition_pipeline('../data/test_lucas.png')[0].embedding()
    # results = database.query_face(embedding, 2)
    # for result in results:
    #     print(result[0]['name'], result[1])
    
    # Query plate
    # query_plate = "EFG5677"
    # results = database.query_plate(query_plate, 1)
    # for result in results:
    #     print(result[0]['plates'], result[1])