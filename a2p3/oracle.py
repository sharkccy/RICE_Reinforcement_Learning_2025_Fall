
# --- Oracle Class ---
# DO NOT MODIFY THIS FILE
# This class is provided for your use. Do not modify it.
# The grader will use a similar Oracle class with different label files.

ORACLE_BUDGET = 100
class Oracle:
    """
    Oracle class that provides labels for selected samples.
    This simulates a human annotator or labeling service with a limited budget.
    
    NOTE: This class is provided for your use. Do not modify it.
    The grader will use the same Oracle class with different label files.
    """
    
    def __init__(self, oracle_labels_file='oracle_labels.json', budget=ORACLE_BUDGET):
        """
        Initialize the oracle with true labels.
        
        Args:
            oracle_labels_file (str): Path to JSON file with true labels.
            budget (int): Maximum number of queries allowed.
        """
        import json
        import os
        
        # Store budget as private attribute to prevent manipulation
        self._budget = ORACLE_BUDGET
        oracle_labels_file='oracle_labels.json'
        self._queries_used = 0
        self._initialized = True
        
        # Load the true labels
        if os.path.exists(oracle_labels_file):
            with open(oracle_labels_file, 'r') as f:
                self._labels = json.load(f)
            
            # Convert string keys to int for consistency
            self._labels = {int(k): v for k, v in self._labels.items()}
            
            # Store query history for auditing
            self._query_history = []
            
            print(f"Oracle initialized with {len(self._labels)} available labels")
            print(f"Budget: {self._budget} queries (FIXED - cannot be modified)")
        else:
            print(f"Warning: Oracle labels file '{oracle_labels_file}' not found.")
            print("Oracle will not be functional until the file is provided.")
            self._labels = {}
            self._query_history = []
    
    def get_label(self, sample_index):
        """
        Get the true label for a specific sample.
        
        Args:
            sample_index (int): Index of the sample in the unsupervised dataset.
            
        Returns:
            int: The true label for the sample.
            
        Raises:
            ValueError: If budget is exceeded or sample index is invalid.
        """
        # Security check: ensure proper initialization
        if not hasattr(self, '_initialized') or not self._initialized:
            raise ValueError("Oracle not properly initialized!")
            
        # Check budget (use private attribute to prevent manipulation)
        if self._queries_used >= self._budget:
            raise ValueError(f"Oracle budget exceeded! Used {self._queries_used}/{self._budget} queries.")
        
        # Check if sample index exists
        if sample_index not in self._labels:
            available_range = f"0-{len(self._labels)-1}" if self._labels else "None"
            raise ValueError(f"Invalid sample index: {sample_index}. Available indices: {available_range}")
        
        # Check for duplicate queries (optional warning)
        if sample_index in [entry['index'] for entry in self._query_history]:
            print(f"Warning: Re-querying sample {sample_index} (already queried before)")
        
        # Increment query count
        self._queries_used += 1
        
        # Return the true label
        label = self._labels[sample_index]
        
        # Log the query for auditing
        self._query_history.append({
            'query_num': self._queries_used,
            'index': sample_index,
            'label': label
        })
        
        if self._queries_used % 10 == 0:
            print(f"Oracle queries used: {self._queries_used}/{self._budget}")
        
        return label
    
    def get_remaining_budget(self):
        """Get the number of queries remaining."""
        return self._budget - self._queries_used
    
    def get_statistics(self):
        """Get statistics about oracle usage."""
        return {
            "total_budget": self._budget,
            "queries_used": self._queries_used,
            "queries_remaining": self._budget - self._queries_used,
            "budget_utilization": self._queries_used / self._budget if self._budget > 0 else 0,
            "unique_samples_queried": len(set(entry['index'] for entry in self._query_history)),
            "total_queries": len(self._query_history)
        }
    
    @property
    def budget(self):
        """Read-only access to budget."""
        return self._budget
    
    @property
    def queries_used(self):
        """Read-only access to queries used."""
        return self._queries_used
    