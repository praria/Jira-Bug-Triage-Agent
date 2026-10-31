"""Retrieval-augmented triage.

``embeddings`` turns a bug's summary and description into a vector.
``retrieval`` finds the closest stored bugs for a new bug.
``augmentation`` turns those bugs into the recommendation shown for approval.
"""
