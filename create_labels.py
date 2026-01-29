import pickle

# Map the label IDs to names (same as your dataset folders)
labels = {
    0: "Akansha",
    1: "Friend1"   # Add more if you have more people
}

# Save to pickle
with open("labels.pickle", "wb") as f:
    pickle.dump(labels, f)

print("✅ labels.pickle created successfully")
