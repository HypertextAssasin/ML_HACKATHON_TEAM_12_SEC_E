import numpy as np
import joblib

def diagnose_hmm_emission(hmm_models):
    """
    Diagnose the emission probabilities of trained HMM models.
    Prints the emission probabilities for a few word lengths and checks if they are uniform.
    """
    if not hmm_models:
        print("No trained models found.")
        return

    for length, model in hmm_models.items():
        print(f"\n--- Emission probabilities for word length {length} ---")
        # Show first 5x5 block of emission probabilities
        print("First 5x5 block of emissionprob_:")
        print(model.emissionprob_[:5, :5])

        # Compute entropy for each hidden state
        entropy = -np.sum(model.emissionprob_ * np.log(model.emissionprob_ + 1e-12), axis=1)
        print(f"Entropy per hidden state: {entropy}")
        print(f"Mean entropy: {np.mean(entropy):.3f}")
        print(f"Log(26): {np.log(26):.3f}")

        # If mean entropy is close to log(26) (~3.258), emission probabilities are nearly uniform
        if np.abs(np.mean(entropy) - np.log(26)) < 0.1:
            print("Warning: Emission probabilities are nearly uniform. HMM may not be learning meaningful patterns.")
        else:
            print("Emission probabilities show meaningful variation.")

if __name__ == "__main__":
    # Load hmm models dict from pkl file
    hmm_models = joblib.load("hmm_models.pkl")
    
    diagnose_hmm_emission(hmm_models)
