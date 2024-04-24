
import random
import numpy as np
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from lenses_experiments.utils.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union

logger = get_logger(name = __name__)

DEBUG = True

def extract_trajectory_anomaly(dataloader: DataLoader, model: AutoModelForCausalLM, 
                           tokenizer: AutoTokenizer, lenses: Union[TunedLens,LogitLens]) -> torch.Tensor:
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Config
    n_layers = model.config.num_hidden_layers + 1
    n_vocab = model.config.vocab_size

    correct_sum = 0
    n_batches = len(dataloader)
    features = []

    logger.debug(f"CUDA memory: total available: {torch.cuda.get_device_properties(0).total_memory}, reserved: {torch.cuda.memory_reserved(0)}, allocated: {torch.cuda.memory_allocated(0)}")

    for batch in tqdm(dataloader):
        # Unpack the batch
        _, encoded_prompt, encoded_labels, correct_encoded_labels = batch

        # Preprocessing label tensors
        encoded_labels = torch.tensor(encoded_labels).squeeze()
        correct_encoded_labels = torch.tensor(correct_encoded_labels).squeeze()
        correct_indices = torch.tensor([torch.nonzero(encoded_labels[idx] == val).item()
                           for idx, val in enumerate(correct_encoded_labels)])

        inputs = encoded_prompt.input_ids
        attention_mask = encoded_prompt.attention_mask
        last_nonzero_idx = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Compute trajectory probabilities
            traj_log_probs = torch.zeros(len(inputs), n_layers, n_vocab)
            traj_log_probs[:, -1, :] = outputs.logits[range(len(inputs)), last_nonzero_idx, :].log_softmax(dim=-1)

            hidden_states_at_last_token = torch.stack([hs[range(len(inputs)), last_nonzero_idx, :] for hs in hidden_states[:-1]], dim=1)

            # Compute lens outputs
            for layer_index in range(n_layers - 1):
                lens_output = lenses.forward(hidden_states_at_last_token[:, layer_index, :], layer_index)
                traj_log_probs[:, layer_index, :] = lens_output.log_softmax(dim=-1)

            # Filter probabilities only for choices + determine correctness
            traj_log_probs_filtered = torch.stack([batch_probs[:, labels] for batch_probs, labels in zip(traj_log_probs, encoded_labels)], dim=0)
            correct_answers = (traj_log_probs_filtered[:, -1, :].argmax(dim=1) == correct_indices).float()
            correct_batch = correct_answers.mean().item()
            correct_sum += correct_batch
            logger.debug(f"correct answer ratio for this batch: {correct_batch}")

            features.append(traj_log_probs_filtered.flatten(start_dim=1))

    overall_accuracy = correct_sum / n_batches
    logger.debug(f"Overall correct answer ratio: {overall_accuracy}")
    return torch.cat(features, dim=0)

#NewLens

#forward(hidden_states, layer_index)

#lens_output = lenses.forward(hidden_states_at_last_token[:, layer_index, :], layer_index)

class NewLens:
    def __init__(self, model: AutoModelForCausalLM):
        self._model = model 

    def forward(self, cache_to_patch, layer_index):
        """apply newLens to the model by providing the hidden states and its corresponding layer
        example usage: lenses.forward(hidden_states_at_last_token[:, layer_index, :], layer_index) #hidden states: batch x d_model

        Args:
            cache_to_patch (_type_): _description_
            layer_index (_type_): _description_

        Returns:
            _type_: _description_
        """
        #cache_to_patch : batch x d_model
        n_batch = cache_to_patch.shape[0]

        def hook(
            model,input,output
        ):
            #output[0] here is the output tensor of the block n - 1, 
            #which corresponds to the hidden state of layer n
            return (cache_to_patch.unsqueeze(1), output[1])
        
        if(layer_index == 0):
            out = self._model(inputs_embeds = cache_to_patch.unsqueeze(1))
            return out.logits.squeeze()
        
        #layer_index > 0, so cache_to_patch is a hidden state and not an embedding
        handle = self._model.gpt_neox.layers[layer_index - 1].register_forward_hook(hook)
        fake_inputs = torch.zeros(n_batch, 1, device="cpu").int().to(self._model.device) #patching with batch x position long 1
        with torch.no_grad():
            out = self._model(input_ids = fake_inputs)
        #assert torch.allclose(out.hidden_states[layer_index], output[0] of the hook, atol=1e-4)
        handle.remove()
        return out.logits.squeeze()
    
class AnomalyDetectionAlgos:
    def __init__(self, correct_features: torch.Tensor, injected_features: torch.Tensor):
        #just use half for training and half for validation

        X = correct_features[:correct_features.shape[0]//2]
        val_correct = correct_features[correct_features.shape[0]//2:]
        val_incorrect = injected_features[injected_features.shape[0]//2:]
        val = torch.cat([val_correct, val_incorrect], dim=0)
        labels = torch.cat([torch.ones(val_correct.shape[0]), torch.zeros(val_incorrect.shape[0])], dim=0)
        indices = torch.randperm(val.size(0))

        # Shuffle features and labels using the random indices
        shuffled_val = val[indices]
        shuffled_labels = labels[indices]
        shuffled_labels = torch.where(shuffled_labels == 0, torch.tensor(-1), shuffled_labels)

        X_arr = X.cpu().numpy()
        shuffled_val_arr = shuffled_val.cpu().numpy()
        shuffled_labels_arr = shuffled_labels.cpu().numpy()

        aurocs_lof = []
        accuracy_lof = []
        for seed in range(10):
            estimator_lof = LocalOutlierFactor(novelty=True, metric="cosine").fit(X_arr)
            predictions_lof = estimator_lof.predict(shuffled_val_arr)
            accuracy_lof.append(metrics.accuracy_score(y_true = shuffled_labels_arr, y_pred = predictions_lof))
            aurocs_lof.extend(self._bootstrap_auroc_calculate(shuffled_labels_arr, estimator_lof.decision_function(shuffled_val_arr)))
        self._lof_acc = np.mean(accuracy_lof)
        self._lof_roc = np.quantile(aurocs_lof, [0.025, 0.5, 0.975])

        aurocs_if = []
        for seed in range(10):
            estimator_if = IsolationForest(random_state=seed).fit(X_arr)
            predictions_if = estimator_if.score_samples(shuffled_val_arr)
            aurocs_if.extend(self._bootstrap_auroc_calculate(shuffled_labels_arr, predictions_if))
        self._if_roc = np.quantile(aurocs_if, [0.025, 0.5, 0.975])
    
    @property
    def if_roc(self):
        return self._if_roc

    @property
    def lof_roc(self):
        return self._lof_roc

    @property
    def lof_accuracy(self):
        return self._lof_acc

    #copied from Tuned Lens code repository https://github.com/AlignmentResearch/tuned-lens.git
    def _bootstrap_auroc_calculate(
    self, labels: np.ndarray, scores: np.ndarray, num_samples: int = 1000, seed: int = 0
    ) -> list[float]:
        from sklearn.metrics import roc_auc_score

        rng = random.Random(seed)
        n = len(labels)
        aurocs = []

        for _ in range(num_samples):
            idx = rng.choices(range(n), k=n)
            aurocs.append(roc_auc_score(labels[idx], scores[idx]))

        return aurocs