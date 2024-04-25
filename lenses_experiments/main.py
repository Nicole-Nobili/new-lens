
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens
from datasets import load_dataset
from lenses_experiments.utils.logger import get_logger
from lenses_experiments.utils.datasets import Loader
from lenses_experiments.utils.anomaly_detection import extract_trajectory_anomaly
from lenses_experiments.utils.anomaly_detection import AnomalyDetectionAlgos, NewLens

if __name__ == '__main__':

    logger = get_logger(name = __name__)
    logger.info("Initializing logger...")

    parser = argparse.ArgumentParser()
    #"EleutherAI/pythia-410m-deduped"
    parser.add_argument('model_name', type=str, help="Model name")
    parser.add_argument('batch_size', type=int, default=32, help="Batch size")
    args = parser.parse_args()

    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit = True, low_cpu_mem_usage = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
    new_lens = NewLens(model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")

    loader = Loader("allenai/ai2_arc", dataset, model.device, tokenizer, args.batch_size)

    logger.info("Extracting features for Tuned Lens...")
    
    normal_features = extract_trajectory_anomaly(loader.loader_normal, model, tokenizer, tuned_lens)
    injected_features = extract_trajectory_anomaly(loader.loader_injected, model, tokenizer, tuned_lens)

    anomaly_detection_algo = AnomalyDetectionAlgos(correct_features=normal_features, injected_features=injected_features)
    lof_acc = anomaly_detection_algo.lof_accuracy
    lo_lof, mi_lof, hi_lof = anomaly_detection_algo.lof_roc
    lo_if, mi_if, hi_if = anomaly_detection_algo.if_roc

    with open("tuned_lens_results.txt", "w") as f:
        f.write(f"LOF accuracy: {lof_acc}\n")
        f.write(f"LOF ROC:\n{mi_lof:.2f}\\;({lo_lof:.2f}, {hi_lof:.2f})\n")
        f.write(f"Isolation Forest ROC:\n{mi_if:.2f}\\;({lo_if:.2f}, {hi_if:.2f})")

    
    logger.info("Extracting features for New Lens...")
    
    normal_features = extract_trajectory_anomaly(loader.loader_normal, model, tokenizer, new_lens)
    injected_features = extract_trajectory_anomaly(loader.loader_injected, model, tokenizer, new_lens)

    anomaly_detection_algo = AnomalyDetectionAlgos(correct_features=normal_features, injected_features=injected_features)
    lof_acc = anomaly_detection_algo.lof_accuracy
    lo_lof, mi_lof, hi_lof = anomaly_detection_algo.lof_roc
    lo_if, mi_if, hi_if = anomaly_detection_algo.if_roc

    with open("new_lens_results.txt", "w") as f:
        f.write(f"LOF accuracy: {lof_acc}\n")
        f.write(f"LOF ROC:\n{mi_lof:.2f}\\;({lo_lof:.2f}, {hi_lof:.2f})\n")
        f.write(f"Isolation Forest ROC:\n{mi_if:.2f}\\;({lo_if:.2f}, {hi_if:.2f})")
    