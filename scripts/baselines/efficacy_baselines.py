import argparse
import json
import logging
import os
from typing import Literal

from src import data, editors, functional, metrics, models, operators, sweeps
from src.data import RelationSample
from src.hparams import RelationHParams
from src.utils import dataclasses_utils, experiment_utils, logging_utils
from src.utils.sweep_utils import read_sweep_results, relation_from_dict

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def filter_not_in_train_samples(
    sample: RelationSample, train_samples: list[RelationSample]
) -> bool:
    for train_sample in train_samples:
        if (
            sample.subject == train_sample.subject
            and sample.object == train_sample.object
        ):
            return False
    return True


BASELINE_EDITOR_TYPES = {
    "hidden_baseline": editors.InsertSubjectHEditor,
    "embed_baseline": editors.InsertObjectEmbeddingEditor,
    "low_rank_pinv": editors.LowRankPInvEditor,
    "hidden_baseline_z": editors.InsertObjectZEditor,
}

from src.utils.sweep_utils import (
    EfficacyBaselineLayerResult,
    EfficacyBaselineRelationResult,
    EfficacyBaselineResults,
    EfficacyBaselineTrialResult,
    EfficacyTestPair,
)


def run_causality_baselines(
    model_name: Literal["gptj", "gpt2-xl", "llama-13b"] = "gptj",
    sweep_results_dir: str = "results/sweep",
    save_dir: str = "results/efficacy_baselines/",
    device: str | None = None,
    batch_size: int = sweeps.DEFAULT_BATCH_SIZE,
    experiment_name: str | None = None,
    rel_names: list[str] | None = None,
    use_hparams: bool = False,
    n_trials: int = 3,
    n_train_samples: int = functional.DEFAULT_N_ICL_LM,
) -> None:
    save_dir = f"{save_dir}/{model_name}"
    if experiment_name is not None:
        save_dir = f"{save_dir}/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    device = models.determine_default_device(device)
    mt = models.load_model(name=model_name, device=device)
    dataset = data.load_dataset()

    if use_hparams:
        relation_names = rel_names or [relation.name for relation in dataset.relations]
        sweep_results = {relation_name: None for relation_name in relation_names}
    else:
        sweep_results_dir = f"{sweep_results_dir}/{model_name}"
        sweep_results = read_sweep_results(sweep_results_dir, relation_names=rel_names)

    logger.info("found %d relations", len(sweep_results))
    logger.info(json.dumps(list(sweep_results.keys()), indent=4))

    all_results: list[EfficacyBaselineRelationResult] = []
    for relation_name, sweep_result in tqdm(sweep_results.items()):
        if rel_names is not None and relation_name not in rel_names:
            logger.info("skipping %s", relation_name)
            continue
        logger.info("relation: %s", relation_name)

        if use_hparams:
            hparams_model_name = "llama" if "llama" in model_name else model_name
            relation_hparams = RelationHParams.from_relation(
                hparams_model_name, relation_name
            )
            if relation_hparams is None:
                logger.info(f"skipping {relation_name}, no hparams file")
                continue
            if relation_hparams.rank is None:
                logger.info(f"skipping {relation_name}, no rank in hparams")
                continue

            relation = dataset.filter(relation_names=[relation_name])[0]
            layer_from_hparams = (
                relation_hparams.h_layer_edit or relation_hparams.h_layer
            )
            rank_from_hparams = relation_hparams.rank
            beta_from_hparams = relation_hparams.beta
            n_relation_trials = n_trials
            by_layer = None
            relation_results = None
            logger.info(
                f"{relation_name} | h_layer_edit={layer_from_hparams} | "
                f"rank={rank_from_hparams} | beta={beta_from_hparams}"
            )
        else:
            assert sweep_result is not None
            relation_results = relation_from_dict(sweep_result)
            if len(relation_results.trials) < 3:
                logger.info(f"skipping {relation_name}, not enough trials")
                continue
            relation = dataset.filter(relation_names=[relation_results.relation_name])[
                0
            ]
            by_layer = relation_results.by_layer()
            n_relation_trials = len(relation_results.trials)

        efficacy_baseline_relation_result = EfficacyBaselineRelationResult(
            relation_name=relation_name,
            trials=[],
        )
        for n_trial in range(n_relation_trials):
            logger.info(f"trial: {n_trial+1}/{n_relation_trials}")
            if use_hparams:
                prompt_template = relation.prompt_templates[0]
                train_relation, _ = relation.set(
                    prompt_templates=[prompt_template]
                ).split(n_train_samples)
                train_samples = train_relation.samples
                h_layers = [layer_from_hparams]
            else:
                assert relation_results is not None
                trial_results = relation_results.trials[n_trial]
                prompt_template = trial_results.prompt_template
                train_samples = trial_results.train_samples
                efficacy_trials = trial_results.efficacy_trials
                h_layers = [layer.layer for layer in trial_results.layers]

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=train_samples,
            )
            test_samples = [
                sample
                for sample in relation.samples
                if filter_not_in_train_samples(sample, train_samples)
            ]
            test_relation = relation.set(samples=test_samples)

            test_relation = (
                functional.filter_relation_samples_based_on_provided_fewshots(
                    mt=mt,
                    test_relation=test_relation,
                    prompt_template=icl_prompt,
                    subj_token_filter="all",
                )
            )
            logger.info(
                f"filtered test relation to {len(test_relation.samples)} samples"
            )

            if len(test_relation.samples) <= len(train_samples):
                logger.warning(
                    f"Not enough samples ( < {len(train_samples)}) to test for faithfulness and efficacy."
                )
                break  # only consider relations that have enough number of known test samples

            if use_hparams:
                test_targets = functional.random_edit_targets(test_relation.samples)
                efficacy_trials = [
                    EfficacyTestPair(source=source, target=target)
                    for source, target in test_targets.items()
                ]
                if len(efficacy_trials) == 0:
                    logger.info(
                        f"skipping trial for {relation_name}, no efficacy pairs"
                    )
                    continue

            logger.info("precomputing test hs and zs...")
            hs_by_subj, zs_by_subj = functional.compute_hs_and_zs(
                mt=mt,
                prompt_template=prompt_template,
                subjects=[x.subject for x in relation.samples],
                h_layer=h_layers,
                z_layer=-1,
                batch_size=batch_size,
                examples=train_samples,
            )

            layerwise_baseline_results: list[EfficacyBaselineLayerResult] = []
            for layer_no in h_layers:
                if use_hparams:
                    rank = rank_from_hparams
                    efficacy_mean = 0.0
                    logger.info(f"layer: {layer_no}, {rank=}")
                else:
                    assert by_layer is not None
                    efficacy = by_layer[layer_no].efficacy
                    # rank = int(np.floor(by_layer[layer_no].rank.mean)) # use the mean rank for each layer

                    # use different best ranks for each trial
                    rank = by_layer[layer_no].rank.values[n_trial]  # type: ignore
                    efficacy_mean = efficacy.mean
                    logger.info(
                        f"layer: {layer_no}, efficacy = {efficacy.mean} +/- {efficacy.stderr}, {rank=}"
                    )

                estimator = operators.JacobianIclMeanEstimator(
                    mt=mt,
                    h_layer=layer_no,
                    beta=beta_from_hparams if use_hparams else None,
                )
                operator = estimator(
                    relation.set(
                        samples=train_samples,
                        prompt_templates=[prompt_template],
                    )
                )
                assert operator.weight is not None
                svd = torch.svd(operator.weight.float())

                baseline_results: dict[str, float] = {}

                for editor_type in BASELINE_EDITOR_TYPES:
                    editor_class = BASELINE_EDITOR_TYPES[editor_type]
                    editor: editors.Editor = (
                        dataclasses_utils.create_with_optional_kwargs(
                            editor_class,
                            h_layer=layer_no,
                            rank=rank,
                            lre=operator,
                            svd=svd,
                            prompt_template=operator.prompt_template,
                            mt=mt,
                            n_samples=1,
                            n_new_tokens=1,
                        )
                    )

                    pred_objects = []
                    target_objects = []
                    for efficacy_test_pair in efficacy_trials:
                        original = efficacy_test_pair.source
                        target = efficacy_test_pair.target
                        target_objects.append(target.object)

                        z_original = zs_by_subj[original.subject]
                        z_target = zs_by_subj[target.subject]
                        # h_original = hs_by_subj[original.subject][layer_no]
                        # h_target = hs_by_subj[target.subject][layer_no]

                        if editor.expects() == "object":
                            result = dataclasses_utils.call_with_optional_kwargs(
                                editor.__call__,
                                subject=original.subject,
                                target=target.object,
                                z_original=z_original,
                            )
                        else:
                            assert editor.expects() == "subject"
                            result = dataclasses_utils.call_with_optional_kwargs(
                                editor.__call__,
                                subject=original.subject,
                                target=target.subject,
                                z_original=z_original,
                                z_target=z_target,
                            )
                        pred_objects.append([result.predicted_tokens[0].token])

                        pred = str(result.predicted_tokens[0])
                        logger.debug(
                            f"editing: {original.subject=} | {target.subject=} -> {target.object=} |>> {pred=}"
                        )

                    [baseline_efficacy] = metrics.recall(pred_objects, target_objects)
                    logger.info(
                        f"editing finished: {layer_no=} {rank=} {editor_type}={baseline_efficacy:.2f}"
                    )
                    logger.debug("--------------------------------------------------")
                    baseline_results[editor_type] = baseline_efficacy

                if use_hparams:
                    efficacy_mean = baseline_results["low_rank_pinv"]
                assert isinstance(rank, int)
                layerwise_baseline_results.append(
                    EfficacyBaselineLayerResult(
                        layer=layer_no,
                        efficacy=efficacy_mean,
                        rank=rank,
                        results=baseline_results,
                    )
                )
            efficacy_baseline_relation_result.trials.append(
                EfficacyBaselineTrialResult(
                    train_samples=train_samples,
                    prompt_template=prompt_template,
                    layerwise_baseline_results=layerwise_baseline_results,
                )
            )

            experiment_utils.save_results_file(
                results_dir=save_dir,
                name=relation_name,
                results=efficacy_baseline_relation_result,
            )

        if use_hparams:
            relation_dir = os.path.join(
                save_dir, relation_name.replace(" ", "_").replace("'", "")
            )
            os.makedirs(relation_dir, exist_ok=True)
            relation_results_file = os.path.join(relation_dir, "results_all.json")
            logger.info(f"saving {relation_name} results to {relation_results_file}")
            with open(relation_results_file, "w") as handle:
                handle.write(
                    EfficacyBaselineResults(
                        relations=[efficacy_baseline_relation_result]
                    ).to_json(indent=4)
                )

        all_results.append(efficacy_baseline_relation_result)

    efficacy_baseline_results = EfficacyBaselineResults(relations=all_results)
    results_file = f"{save_dir}/results_all.json"
    logger.info(f"saving all results to {results_file}")
    with open(results_file, "w") as handle:
        handle.write(efficacy_baseline_results.to_json(indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="calculate layerwise efficacy baselines"
    )

    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=sweeps.DEFAULT_BATCH_SIZE,
        help="max batch size for lm",
    )

    parser.add_argument(
        "--sweep-results-dir",
        type=str,
        default="results/sweep",
        help="directory to find sweep results",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/efficacy_baselines/",
        help="directory to find sweep results",
    )
    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )
    parser.add_argument(
        "--use-hparams",
        action="store_true",
        default=False,
        help="use committed hparams/<model>/<relation>.json instead of sweep results",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="number of random trials to run when --use-hparams is set",
    )
    parser.add_argument(
        "--n-training",
        type=int,
        default=functional.DEFAULT_N_ICL_LM,
        help="number of training samples to use when --use-hparams is set",
    )

    args = parser.parse_args()

    logging_utils.configure(args)

    logger.info(args)

    run_causality_baselines(
        model_name=args.model,
        sweep_results_dir=args.sweep_results_dir,
        save_dir=args.save_dir,
        device=args.device,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        rel_names=args.rel_names,
        use_hparams=args.use_hparams,
        n_trials=args.n_trials,
        n_train_samples=args.n_training,
    )
