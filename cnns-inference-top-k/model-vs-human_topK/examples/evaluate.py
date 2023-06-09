from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template
from plotting_definition import plotting_definition_template_VGG
from plotting_definition import plotting_definition_template_AlexNet
from plotting_definition import plotting_definition_template_General
import argparse


def run_evaluation(plot):
    if plot=="VGG":
        models = ["VGG_topK_50","VGG_topK_40","VGG_topK_30", "VGG_topK_20","VGG_topK_10","VGG_topK_5","VGG_normal"]
    elif plot=="AlexNet":
        models = ["AlexNet_topK_50","AlexNet_topK_40","AlexNet_topK_30", "AlexNet_topK_20","AlexNet_topK_10","AlexNet_topK_5","AlexNet_normal"]
    elif plot=="General":
        models = ["AlexNet_topK_10","AlexNet_normal","VGG_topK_5","VGG_normal","vit_large_patch16_224","clip"]
    else:
        models = ["AlexNet_topK_50","AlexNet_topK_40","AlexNet_topK_30", "AlexNet_topK_20","AlexNet_topK_10","AlexNet_topK_5","AlexNet_normal","VGG_topK_50","VGG_topK_40","VGG_topK_30", "VGG_topK_20","VGG_topK_10","VGG_topK_5","VGG_normal","vit_large_patch16_224","clip"]
    datasets = ["cue-conflict"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting(plot):
    plot_types=["shape-bias"]
    if plot=="VGG":
        plotting_def = plotting_definition_template_VGG
    elif plot=="AlexNet":
        plotting_def = plotting_definition_template_AlexNet
    elif plot=="General":
        plotting_def = plotting_definition_template_General
    else:
        plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--plot", type=str)
    args = parser.parse_args()
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation(args.plot)
    # 2. plot the evaluation results
    run_plotting(args.plot)
