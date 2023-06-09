from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    # models = ["VGG_topK_50","VGG_topK_40","VGG_topK_30", "VGG_topK_20","VGG_topK_10","VGG_topK_5","VGG_normal"]
    # models = ["AlexNet_topK_50","AlexNet_topK_40","AlexNet_topK_30", "AlexNet_topK_20","AlexNet_topK_10","AlexNet_topK_5","AlexNet_normal"]
    # models = ["AlexNet_topK_10","VGG_topK_5","AlexNet_normal","VGG_normal","vit_large_patch16_224","clip"]
    models = ["AlexNet_topK_10","AlexNet_topK_5_Revert","AlexNet_topK_10_Revert","AlexNet_topK_20_Revert","AlexNet_topK_30_Revert","AlexNet_topK_40_Revert","AlexNet_topK_50_Revert","AlexNet_normal"]
    datasets = ["cue-conflict"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types=["shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
