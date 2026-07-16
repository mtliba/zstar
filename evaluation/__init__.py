from .extract import extract_zstar
from .latent_analysis import umap_plot, tsne_plot, cluster_analysis, latent_interpolation
from .reconstruction import cross_modal_reconstruction
from .downstream import GraftLossPredictor, EGFRTrajectoryPredictor, train_downstream_head
from .comparison import compare_approaches
from .plots import (
    plot_training_curves,
    plot_embedding_pipeline_diagram,
    plot_latent_variance,
    plot_reconstruction_scatter,
    plot_roc_pr_curves,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_downstream_comparison,
    plot_missingness,
    generate_full_report,
)
from .survival import (
    kaplan_meier,
    median_survival_time,
    concordance_index,
    plot_kaplan_meier,
    plot_survival_overview,
    plot_followup_distribution,
    plot_km_by_zstar_risk,
    plot_landmark_diagnostic,
)
