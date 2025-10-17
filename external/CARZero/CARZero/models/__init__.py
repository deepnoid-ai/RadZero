from . import (
    CARZero_model_dqn_wo_self_atten,
    CARZero_model_dqn_wo_self_atten_gl,
    CARZero_model_dqn_wo_self_atten_gl_mlp,
    CARZero_model_dqn_wo_self_atten_global,
    bert_model,
    cnn_backbones,
    dqn,
    dqn_wo_self_atten,
    dqn_wo_self_atten_mlp,
    fusion_module,
    mrm_pretrain_model,
    text_model,
    vision_model,
)

IMAGE_MODELS = {
    "pretrain_llm_dqn_wo_self_atten": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_global": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_gl": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl": vision_model.ImageEncoder,
}
