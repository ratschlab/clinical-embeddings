import hashlib
import json
import logging
import os
import pickle
from typing import Any, Dict

import gin
import numpy as np
import torch

from icu_benchmarks.common.constants import CIRC_FAILURE_NAME, RESP_FAILURE_NAME


def save_model(model, optimizer, epoch, save_file):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state


def load_model_state(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    logging.info("Loaded model and optimizer")


def save_config_file(log_dir):
    with open(os.path.join(log_dir, "train_config.gin"), "w") as f:
        f.write(gin.operative_config_str())


def get_modalities_list(modality_splitting):
    all_mods = pickle.load(open("files/dataset_stats/all_modsplits.pkl", "rb"))
    if modality_splitting in all_mods.keys():
        return all_mods[modality_splitting]

    elif "hirid_random" in modality_splitting:

        # expects `modality_splitting`: "hirid_random_{x}"
        # where {x} is an integer
        random_seed = int(modality_splitting.split("_")[-1])
        numpy_rng = np.random.RandomState(random_seed)

        path_to_cat_dict = "files/cat_dicts/cat_dict_hirid.pkl"
        with open(path_to_cat_dict, "rb") as file:
            cat_dict = pickle.load(file)

        # collect the type of variable found at each index in the raw data
        variable_type = [
            (col_ids, "cat", cat_dict[col_ids])
            if col_ids in cat_dict
            else (col_ids, "num", None)
            for col_ids in range(231)
        ]

        # collect only "real" variables; collapsing one-hot enc. categoricals
        i = 0
        variable_type_reduced = []
        while i < 231:
            variable_tuple = variable_type[i]
            variable_type_reduced.append(variable_tuple)
            if variable_tuple[1] == "num" or (
                variable_tuple[1] == "cat" and variable_tuple[2] == 1
            ):
                i += 1
            else:
                assert variable_tuple[1] == "cat" and variable_tuple[2] > 1
                i += variable_tuple[2]

        num_distinct_vars = len(variable_type_reduced)

        hirid_num_vars = 231
        ids = np.arange(num_distinct_vars)

        num_groups = int(numpy_rng.randint(low=3, high=10, size=1))
        logging.debug(f"[SPLITTING] create random split groups: {num_groups}")

        mean_size = num_distinct_vars // num_groups
        group_means = [mean_size for _ in range(num_groups)]
        group_sample = numpy_rng.dirichlet(group_means)
        group_sample = (group_sample * 231).astype(np.int32)

        diff = num_distinct_vars - sum(group_sample)
        group_sample[-1] += diff

        assert sum(group_sample) == num_distinct_vars

        group_split = np.array_split(ids, np.cumsum(group_sample))[:-1]
        groups_list = [list(a) for a in list(group_split)]

        assert len(groups_list) == num_groups
        assert sum([len(l) for l in groups_list]) == num_distinct_vars

        # convert group list back
        converted_groups_list = []
        for group in groups_list:
            new_group = []
            for variable_idx in group:
                variable = variable_type_reduced[variable_idx]
                if (
                    variable[2] is None or variable[2] == 1
                ):  # for numericals and binary cats.
                    new_group.append(variable[0])
                else:  # for one-hot encoded categoricals
                    new_group.extend([variable[0] + i for i in range(variable[2])])
            converted_groups_list.append(new_group)

        assert sum([len(l) for l in converted_groups_list]) == hirid_num_vars

        return converted_groups_list

    else:
        raise ValueError(f"Modality splitting: {modality_splitting} not supported")


def get_bindings_and_params(args):
    gin_bindings = []
    log_dir = args.logdir
    if args.num_class:
        num_class = args.num_class
        gin_bindings += ["NUM_CLASSES = " + str(num_class)]

    if args.task:
        task = "_".join(args.task[0].split("_")[:-1])
        if task == CIRC_FAILURE_NAME and args.aux_task:
            gin_bindings += ["AUX_IDX = " + str([2, 185, 228])]
            gin_bindings += ["NUM_AUX = " + str(3)]

        elif task == RESP_FAILURE_NAME and args.aux_task:
            gin_bindings += ["AUX_IDX = " + str([54, 145, 187, 188, 222])]
            gin_bindings += ["NUM_AUX = " + str(5)]

        else:
            gin_bindings += ["AUX_IDX = " + "None"]
            gin_bindings += ["NUM_AUX = " + str(0)]

    if args.res:
        res = args.res
        gin_bindings += ["RES = " + str(res)]
        log_dir = os.path.join(log_dir, "data-res_" + str(res))

    if args.res_lab:
        res_lab = args.res_lab
        gin_bindings += ["RES_LAB = " + str(res_lab)]
        log_dir = os.path.join(log_dir, "pre-res_" + str(res_lab))

    if args.horizon:
        if args.rs:
            horizon = args.horizon[np.random.randint(len(args.horizon))]
        else:
            horizon = args.horizon[0]
        gin_bindings += ["HORIZON  = " + str(horizon)]
        log_dir = log_dir + "_horizon_" + str(horizon)

    if args.embedding_depth:
        if args.rs:
            embedding_depth = args.embedding_depth[
                np.random.randint(len(args.embedding_depth))
            ]
        else:
            embedding_depth = args.embedding_depth[0]
        gin_bindings += ["EMB_DEPTH = " + str(embedding_depth)]

        log_dir = log_dir + "_embdepth_" + str(embedding_depth)

    if args.embedding_latent:
        if args.rs:
            embedding_latent = args.embedding_latent[
                np.random.randint(len(args.embedding_latent))
            ]
        else:
            embedding_latent = args.embedding_latent[0]
        gin_bindings += ["EMB_LATENT = " + str(embedding_latent)]

        log_dir = log_dir + "_emblatent_" + str(embedding_latent)

    # FTT params
    if args.self_attention_dim_emb:
        if args.rs:
            self_attention_dim_emb = args.self_attention_dim_emb[
                np.random.randint(len(args.self_attention_dim_emb))
            ]
        else:
            self_attention_dim_emb = args.self_attention_dim_emb[0]
        gin_bindings += ["SELF_ATTENTION_DIM_EMB = " + str(self_attention_dim_emb)]
        log_dir = log_dir + "_selfattdimemb_" + str(self_attention_dim_emb)

    if args.heads_emb:
        if args.rs:
            heads_emb = args.heads_emb[np.random.randint(len(args.heads_emb))]
        else:
            heads_emb = args.heads_emb[0]
        gin_bindings += ["HEADS_EMB  = " + str(heads_emb)]
        log_dir = log_dir + "_headsemb_" + str(heads_emb)

    if args.mult_emb:
        if args.rs:
            mult_emb = args.mult_emb[np.random.randint(len(args.mult_emb))]
        else:
            mult_emb = args.mult_emb[0]
        gin_bindings += ["MULT_EMB  = " + str(mult_emb)]
        log_dir = log_dir + "_multemb_" + str(mult_emb)

    # AGG params
    if args.self_attention_dim_agg:
        if args.rs:
            self_attention_dim_agg = args.self_attention_dim_agg[
                np.random.randint(len(args.self_attention_dim_agg))
            ]
        else:
            self_attention_dim_agg = args.self_attention_dim_agg[0]
        gin_bindings += ["SELF_ATTENTION_DIM_AGG = " + str(self_attention_dim_agg)]
        log_dir = log_dir + "_selfattdimagg_" + str(self_attention_dim_agg)

    if args.heads_agg:
        if args.rs:
            heads_agg = args.heads_agg[np.random.randint(len(args.heads_agg))]
        else:
            heads_agg = args.heads_agg[0]
        gin_bindings += ["HEADS_EMB  = " + str(heads_agg)]
        log_dir = log_dir + "_headsagg_" + str(heads_agg)

    if args.mult_agg:
        if args.rs:
            mult_agg = args.mult_agg[np.random.randint(len(args.mult_agg))]
        else:
            mult_agg = args.mult_agg[0]
        gin_bindings += ["MULT_AGG  = " + str(mult_agg)]
        log_dir = log_dir + "_multagg_" + str(mult_agg)

    if args.do_agg:
        if args.rs:
            do_agg = args.do_agg[np.random.randint(len(args.do_agg))]
        else:
            do_agg = args.do_agg[0]
        gin_bindings += ["DO_AGG  = " + str(do_agg)]
        log_dir = log_dir + "_doagg_" + str(do_agg)

    if args.do_att_agg:
        if args.rs:
            do_att_agg = args.do_att_agg[np.random.randint(len(args.do_att_agg))]
        else:
            do_att_agg = args.do_att_agg[0]
        gin_bindings += ["DO_ATT_AGG  = " + str(do_att_agg)]
        log_dir = log_dir + "_doattagg_" + str(do_att_agg)

    if args.modality_splitting:
        if args.rs:
            modality_splitting = args.modality_splitting[
                np.random.randint(len(args.modality_splitting))
            ]
        else:
            modality_splitting = args.modality_splitting[0]
        gin_bindings += [
            "MODALITY_SPLITTING = " + str(get_modalities_list(modality_splitting))
        ]

        log_dir = log_dir + "_modsplit_" + modality_splitting.replace("_", "")

    if args.embedding_merge:
        if args.rs:
            embedding_merge = args.embedding_merge[
                np.random.randint(len(args.embedding_merge))
            ]
        else:
            embedding_merge = args.embedding_merge[0]
        gin_bindings += ["EMB_MERGE = " + '"' + embedding_merge + '"']

        log_dir = log_dir + "_merge_" + embedding_merge.replace("_", "")

    if args.reg:
        if args.rs:
            reg = args.reg[np.random.randint(len(args.reg))]
            reg_weight = args.reg_weight[np.random.randint(len(args.reg_weight))]
        else:
            reg_weight = args.reg_weight[0]
            reg = args.reg[0]
        gin_bindings += ["REG_WEIGHT  = " + str(reg_weight)]
        gin_bindings += ["REG_TYPE = " + '"' + str(reg) + '"']

        log_dir = log_dir + "_" + str(reg) + "-reg_" + str(reg_weight)

    if args.batch_size:
        if args.rs:
            batch_size = args.batch_size[np.random.randint(len(args.batch_size))]
        else:
            batch_size = args.batch_size[0]
        gin_bindings += ["BS = " + str(batch_size)]
        log_dir = log_dir + "_bs_" + str(batch_size)

    if args.lr:
        if args.rs:
            lr = args.lr[np.random.randint(len(args.lr))]
        else:
            lr = args.lr[0]
        gin_bindings += ["LR = " + str(lr)]
        log_dir = log_dir + "_lr_" + str(lr)

    if args.lr_decay:
        if args.rs:
            lr_decay = args.lr_decay[np.random.randint(len(args.lr_decay))]
        else:
            lr_decay = args.lr_decay[0]
        gin_bindings += ["LR_DECAY = " + str(lr_decay)]
        log_dir = log_dir + "_lr-decay_" + str(lr_decay)

    if args.maxlen:
        maxlen = args.maxlen
        gin_bindings += ["MAXLEN = " + str(maxlen)]
        log_dir = log_dir + "_maxlen_" + str(maxlen)

    if args.aux_weight:
        if args.rs:
            aux_weight = args.aux_weight[np.random.randint(len(args.aux_weight))]
        else:
            aux_weight = args.aux_weight[0]
        gin_bindings += ["AUX_WEIGHT = " + str(aux_weight)]
        log_dir = log_dir + "_aux-weight_" + str(aux_weight)

    if args.aux_type:
        if args.rs:
            aux_type = args.aux_weight[np.random.randint(len(args.aux_type))]
        else:
            aux_type = args.aux_type[0]
        gin_bindings += ["AUX_TYPE = " + "'" + str(aux_type) + "'"]
        log_dir = log_dir + "_aux-type_" + str(aux_type)

    if args.emb:
        if args.rs:
            emb = args.emb[np.random.randint(len(args.emb))]
        else:
            emb = args.emb[0]
        gin_bindings += ["EMB  = " + str(emb)]
        log_dir = log_dir + "_emb_" + str(emb)

    if args.do:
        if args.rs:
            do = args.do[np.random.randint(len(args.do))]
        else:
            do = args.do[0]
        gin_bindings += ["DO  = " + str(do)]
        log_dir = log_dir + "_do_" + str(do)

    if args.do_att:
        if args.rs:
            do_att = args.do_att[np.random.randint(len(args.do_att))]
        else:
            do_att = args.do_att[0]
        gin_bindings += ["DO_ATT  = " + str(do_att)]
        log_dir = log_dir + "_do-att_" + str(do_att)

    if args.kernel:
        if args.rs:
            kernel = args.kernel[np.random.randint(len(args.kernel))]
        else:
            kernel = args.kernel[0]
        gin_bindings += ["KERNEL  = " + str(kernel)]
        log_dir = log_dir + "_kernel_" + str(kernel)

    if args.depth:
        if args.rs:
            depth = args.depth[np.random.randint(len(args.depth))]
        else:
            depth = args.depth[0]

        num_leaves = 2**depth
        gin_bindings += ["DEPTH  = " + str(depth)]
        gin_bindings += ["NUM_LEAVES  = " + str(num_leaves)]
        log_dir = log_dir + "_depth_" + str(depth)

    if args.heads:
        if args.rs:
            heads = args.heads[np.random.randint(len(args.heads))]
        else:
            heads = args.heads[0]
        gin_bindings += ["HEADS  = " + str(heads)]
        log_dir = log_dir + "_heads_" + str(heads)

    if args.latent:
        if args.rs:
            latent = args.latent[np.random.randint(len(args.latent))]
        else:
            latent = args.latent[0]
        gin_bindings += ["LATENT  = " + str(latent)]
        log_dir = log_dir + "_latent_" + str(latent)

    if args.hidden:
        if args.rs:
            hidden = args.hidden[np.random.randint(len(args.hidden))]
        else:
            hidden = args.hidden[0]
        gin_bindings += ["HIDDEN = " + str(hidden)]
        log_dir = log_dir + "_hidden_" + str(hidden)

    if args.subsample_data:
        if args.rs:
            subsample_data = args.subsample_data[
                np.random.randint(len(args.subsample_data))
            ]
        else:
            subsample_data = args.subsample_data[0]
        gin_bindings += ["SUBSAMPLE_DATA = " + str(subsample_data)]
        log_dir = log_dir + "_subsample-data_" + str(subsample_data)

    if args.subsample_feat:
        if args.rs:
            subsample_feat = args.subsample_feat[
                np.random.randint(len(args.subsample_feat))
            ]
        else:
            subsample_feat = args.subsample_data[0]
        gin_bindings += ["SUBSAMPLE_FEAT = " + str(subsample_feat)]
        log_dir = log_dir + "_subsample-feat_" + str(subsample_feat)

    if args.c_parameter:
        if args.rs:
            c_parameter = args.c_parameter[np.random.randint(len(args.c_parameter))]
        else:
            c_parameter = args.c_parameter[0]
        gin_bindings += ["C_PARAMETER = " + str(c_parameter)]
        log_dir = log_dir + "_c-parameter_" + str(c_parameter)

    if args.penalty:
        if args.rs:
            penalty = args.penalty[np.random.randint(len(args.penalty))]
        else:
            penalty = args.penalty[0]
        gin_bindings += ["PENALTY = " + "'" + str(penalty) + "'"]
        log_dir = log_dir + "_penalty_" + str(penalty)

    if args.loss_weight:
        if args.rs:
            loss_weight = args.loss_weight[np.random.randint(len(args.loss_weight))]
        else:
            loss_weight = args.loss_weight[0]
        if loss_weight == "None":
            gin_bindings += ["LOSS_WEIGHT = " + str(loss_weight)]
            log_dir = log_dir + "_loss-weight_no_weight"
        else:
            gin_bindings += ["LOSS_WEIGHT = " + "'" + str(loss_weight) + "'"]

    return gin_bindings, log_dir
