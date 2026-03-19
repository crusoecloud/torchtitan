# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.qwen3 import Qwen3Model

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerQwen3Model(Qwen3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)

        # Cast the cos/sin RoPE cache to the mixed precision param dtype.
        # FSDP2 does this via cast_forward_inputs at the fully_shard boundary;
        # simple_fsdp has no such boundary, so we cast the buffer explicitly.
        # _mp_param_dtype is set by parallelize_qwen3 before init_weights runs.
        mp_dtype = getattr(self, "_mp_param_dtype", None)
        if mp_dtype is not None and self.freqs_cis.is_floating_point():
            self.freqs_cis = self.freqs_cis.to(mp_dtype)
