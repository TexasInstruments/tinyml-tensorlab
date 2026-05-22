#!/usr/bin/env bash

# Copyright (c) 2023-2024, Texas Instruments Incorporated
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Register named remotes (no-op if already exists)
git remote add tinyml-tinyverse       ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-tinyverse.git       2>/dev/null || true
git remote add tinyml-modeloptimization ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-modeloptimization.git 2>/dev/null || true
git remote add tinyml-modelmaker      ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-modelmaker.git      2>/dev/null || true
git remote add tinyml-modelzoo        ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-modelzoo.git        2>/dev/null || true
# git remote add tinyml-docs          ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-docs.git            2>/dev/null || true

# Shallow fetch remotes sequentially (parallel shallow fetches conflict on .git/shallow.lock)
git fetch --depth 1 tinyml-tinyverse        main
git fetch --depth 1 tinyml-modeloptimization main
git fetch --depth 1 tinyml-modelmaker       main
git fetch --depth 1 tinyml-modelzoo         main
# git fetch --depth 1 tinyml-docs           main

# Merge into subtrees sequentially (git index cannot handle parallel merges)
git subtree merge --prefix tinyml-tinyverse        tinyml-tinyverse/main        --squash
git subtree merge --prefix tinyml-modeloptimization tinyml-modeloptimization/main --squash
git subtree merge --prefix tinyml-modelmaker       tinyml-modelmaker/main       --squash
git subtree merge --prefix tinyml-modelzoo         tinyml-modelzoo/main         --squash
# git subtree merge --prefix tinyml-docs           tinyml-docs/main             --squash

git submodule update --remote
