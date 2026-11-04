#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Conan front-end used by the CMake dependency provider.

The cmake-conan provider invokes ``${CONAN_COMMAND}``. CMakeLists points that
variable at this script so that, while CMake is configuring:

* the Artifactory Conan remote (``CONAN_REMOTE_URL``) is registered, with its
  credentials kept out of ``remotes.json`` and out of every log line, and
* the JSON graph printed by ``conan install`` is saved, so that a later
  ``triton-upload`` can push only the packages that were actually built.

Every other invocation is forwarded to the real ``conan`` unchanged.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, unquote, urlsplit, urlunsplit

REMOTE_NAME = "triton-artifactory"
REMOTE_ENV = REMOTE_NAME.upper().replace("-", "_")
GRAPH_ENV = "TRITON_CONAN_GRAPH_FILE"
REAL_CONAN_ENV = "TRITON_CONAN_EXECUTABLE"
UPLOAD_COMMAND = "triton-upload"


def real_conan():
    exe = os.environ.get(REAL_CONAN_ENV) or shutil.which("conan")
    if not exe:
        sys.exit("triton_conan: 'conan' not found; set " + REAL_CONAN_ENV)
    return exe


def run_conan(args, **kwargs):
    return subprocess.run([real_conan()] + list(args), **kwargs)


def split_remote_url(url):
    """Return (url without credentials, username, password)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += ":%d" % parts.port
    clean = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return clean, unquote(parts.username or ""), unquote(parts.password or "")


def configure_remote():
    """Register the remote and export its credentials to the environment.

    Returns the remote name, or None when CONAN_REMOTE_URL is not set (the
    build then uses only the remotes Conan already knows, e.g. conancenter).
    """
    url = os.environ.get("CONAN_REMOTE_URL")
    if not url:
        return None
    clean, user, password = split_remote_url(url)
    user = user or os.environ.get("ARTIFACTORY_USER", "")
    password = password or os.environ.get("ARTIFACTORY_TOKEN", "")
    if not clean.startswith("https://") or not user or not password:
        sys.exit(
            "triton_conan: CONAN_REMOTE_URL must be an https URL and "
            "credentials must be given (in the URL, or ARTIFACTORY_USER and "
            "ARTIFACTORY_TOKEN)"
        )
    # Credentials travel through the environment only; never in argv or on
    # disk. The remote URL stored by Conan carries no secret.
    os.environ["CONAN_LOGIN_USERNAME_" + REMOTE_ENV] = user
    os.environ["CONAN_PASSWORD_" + REMOTE_ENV] = password
    result = run_conan(
        ["remote", "add", "--force", "--index", "0", REMOTE_NAME, clean],
        stdout=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        sys.exit("triton_conan: failed to register remote " + REMOTE_NAME)
    return REMOTE_NAME


def passthrough(args):
    graph_file = os.environ.get(GRAPH_ENV)
    if args and args[0] == "install":
        configure_remote()
        if graph_file:
            # The provider parses stdout as JSON, so stdout must be forwarded
            # untouched after it is saved.
            result = run_conan(args, stdout=subprocess.PIPE)
            sys.stdout.buffer.write(result.stdout)
            sys.stdout.flush()
            if result.returncode == 0:
                with open(graph_file, "wb") as f:
                    f.write(result.stdout)
            return result.returncode
    return run_conan(args).returncode


def built_package_refs(graph_file, statuses=("Build",)):
    """Return 'name/ver#rrev:package_id#prev' for nodes built from source.

    The status comes from the graph printed by 'conan install'; Cache,
    Download, Skip and Missing nodes are never selected.
    """
    with open(graph_file) as f:
        nodes = json.load(f)["graph"]["nodes"]
    refs = []
    for node in nodes.values():
        if node.get("binary") not in statuses:
            continue
        ref, pid, prev = node.get("ref"), node.get("package_id"), node.get("prev")
        if not (ref and pid and prev):
            sys.exit("triton_conan: incomplete graph node for %s" % ref)
        item = "%s:%s#%s" % (ref, pid, prev)
        if item not in refs:
            refs.append(item)
    return refs


def merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def package_list(refs):
    """Conan package list (recipe + binary) for the given package refs."""
    merged = {}
    for ref in refs:
        result = run_conan(["list", ref, "--format=json"], stdout=subprocess.PIPE)
        if result.returncode != 0:
            sys.exit("triton_conan: 'conan list' failed for " + ref)
        listing = json.loads(result.stdout).get("Local Cache", {})
        if "error" in listing:
            sys.exit("triton_conan: %s is not in the local cache" % ref)
        merge(merged, {"Local Cache": listing})
    return merged


def upload(argv):
    graph_file = None
    remote = None
    dry_run = False
    it = iter(argv)
    for arg in it:
        if arg == "--graph":
            graph_file = next(it, None)
        elif arg == "--remote":
            remote = next(it, None)
        elif arg == "--dry-run":
            dry_run = True
        else:
            sys.exit("triton_conan: unknown argument " + quote(arg))
    if not graph_file or not os.path.isfile(graph_file):
        sys.exit(
            "triton_conan: no install graph found; run the CMake configure "
            "step first (expected %s)" % graph_file
        )
    remote = remote or configure_remote()
    if not remote:
        sys.exit("triton_conan: no remote: set CONAN_REMOTE_URL or --remote")
    if remote == REMOTE_NAME:
        configure_remote()

    refs = built_package_refs(graph_file)
    if not refs:
        print("triton_conan: nothing was built from source; nothing to upload")
        return 0
    print("triton_conan: uploading %d rebuilt package(s) to %s" % (len(refs), remote))
    for ref in refs:
        print("  " + ref)
    pkg_list = package_list(refs)
    with tempfile.TemporaryDirectory() as tmp:
        list_file = os.path.join(tmp, "pkglist.json")
        with open(list_file, "w") as f:
            json.dump(pkg_list, f)
        cmd = ["upload", "--list=" + list_file, "-r", remote, "--confirm"]
        if dry_run:
            cmd.append("--dry-run")
        return run_conan(cmd).returncode


def main(argv):
    if argv and argv[0] == UPLOAD_COMMAND:
        return upload(argv[1:])
    return passthrough(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
