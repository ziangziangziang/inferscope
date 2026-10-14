# run.sh, recovered from the GPU box

`run.sh` lived only at `/workspace/inferscope/run.sh` on the GPU box and was
never in the repository, so it was one rebuild away from being lost. It is
recovered here verbatim as `run.sh.host`.

It is the environment the app runs in: the egress proxy neither the server nor
a browser on this network can do without, the model paths, and the ports. It
holds no credentials — only hostnames and paths — so committing it leaks
nothing.

Deploy:

    scp -P 31699 ops/run.sh.host root@36.150.116.206:/workspace/inferscope/run.sh
    chmod +x /workspace/inferscope/run.sh

If the box's paths ever change, this file is now the place that knows it.

The proxy hostname ends in `lablab`, not `lab`. That is not a typo and it cost
an hour once.
