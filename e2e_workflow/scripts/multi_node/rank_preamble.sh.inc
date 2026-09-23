set -e

# The NIC must be discovered where the process will actually run. Interface
# names differ between these boxes, and Ray's NodeManagerAddress is whatever the
# node registered with -- not necessarily the address the collectives should use.
CLUSTER_IF=
CLUSTER_IP=
while read -r _ ifname _ addr _; do
  ifname="${ifname%:}"
  ipaddr="${addr%%/*}"
  case "$ifname" in lo|docker0|br-*|veth*|virbr*) continue ;; esac
  if [ -z "$CLUSTER_IF" ]; then CLUSTER_IF="$ifname"; CLUSTER_IP="$ipaddr"; fi
  case "$ipaddr" in __CLUSTER_SUBNET__*) CLUSTER_IF="$ifname"; CLUSTER_IP="$ipaddr"; break ;; esac
done < <(ip -4 -o addr show)
if [ -z "$CLUSTER_IF" ]; then
  echo "!!! no usable IPv4 interface on this node" >&2
  exit 1
fi
export NCCL_SOCKET_IFNAME="$CLUSTER_IF"
export GLOO_SOCKET_IFNAME="$CLUSTER_IF"
export MASTER_ADDR=__MASTER__

# Both the log and the profile dir live on the shared mount, so rank1's output
# lands where the benchmark host can read it with no copy-back step. `mn.py launch`
# proves the mount really is shared before getting here.
mkdir -p "$(dirname __LOG__)" __PROFILE_DIR__
echo "mn-launch id=__LAUNCH_ID__ rank=__RANK__ if=$CLUSTER_IF ip=$CLUSTER_IP master=$MASTER_ADDR log=__LOG__"
